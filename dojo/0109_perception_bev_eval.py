"""Inspectable camera/LiDAR pipeline: every intermediate is part of the answer."""
# %%
import math
from decimal import Decimal, localcontext, ROUND_FLOOR
import sys
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values, evaluate

# %%
# Local snapshots of the five prerequisite algorithms. These definitions are
# ordinary script cells, never runtime imports of another numbered exercise.
# Teaching composition: modest finite data, baseline/followup.
# Vectorized composition: float32/64 production, widened stable fusion/scatter.
# Chunked composition: same outputs, bounded point intermediates for projection,
# sampling and view reduction. All intermediates/output and backward state
# still grow with N; dense BEV remains O(B*(F+C+1)*Y*X).
# %%
def _validate_projection(points, K, T_camera_from_ego, image_hw, point_valid,
                         camera_valid, min_depth, chunk_size):
    values = (points,K,T_camera_from_ego,image_hw,point_valid,camera_valid)
    if not all(isinstance(x,torch.Tensor) for x in values):
        raise ValueError('inputs must be tensors')
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[0] < 1:
        raise ValueError('points must be [B,N,3], B positive')
    B,N,_ = points.shape
    if K.ndim != 4 or K.shape[0] != B or K.shape[1] < 1 or K.shape[2:] != (3,3):
        raise ValueError('K must be [B,M,3,3], M positive')
    M = K.shape[1]
    if T_camera_from_ego.shape != (B,M,4,4) or image_hw.shape != (B,M,2) or point_valid.shape != (B,N) or camera_valid.shape != (B,M):
        raise ValueError('transform, size, or mask shape mismatch')
    if points.dtype not in (torch.float32,torch.float64) or any(x.dtype != points.dtype for x in (K,T_camera_from_ego)):
        raise ValueError('floating geometry must share float32/float64 dtype')
    if any(x.device != points.device for x in values):
        raise ValueError('all inputs must share a device')
    if point_valid.dtype != torch.bool or camera_valid.dtype != torch.bool or image_hw.dtype != torch.int64:
        raise ValueError('masks must be bool; image_hw must be int64')
    if not isinstance(min_depth,(int,float)) or isinstance(min_depth,bool) or not math.isfinite(min_depth) or min_depth <= 0:
        raise ValueError('min_depth must be finite and positive')
    if chunk_size is not None and (not isinstance(chunk_size,int) or isinstance(chunk_size,bool) or chunk_size <= 0):
        raise ValueError('chunk_size must be a positive integer')
    if not (torch.isfinite(K).all() and torch.isfinite(T_camera_from_ego).all() and (image_hw > 0).all()):
        raise ValueError('finite calibration and positive image sizes required')
    if not torch.isfinite(points[point_valid]).all():
        raise ValueError('valid points must be finite')
    # Small structure tolerance permits ordinary floating poses and local
    # finite-difference probes. It does not project malformed poses onto SE(3).
    tol = 1e-5
    bottom_K = K.new_tensor([0.,0.,1.]).expand(B,M,3)
    bottom_T = K.new_tensor([0.,0.,0.,1.]).expand(B,M,4)
    R = T_camera_from_ego[...,:3,:3]
    eye = torch.eye(3,dtype=points.dtype,device=points.device).expand(B,M,3,3)
    if not (torch.allclose(K[...,2,:],bottom_K,atol=tol,rtol=0)
            and (K[...,0,0] > 0).all() and (K[...,1,1] > 0).all()
            and (torch.linalg.det(K) != 0).all()
            and torch.allclose(T_camera_from_ego[...,3,:],bottom_T,atol=tol,rtol=0)
            and torch.allclose(R.transpose(-1,-2)@R,eye,atol=tol,rtol=0)
            and torch.allclose(torch.linalg.det(R),K.new_ones(B,M),atol=tol,rtol=0)):
        raise ValueError('K must be pinhole/invertible and T must be rigid homogeneous')
    return B,M,N

# %%
def project_per_camera(points, K, T_camera_from_ego, image_hw, point_valid,
                       camera_valid, min_depth=0.001, chunk_size=None):
    B,M,N = _validate_projection(points,K,T_camera_from_ego,image_hw,point_valid,camera_valid,min_depth,chunk_size)
    clean = torch.where(point_valid[...,None],points,torch.zeros_like(points))
    all_uv,all_depth,all_valid = [],[],[]
    for m in range(M):
        T = T_camera_from_ego[:,m]
        xyz = clean @ T[:,:3,:3].transpose(-1,-2) + T[:,None,:3,3]
        eligible = point_valid & camera_valid[:,m,None] & (xyz[...,2] > min_depth) & torch.isfinite(xyz).all(-1)
        safe = torch.where(eligible[...,None],xyz,xyz.new_tensor([0.,0.,1.]))
        h = safe @ K[:,m].transpose(-1,-2)
        divisor = torch.where(eligible,h[...,2],torch.ones_like(h[...,2]))
        coordinates = h[...,:2]/divisor[...,None]
        valid = eligible & torch.isfinite(coordinates).all(-1) & (coordinates[...,0] >= 0) & (coordinates[...,0] <= image_hw[:,m,1,None]-1) & (coordinates[...,1] >= 0) & (coordinates[...,1] <= image_hw[:,m,0,None]-1)
        all_uv.append(torch.where(valid[...,None],coordinates,torch.zeros_like(coordinates)))
        all_depth.append(torch.where(valid,xyz[...,2],torch.zeros_like(xyz[...,2])))
        all_valid.append(valid)
    return torch.stack(all_uv,1),torch.stack(all_depth,1),torch.stack(all_valid,1)

# %%
def project_broadcast(points, K, T_camera_from_ego, image_hw, point_valid,
                      camera_valid, min_depth=0.001, chunk_size=None):
    _validate_projection(points,K,T_camera_from_ego,image_hw,point_valid,camera_valid,min_depth,chunk_size)
    clean = torch.where(point_valid[...,None],points,torch.zeros_like(points))
    xyz = clean[:,None] @ T_camera_from_ego[...,:3,:3].transpose(-1,-2) + T_camera_from_ego[...,None,:3,3]
    eligible = point_valid[:,None] & camera_valid[...,None] & (xyz[...,2] > min_depth) & torch.isfinite(xyz).all(-1)
    safe = torch.where(eligible[...,None],xyz,xyz.new_tensor([0.,0.,1.]))
    h = safe @ K.transpose(-1,-2)
    divisor = torch.where(eligible,h[...,2],torch.ones_like(h[...,2]))
    uv = h[...,:2]/divisor[...,None]
    valid = eligible & torch.isfinite(uv).all(-1) & (uv[...,0] >= 0) & (uv[...,0] <= image_hw[...,1,None]-1) & (uv[...,1] >= 0) & (uv[...,1] <= image_hw[...,0,None]-1)
    return (torch.where(valid[...,None],uv,torch.zeros_like(uv)),
            torch.where(valid,xyz[...,2],torch.zeros_like(xyz[...,2])),valid)

# %%
def project_chunked(points, K, T_camera_from_ego, image_hw, point_valid,
                    camera_valid, min_depth=0.001, chunk_size=None):
    B,M,N = _validate_projection(points,K,T_camera_from_ego,image_hw,point_valid,camera_valid,min_depth,chunk_size)
    if N == 0:
        # Keep zero-size outputs connected to geometry for zero gradients.
        zero = points.sum()*0 + K.sum()*0 + T_camera_from_ego.sum()*0
        return points.new_empty(B,M,0,2)+zero,points.new_empty(B,M,0)+zero,torch.empty(B,M,0,dtype=torch.bool,device=points.device)
    size = N if chunk_size is None else chunk_size
    uv_parts,depth_parts,valid_parts = [],[],[]
    for start in range(0,N,size):
        mask = point_valid[:,start:start+size]
        p = points[:,start:start+size]
        clean = torch.where(mask[...,None],p,torch.zeros_like(p))
        xyz = clean[:,None] @ T_camera_from_ego[...,:3,:3].transpose(-1,-2) + T_camera_from_ego[...,None,:3,3]
        eligible = mask[:,None] & camera_valid[...,None] & (xyz[...,2] > min_depth) & torch.isfinite(xyz).all(-1)
        safe = torch.where(eligible[...,None],xyz,xyz.new_tensor([0.,0.,1.]))
        h = safe @ K.transpose(-1,-2)
        divisor = torch.where(eligible,h[...,2],torch.ones_like(h[...,2]))
        uv = h[...,:2]/divisor[...,None]
        valid = eligible & torch.isfinite(uv).all(-1) & (uv[...,0] >= 0) & (uv[...,0] <= image_hw[...,1,None]-1) & (uv[...,1] >= 0) & (uv[...,1] <= image_hw[...,0,None]-1)
        uv_parts.append(torch.where(valid[...,None],uv,torch.zeros_like(uv)))
        depth_parts.append(torch.where(valid,xyz[...,2],torch.zeros_like(xyz[...,2])))
        valid_parts.append(valid)
    return torch.cat(uv_parts,2),torch.cat(depth_parts,2),torch.cat(valid_parts,2)

# %%
def validate_sampling(features, uv, valid, feature_from_image, chunk_size):
    tensors = (features, uv, valid, feature_from_image)
    if not all(isinstance(t,torch.Tensor) for t in tensors):
        raise ValueError('All inputs must be tensors')
    if features.ndim != 5 or any(d <= 0 for d in features.shape):
        raise ValueError('features must have positive [B,M,C,Hf,Wf] dimensions')
    B,M = features.shape[:2]
    if uv.ndim != 4 or uv.shape[:2] != (B,M) or uv.shape[-1] != 2:
        raise ValueError('uv must be [B,M,N,2]')
    if valid.shape != uv.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError('valid must be bool [B,M,N]')
    if feature_from_image.shape != (B,M,3,3):
        raise ValueError('feature_from_image must be [B,M,3,3]')
    if features.dtype not in (torch.float32,torch.float64):
        raise ValueError('Core uses float32/float64')
    if any(t.device != features.device for t in tensors) or any(t.dtype != features.dtype for t in (uv,feature_from_image)):
        raise ValueError('Floating tensors share dtype and all tensors share device')
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError('chunk_size must be a positive Python integer or None')
    a = feature_from_image
    bottom = a.new_tensor([0.,0.,1.]).expand(B,M,3)
    if not torch.isfinite(a).all() or not torch.equal(a[:,:,2],bottom):
        raise ValueError('Finite affine metadata must have bottom row [0,0,1]')
    if (torch.linalg.det(a[:,:,:2,:2]) == 0).any():
        raise ValueError('Image-to-feature affine must be invertible')
    if not torch.isfinite(uv[valid]).all():
        raise ValueError('Coordinates marked valid must be finite')
    active = valid.any(dim=-1)
    if not torch.isfinite(features[active]).all():
        raise ValueError('Feature maps for cameras with incoming support must be finite')
    return active

# %%
def sample_features_vectorized(features, uv, valid, feature_from_image, chunk_size=None):
    active = validate_sampling(features,uv,valid,feature_from_image,chunk_size)
    B,M,C,H,W = features.shape
    safe_f = torch.where(active[:,:,None,None,None],features,0.)
    safe_uv = torch.where(valid[...,None],uv,0.)
    xy = torch.matmul(safe_uv,feature_from_image[:,:,:2,:2].transpose(-1,-2)) + feature_from_image[:,:,:2,2].unsqueeze(2)
    x,y = xy.unbind(-1)
    supported = valid & torch.isfinite(xy).all(-1) & (x>=0) & (x<=W-1) & (y>=0) & (y<=H-1)
    x = torch.where(supported,x,0.); y = torch.where(supported,y,0.)
    x0,y0 = x.floor().long(),y.floor().long()
    x1,y1 = (x0+1).clamp(max=W-1),(y0+1).clamp(max=H-1)
    dx,dy = (x-x0).unsqueeze(-1),(y-y0).unsqueeze(-1)
    flat = safe_f.reshape(B,M,C,H*W)
    def corner(xx,yy):
        index = (yy*W+xx).unsqueeze(2).expand(B,M,C,uv.shape[2])
        return flat.gather(-1,index).transpose(-1,-2)
    samples = (1-dy)*((1-dx)*corner(x0,y0)+dx*corner(x1,y0)) + dy*((1-dx)*corner(x0,y1)+dx*corner(x1,y1))
    return torch.where(supported[...,None],samples,0.),supported

# %%
def sample_features_chunked(features, uv, valid, feature_from_image, chunk_size=None):
    active = validate_sampling(features,uv,valid,feature_from_image,chunk_size)
    B,M,C,H,W = features.shape
    N = uv.shape[2]
    safe_f = torch.where(active[:,:,None,None,None],features,0.)
    safe_uv = torch.where(valid[...,None],uv,0.)
    zero = safe_f.reshape(-1)[:1].sum()*0 + safe_uv.reshape(-1)[:1].sum()*0 + feature_from_image.reshape(-1)[:1].sum()*0
    if N == 0:
        return features.new_empty(B,M,0,C)+zero,torch.empty(B,M,0,dtype=torch.bool,device=features.device)
    flat = safe_f.reshape(B,M,C,H*W)
    outputs,masks = [],[]
    for start in range(0,N,chunk_size or N):
        coords = safe_uv[:,:,start:start+(chunk_size or N)]
        eligible = valid[:,:,start:start+(chunk_size or N)]
        xy = torch.matmul(coords,feature_from_image[:,:,:2,:2].transpose(-1,-2)) + feature_from_image[:,:,:2,2].unsqueeze(2)
        x,y = xy.unbind(-1)
        supported = eligible & torch.isfinite(xy).all(-1) & (x>=0) & (x<=W-1) & (y>=0) & (y<=H-1)
        x = torch.where(supported,x,0.); y = torch.where(supported,y,0.)
        x0,y0 = x.floor().long(),y.floor().long()
        x1,y1 = (x0+1).clamp(max=W-1),(y0+1).clamp(max=H-1)
        dx,dy = (x-x0).unsqueeze(-1),(y-y0).unsqueeze(-1)
        def corner(xx,yy):
            index = (yy*W+xx).unsqueeze(2).expand(B,M,C,coords.shape[2])
            return flat.gather(-1,index).transpose(-1,-2)
        values = (1-dy)*((1-dx)*corner(x0,y0)+dx*corner(x1,y0)) + dy*((1-dx)*corner(x0,y1)+dx*corner(x1,y1))
        outputs.append(torch.where(supported[...,None],values,0.)); masks.append(supported)
    return torch.cat(outputs,dim=2),torch.cat(masks,dim=2)

# %%
def _validate(samples, valid, confidence, chunk_size):
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
        raise ValueError('samples must be [B,M,N,C]')
    b, m, n, c = samples.shape
    if min(b, m, c) < 1 or samples.dtype not in (torch.float32, torch.float64):
        raise ValueError('positive B/M/C and float32/float64 required')
    if (not isinstance(valid, torch.Tensor) or valid.shape != (b,m,n)
            or valid.dtype != torch.bool or valid.device != samples.device):
        raise ValueError('valid must be boolean [B,M,N] on sample device')
    if (not isinstance(confidence, torch.Tensor) or confidence.shape != valid.shape
            or confidence.dtype != samples.dtype or confidence.device != samples.device):
        raise ValueError('confidence must share sample dtype/device and valid shape')
    if not torch.all(torch.isfinite(confidence[valid]) & (confidence[valid] >= 0)):
        raise ValueError('supported confidence must be finite and nonnegative')
    eligible = valid & (confidence > 0)
    if not torch.isfinite(samples[eligible]).all():
        raise ValueError('positive-weight supported samples must be finite')
    if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError('chunk_size must be a positive integer')

# %%
def fuse_transparent(samples, valid, confidence, chunk_size=None):
    _validate(samples, valid, confidence, chunk_size)
    eligible = valid & (confidence > 0)
    weights = torch.where(eligible, confidence, 0)
    values = torch.where(eligible[...,None], samples, 0)
    mass = weights.sum(dim=1)
    supported = mass > 0
    denominator = torch.where(supported, mass, 1)
    fused = (values * weights[...,None]).sum(dim=1) / denominator[...,None]
    return fused, supported

# %%
def fuse_stable(samples, valid, confidence, chunk_size=None):
    _validate(samples, valid, confidence, chunk_size)
    eligible = valid & (confidence > 0)
    weights = torch.where(eligible, confidence, 0).to(torch.float64)
    values = torch.where(eligible[...,None], samples, 0).to(torch.float64)
    scale = weights.amax(dim=1, keepdim=True)
    scaled = weights / torch.where(scale > 0, scale, 1)
    mass = scaled.sum(dim=1)
    supported = mass > 0
    normalized = scaled / torch.where(supported, mass, 1)[:,None,:]
    feature_scale=values.detach().abs().amax(dim=1,keepdim=True).clamp_min(1.)
    unit_mean=((values/feature_scale)*normalized[...,None]).sum(dim=1)
    # Correct floating excursions outside the convex bound; straight-through
    # rounding correction retains the linear mean derivative in ordinary range.
    conditioned=unit_mean+(unit_mean.clamp(-1.,1.)-unit_mean).detach()
    fused=conditioned*feature_scale[:,0]
    return fused.to(samples.dtype), supported

# %%
def fuse_chunked(samples, valid, confidence, chunk_size=None):
    _validate(samples, valid, confidence, chunk_size)
    n = samples.shape[2]
    if n == 0:
        # Keep an empty autograd path to both inputs.
        return samples.sum(dim=1) + confidence.sum(dim=1)[...,None], valid.any(dim=1)
    outputs, masks = [], []
    for start in range(0, n, chunk_size or n):
        stop = min(start + (chunk_size or n), n)
        flags = valid[:,:,start:stop]
        conf = confidence[:,:,start:stop]
        eligible = flags & (conf > 0)
        weights = torch.where(eligible, conf, 0).to(torch.float64)
        values = torch.where(eligible[...,None], samples[:,:,start:stop], 0).to(torch.float64)
        scale = weights.amax(dim=1, keepdim=True)
        scaled = weights / torch.where(scale > 0, scale, 1)
        mass = scaled.sum(dim=1)
        support = mass > 0
        normalized = scaled / torch.where(support, mass, 1)[:,None,:]
        feature_scale=values.detach().abs().amax(dim=1,keepdim=True).clamp_min(1.)
        unit_mean=((values/feature_scale)*normalized[...,None]).sum(dim=1)
        conditioned=unit_mean+(unit_mean.clamp(-1.,1.)-unit_mean).detach()
        outputs.append((conditioned*feature_scale[:,0]).to(samples.dtype))
        masks.append(support)
    return torch.cat(outputs,dim=1), torch.cat(masks,dim=1)

# %%
def _included_points(points,features,point_valid,origin,voxel_size,grid_shape,pillar):
    tensors=(points,features,point_valid,origin,voxel_size,grid_shape)
    if not all(isinstance(t,torch.Tensor) for t in tensors): raise ValueError('tensor inputs required')
    if points.ndim!=3 or points.shape[-1]!=3 or points.shape[0]<1: raise ValueError('points must be [B,N,3]')
    B,N,_=points.shape
    if features.ndim!=3 or features.shape[:2]!=(B,N) or features.shape[2]<1 or point_valid.shape!=(B,N): raise ValueError('features/mask shape mismatch')
    if any(t.shape!=(3,) for t in (origin,voxel_size,grid_shape)): raise ValueError('xyz metadata must be [3]')
    if points.dtype not in (torch.float32,torch.float64) or any(t.dtype!=points.dtype for t in (features,origin,voxel_size)): raise ValueError('floating tensors must share float32/float64 dtype')
    if any(t.device!=points.device for t in tensors): raise ValueError('common device required')
    if point_valid.dtype!=torch.bool or grid_shape.dtype!=torch.int64 or not isinstance(pillar,bool): raise ValueError('bool mask/pillar and int64 grid required')
    if not (torch.isfinite(origin).all() and torch.isfinite(voxel_size).all() and (voxel_size>0).all() and (grid_shape>0).all()): raise ValueError('finite origin, positive size/shape required')
    upper=origin+voxel_size*grid_shape
    if not torch.isfinite(upper).all() or not (upper>origin).all(): raise ValueError('bounds must be finite and distinct in working dtype')
    if not (torch.isfinite(points[point_valid]).all() and torch.isfinite(features[point_valid]).all()): raise ValueError('valid points/features must be finite')
    clean=torch.where(point_valid[...,None],points,origin)
    inside=point_valid & (clean>=origin).all(-1) & (clean<upper).all(-1)
    locations=inside.nonzero(as_tuple=False)
    selected=clean[inside]
    cells=torch.floor((selected-origin)/voxel_size)
    # No packed integer keys. Reject only unrepresentable computed coordinates.
    if not (torch.isfinite(cells).all() and (cells>=0).all() and (cells<2**63).all()): raise ValueError('cell coordinate not representable as int64')
    # Physical half-open membership was decided first. Subtraction/division
    # may round an interior point onto the upper cell boundary; keep it in
    # the final cell. Integer conversion occurs only after representability.
    xyz=torch.minimum(cells.to(torch.int64),grid_shape-1)
    z=torch.zeros_like(xyz[:,2]) if pillar else xyz[:,2]
    indices=torch.stack((locations[:,0],z,xyz[:,1],xyz[:,0]),1)
    return indices,locations,features[inside]

# %%
def pool_unique_scatter(points,features,point_valid,origin,voxel_size,grid_shape,pillar=False):
    rows,locations,selected=_included_points(points,features,point_valid,origin,voxel_size,grid_shape,pillar)
    indices,ids,counts=torch.unique(rows,dim=0,sorted=True,return_inverse=True,return_counts=True)
    inverse=torch.full(points.shape[:2],-1,dtype=torch.int64,device=points.device)
    inverse[locations[:,0],locations[:,1]]=ids
    work=selected.double() if selected.dtype==torch.float32 else selected
    scale=work.new_zeros(len(indices),features.shape[-1]).scatter_reduce(0,ids[:,None].expand_as(work),work.abs(),reduce='amax',include_self=True)
    safe=torch.where(scale>0,scale,torch.ones_like(scale))
    sums=work.new_zeros(len(indices),features.shape[-1]).index_add(0,ids,work/safe[ids])
    means=((sums/counts[:,None]).clamp(-1,1)*safe).to(features.dtype)
    return indices,counts,means,inverse

# %%
def validate_bev(indices,features,counts,batch_size,grid_shape):
    if not all(isinstance(t,torch.Tensor) for t in (indices,features,counts,grid_shape)):
        raise ValueError('indices, features, counts and grid_shape must be tensors')
    if type(batch_size) is not int or batch_size<=0:
        raise ValueError('batch_size must be a positive Python integer')
    if features.ndim!=2 or features.shape[1]<=0 or features.dtype not in (torch.float32,torch.float64):
        raise ValueError('features must be float32/64 [K,C], C>0')
    K,C=features.shape
    if indices.shape!=(K,4) or indices.dtype!=torch.int64:
        raise ValueError('indices must be int64 [K,4] in batch,z,y,x order')
    if counts.shape!=(K,) or counts.dtype!=torch.int64:
        raise ValueError('counts must be positive int64 [K]')
    if grid_shape.shape!=(3,) or grid_shape.dtype!=torch.int64:
        raise ValueError('grid_shape must be int64 [3] in x,y,z order')
    if any(t.device!=features.device for t in (indices,counts,grid_shape)):
        raise ValueError('All tensors share device')
    X,Y,Z=grid_shape.tolist()
    if min(X,Y,Z)<=0:
        raise ValueError('Grid dimensions must be positive')
    cells=batch_size*Y*X
    if cells>torch.iinfo(torch.int64).max or cells*C>sys.maxsize//8 or cells>sys.maxsize//8:
        raise ValueError('Dense shape/byte count or flattened cell ID overflows')
    if not torch.isfinite(features).all():
        raise ValueError('Sparse cell features must be finite')
    bounds=indices.new_tensor([batch_size,Z,Y,X])
    if ((indices<0)|(indices>=bounds)).any() or (counts<=0).any():
        raise ValueError('Indices out of bounds or counts not positive')
    totals={}
    for row,count in zip(indices.tolist(),counts.tolist()):
        b,z,y,x=row
        key=(b,y,x)
        totals[key]=totals.get(key,0)+count
        if totals[key]>torch.iinfo(torch.int64).max:
            raise ValueError('Point count for a BEV cell overflows int64')
    return X,Y,Z

# %%
def scatter_bev_explicit(indices,features,counts,batch_size,grid_shape):
    X,Y,Z=validate_bev(indices,features,counts,batch_size,grid_shape)
    C=features.shape[1]
    zero=features.reshape(-1)[:1].double().sum()*0
    numerator=torch.zeros(batch_size,C,Y,X,dtype=torch.float64,device=features.device)+zero
    totals=torch.zeros(batch_size,1,Y,X,dtype=torch.int64,device=features.device)
    for i,(b,z,y,x) in enumerate(indices.tolist()):
        numerator[b,:,y,x] += features[i].double()*counts[i].double()
        totals[b,0,y,x] += counts[i]
    values=numerator/totals.clamp(min=1).double()
    return values.to(features.dtype),totals,totals>0

# %%
def scatter_bev_index_add(indices,features,counts,batch_size,grid_shape):
    X,Y,Z=validate_bev(indices,features,counts,batch_size,grid_shape)
    C=features.shape[1]; cells=batch_size*Y*X
    keys=(indices[:,0]*Y+indices[:,2])*X+indices[:,3]
    totals=torch.zeros(cells,dtype=torch.int64,device=features.device)
    totals.index_add_(0,keys,counts)
    weights=counts.double()/totals[keys].double()
    # Scale per destination/channel; detach this numerical conditioning value.
    # A scale of at least one avoids amplifying subnormal input/gradient noise.
    magnitudes=torch.zeros(cells,C,dtype=torch.float64,device=features.device)
    magnitudes.scatter_reduce_(0,keys[:,None].expand(-1,C),features.detach().double().abs(),reduce='amax',include_self=True)
    scale=magnitudes.clamp_min(1.)
    source=(features.double()/scale[keys])*weights[:,None]
    sums=torch.zeros(cells,C,dtype=torch.float64,device=features.device).index_add(0,keys,source)
    # Exact convex means lie in [-1,1]. Only rounding excursions are corrected;
    # retain the linear mean's derivative rather than a clamp-zero derivative.
    conditioned=sums+(sums.clamp(-1.,1.)-sums).detach()
    bev=(conditioned*scale).to(features.dtype).reshape(batch_size,Y,X,C).permute(0,3,1,2).contiguous()
    point_counts=totals.reshape(batch_size,1,Y,X)
    return bev,point_counts,point_counts>0

# %%
def scatter_bev_grouped(indices,features,counts,batch_size,grid_shape):
    X,Y,Z=validate_bev(indices,features,counts,batch_size,grid_shape)
    K,C=features.shape; cells=batch_size*Y*X
    keys=(indices[:,0]*Y+indices[:,2])*X+indices[:,3]
    order=keys.argsort(stable=True)
    unique,lengths=torch.unique_consecutive(keys[order],return_counts=True)
    grouped_counts=counts[order]
    totals=torch.stack([segment.sum() for segment in grouped_counts.split(lengths.tolist())]) if K else counts.new_empty(0)
    row_totals=torch.repeat_interleave(totals,lengths)
    if K:
        scales=torch.segment_reduce(features[order].detach().double().abs(),'max',lengths=lengths).clamp_min(1.)
        row_scales=torch.repeat_interleave(scales,lengths,dim=0)
        normalized=features[order].double()/row_scales*(grouped_counts.double()/row_totals.double())[:,None]
        sums=torch.segment_reduce(normalized,'sum',lengths=lengths)
        conditioned=sums+(sums.clamp(-1.,1.)-sums).detach()
        means=conditioned*scales
    else:
        means=features.new_empty(0,C,dtype=torch.float64)+features.double().sum()*0
    flat=features.new_zeros(cells,C).index_copy(0,unique,means.to(features.dtype))
    dense_counts=counts.new_zeros(cells).index_copy(0,unique,totals)
    bev=flat.reshape(batch_size,Y,X,C).permute(0,3,1,2).contiguous()
    point_counts=dense_counts.reshape(batch_size,1,Y,X)
    return bev,point_counts,point_counts>0
# %%
def _validate_scene(scene):
    required={'points','K','T_camera_from_ego','image_hw','point_valid','camera_valid','features','feature_from_image','confidence','point_features','origin','voxel_size','grid_shape'}
    if not isinstance(scene,dict) or not required.issubset(scene):
        raise ValueError('scene must contain the documented tensor keys')
    p,attributes=scene['points'],scene['point_features']
    if not isinstance(p,torch.Tensor) or p.ndim!=3 or p.shape[-1]!=3:
        raise ValueError('scene points must be [B,N,3]')
    if not isinstance(attributes,torch.Tensor) or attributes.ndim!=3 or attributes.shape[:2]!=p.shape[:2] or attributes.shape[2]<1 or attributes.dtype!=p.dtype or attributes.device!=p.device:
        raise ValueError('point_features must be [B,N,F], F>0 with point dtype/device')

# %%
# Version1: expose each stage with modest finite data; baseline/followup.
# Per-camera projection and raw confidence/numerator reductions make the
# geometry and weighted-mean equations visible. The supplied sampling/grouping
# kernels stay shared so the capstone does not repeat all prerequisite variants.
def perception_bev_transparent(scene):
    _validate_scene(scene)
    p=scene['points']; chunk=scene.get('chunk_size'); depth_cut=scene.get('min_depth',0.001)
    uv,depth,projected_valid=project_per_camera(p,scene['K'],scene['T_camera_from_ego'],scene['image_hw'],scene['point_valid'],scene['camera_valid'],depth_cut,chunk)
    samples,sampled_valid=sample_features_vectorized(scene['features'],uv,projected_valid,scene['feature_from_image'],chunk)
    fused,image_supported=fuse_transparent(samples,sampled_valid,scene['confidence'],chunk)
    lidar=torch.where(scene['point_valid'][...,None],scene['point_features'],0.)
    concatenated=torch.cat([lidar,fused,image_supported[...,None].to(p.dtype)],dim=-1)
    indices,counts,means,inverse=pool_unique_scatter(p,concatenated,scene['point_valid'],scene['origin'],scene['voxel_size'],scene['grid_shape'])
    bev,point_counts,occupied=scatter_bev_explicit(indices,means,counts,p.shape[0],scene['grid_shape'])
    return dict(uv=uv,depth=depth,projected_valid=projected_valid,samples=samples,sampled_valid=sampled_valid,
                fused=fused,image_supported=image_supported,point_features=concatenated,
                indices=indices,counts=counts,means=means,inverse=inverse,
                bev=bev,point_counts=point_counts,occupied=occupied)

# %%
# Version2: production float32/64. Broadcast/gather work, conditioned camera
# means and exact-count stable BEV scatter. Voxel membership is discrete;
# feature gradients are ordinary away from numerical correction boundaries.
def perception_bev_vectorized(scene):
    _validate_scene(scene)
    p=scene['points']; chunk=scene.get('chunk_size'); depth_cut=scene.get('min_depth',0.001)
    uv,depth,projected_valid=project_broadcast(p,scene['K'],scene['T_camera_from_ego'],scene['image_hw'],scene['point_valid'],scene['camera_valid'],depth_cut,chunk)
    samples,sampled_valid=sample_features_vectorized(scene['features'],uv,projected_valid,scene['feature_from_image'],chunk)
    fused,image_supported=fuse_stable(samples,sampled_valid,scene['confidence'],chunk)
    lidar=torch.where(scene['point_valid'][...,None],scene['point_features'],0.)
    concatenated=torch.cat([lidar,fused,image_supported[...,None].to(p.dtype)],dim=-1)
    indices,counts,means,inverse=pool_unique_scatter(p,concatenated,scene['point_valid'],scene['origin'],scene['voxel_size'],scene['grid_shape'])
    bev,point_counts,occupied=scatter_bev_index_add(indices,means,counts,p.shape[0],scene['grid_shape'])
    return dict(uv=uv,depth=depth,projected_valid=projected_valid,samples=samples,sampled_valid=sampled_valid,
                fused=fused,image_supported=image_supported,point_features=concatenated,
                indices=indices,counts=counts,means=means,inverse=inverse,
                bev=bev,point_counts=point_counts,occupied=occupied)

# %%
# Version3: production float32/64; point chunks for camera stages and grouped
# scatter avoid a dense float64 BEV accumulator. Complete output schema and
# required intermediate tensors remain unchanged. Validation scans full inputs.
def perception_bev_chunked(scene):
    _validate_scene(scene)
    p=scene['points']; chunk=scene.get('chunk_size'); depth_cut=scene.get('min_depth',0.001)
    uv,depth,projected_valid=project_chunked(p,scene['K'],scene['T_camera_from_ego'],scene['image_hw'],scene['point_valid'],scene['camera_valid'],depth_cut,chunk)
    samples,sampled_valid=sample_features_chunked(scene['features'],uv,projected_valid,scene['feature_from_image'],chunk)
    fused,image_supported=fuse_chunked(samples,sampled_valid,scene['confidence'],chunk)
    lidar=torch.where(scene['point_valid'][...,None],scene['point_features'],0.)
    concatenated=torch.cat([lidar,fused,image_supported[...,None].to(p.dtype)],dim=-1)
    indices,counts,means,inverse=pool_unique_scatter(p,concatenated,scene['point_valid'],scene['origin'],scene['voxel_size'],scene['grid_shape'])
    bev,point_counts,occupied=scatter_bev_grouped(indices,means,counts,p.shape[0],scene['grid_shape'])
    return dict(uv=uv,depth=depth,projected_valid=projected_valid,samples=samples,sampled_valid=sampled_valid,
                fused=fused,image_supported=image_supported,point_features=concatenated,
                indices=indices,counts=counts,means=means,inverse=inverse,
                bev=bev,point_counts=point_counts,occupied=occupied)

# %%
perception_bev=perception_bev_vectorized

# %%
def perception_experiment(repeats_of_scene=8,chunk_size=8,device='cpu',dtype=torch.float32):
    from torch_judge.measurement import measure
    scene=base_scene(dtype)
    for k in ['points','point_valid','point_features']:scene[k]=scene[k].repeat(1,repeats_of_scene,1) if scene[k].ndim==3 else scene[k].repeat(1,repeats_of_scene)
    scene['confidence']=scene['confidence'].repeat(1,1,repeats_of_scene)
    scene={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in scene.items()}
    scene['chunk_size']=chunk_size
    expected=scalar_pipeline(scene)['bev'];reports={}
    for name,fn in [('transparent',perception_bev_transparent),('vectorized',perception_bev_vectorized),('chunked',perception_bev_chunked)]:
        actual=fn(scene)['bev']
        reports[name]={'max_abs_error':(actual-expected).abs().max().item(),
                       'timing':measure(lambda:fn(scene),device=device,workload={'B':1,'M':1,'N':scene['points'].shape[1],'image_C':1,'lidar_F':1,'grid_xyz':(4,3,3),'chunk_size':chunk_size,'dtype':str(dtype),'fixture':'repeated_hand_scene'}),
                       'analytic_required_bev_bytes':actual.numel()*actual.element_size()}
    return reports

# %% [verifiers]
def _scalar_cell(coordinate,origin,size):
    # Decimal arithmetic preserves a tiny displacement next to a unit origin.
    # 1100 digits cover the finite float64 exponent range for these ratios.
    # This fixture oracle does not promise agreement with working-dtype floor
    # at every near-internal boundary; literal outer-boundary cases establish
    # the reviewed membership/grouping behavior independently.
    with localcontext() as context:
        context.prec=1100
        ratio=(Decimal.from_float(float(coordinate))-Decimal.from_float(float(origin)))/Decimal.from_float(float(size))
        return int(ratio.to_integral_value(rounding=ROUND_FLOOR))


def base_scene(dtype=torch.float64):
    # Ego x-forward/y-left/z-up -> camera right/down/forward.
    R=torch.tensor([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]],dtype=dtype)
    T=torch.eye(4,dtype=dtype); T[:3,:3]=R
    y,x=torch.meshgrid(torch.arange(3,dtype=dtype),torch.arange(4,dtype=dtype),indexing='ij')
    return dict(points=torch.tensor([[[1.,-1.,-1.],[1.,-1.,.5],[2.,-1.,-1.]]],dtype=dtype),
                K=torch.eye(3,dtype=dtype)[None,None],T_camera_from_ego=T[None,None],image_hw=torch.tensor([[[3,4]]]),
                point_valid=torch.ones(1,3,dtype=torch.bool),camera_valid=torch.ones(1,1,dtype=torch.bool),
                features=(x+10*y)[None,None,None],feature_from_image=torch.eye(3,dtype=dtype)[None,None],
                confidence=torch.ones(1,1,3,dtype=dtype),point_features=torch.tensor([[[2.],[8.],[10.]]],dtype=dtype),
                origin=torch.tensor([0.,-2.,-2.],dtype=dtype),voxel_size=torch.ones(3,dtype=dtype),grid_shape=torch.tensor([4,3,3]))


def verify_baseline_known_bev(perception_bev):
    result=perception_bev(base_scene())
    expected=torch.zeros(1,3,3,4,dtype=torch.float64)
    expected[0,:,1,1]=torch.tensor([5.,5.5,.5],dtype=torch.float64)
    expected[0,:,1,2]=torch.tensor([10.,5.5,1.],dtype=torch.float64)
    torch.testing.assert_close(result['bev'],expected)
    assert result['point_counts'][0,0,1,1]==2
    assert result['image_supported'].tolist()==[[True,False,True]]

def scalar_pipeline(scene):
    """Detached scalar geometry and direct point-to-pillar means, no stage refs."""
    p=scene['points']; B,N,_=p.shape; M,C,H,W=scene['features'].shape[1:]
    F=scene['point_features'].shape[-1]; width=F+C+1
    uv=p.new_zeros(B,M,N,2); depth=p.new_zeros(B,M,N)
    projected=torch.zeros(B,M,N,dtype=torch.bool,device=p.device)
    samples=p.new_zeros(B,M,N,C); sampled=projected.clone()
    fused=p.new_zeros(B,N,C); image_supported=torch.zeros(B,N,dtype=torch.bool,device=p.device)
    concatenated=p.new_zeros(B,N,width)
    origin=scene['origin'].tolist(); size=scene['voxel_size'].tolist(); X,Y,Z=scene['grid_shape'].tolist()
    groups={}; pillars={}
    for b in range(B):
        for n in range(N):
            if not scene['point_valid'][b,n]:
                continue
            point=p[b,n].tolist()
            eligible=[]
            for m in range(M):
                if not scene['camera_valid'][b,m]:
                    continue
                T=scene['T_camera_from_ego'][b,m].tolist(); K=scene['K'][b,m].tolist()
                xyz=[math.fsum(T[i][j]*point[j] for j in range(3))+T[i][3] for i in range(3)]
                if xyz[2]<=scene.get('min_depth',.001):
                    continue
                homogeneous=[math.fsum(K[i][j]*xyz[j] for j in range(3)) for i in range(3)]
                u,v=homogeneous[0]/homogeneous[2],homogeneous[1]/homogeneous[2]
                ih,iw=scene['image_hw'][b,m].tolist()
                if not (0<=u<=iw-1 and 0<=v<=ih-1):
                    continue
                projected[b,m,n]=True; uv[b,m,n]=p.new_tensor([u,v]); depth[b,m,n]=xyz[2]
                a=scene['feature_from_image'][b,m].tolist()
                fx=a[0][0]*u+a[0][1]*v+a[0][2]; fy=a[1][0]*u+a[1][1]*v+a[1][2]
                if not (0<=fx<=W-1 and 0<=fy<=H-1):
                    continue
                sampled[b,m,n]=True
                x0,y0=math.floor(fx),math.floor(fy); dx,dy=fx-x0,fy-y0
                for c in range(C):
                    values=[]
                    for yy,wy in [(y0,1-dy),(min(y0+1,H-1),dy)]:
                        for xx,wx in [(x0,1-dx),(min(x0+1,W-1),dx)]:
                            values.append(scene['features'][b,m,c,yy,xx].item()*wy*wx)
                    samples[b,m,n,c]=math.fsum(values)
                confidence=scene['confidence'][b,m,n].item()
                if confidence>0:
                    eligible.append((m,confidence))
            if eligible:
                image_supported[b,n]=True
                scale=max(w for m,w in eligible); total=math.fsum(w/scale for m,w in eligible)
                for c in range(C):
                    fused[b,n,c]=math.fsum(samples[b,m,n,c].item()*((w/scale)/total) for m,w in eligible)
            concatenated[b,n,:F]=scene['point_features'][b,n]
            concatenated[b,n,F:F+C]=fused[b,n]
            concatenated[b,n,-1]=int(image_supported[b,n])
            if all(origin[j]<=point[j]<origin[j]+size[j]*[X,Y,Z][j] for j in range(3)):
                x,y,z=[_scalar_cell(point[j],origin[j],size[j]) for j in range(3)]
                groups.setdefault((b,z,y,x),[]).append(n)
                pillars.setdefault((b,y,x),[]).append(n)
    keys=sorted(groups); indices=torch.tensor(keys,dtype=torch.int64,device=p.device).reshape(-1,4)
    counts=torch.tensor([len(groups[k]) for k in keys],dtype=torch.int64,device=p.device)
    means=p.new_zeros(len(keys),width); inverse=torch.full((B,N),-1,dtype=torch.int64,device=p.device)
    for i,key in enumerate(keys):
        b,z,y,x=key; rows=groups[key]; inverse[b,rows]=i
        for c in range(width):
            means[i,c]=math.fsum(concatenated[b,n,c].item()/len(rows) for n in rows)
    bev=p.new_zeros(B,width,Y,X); totals=torch.zeros(B,1,Y,X,dtype=torch.int64,device=p.device)
    for (b,y,x),rows in pillars.items():
        totals[b,0,y,x]=len(rows)
        for c in range(width):
            bev[b,c,y,x]=math.fsum(concatenated[b,n,c].item()/len(rows) for n in rows)
    return dict(uv=uv,depth=depth,projected_valid=projected,samples=samples,sampled_valid=sampled,
                fused=fused,image_supported=image_supported,point_features=concatenated,
                indices=indices,counts=counts,means=means,inverse=inverse,bev=bev,point_counts=totals,occupied=totals>0)


def make_eval_data():
    cases={}
    def add(name,stage,purpose,scene):
        cases[name]=EvalCase(name=name,stage=stage,purpose=purpose,inputs=dict(scene=scene),expected=scalar_pipeline(scene),comparison=partial(compare_values,atol=2e-6,rtol=2e-6))
    add('hand_scene','baseline','Two pillars: no-image point retains LiDAR and contributes to count-weighted BEV.',base_scene())
    s=base_scene(); permutation=torch.tensor([0,0,1,2])
    for key in ['points','point_valid','point_features']:s[key]=s[key][:,permutation]
    s['point_features']=torch.tensor([[[2.],[4.],[8.],[10.]]],dtype=torch.float64)
    s['confidence']=torch.ones(1,1,4,dtype=torch.float64)
    add('duplicate_unequal_z','followup','Two points in one z cell and one in another: direct point mean14/3, not a mean of voxel means.',s)
    s=base_scene()
    for key in ['K','T_camera_from_ego','image_hw','camera_valid','features','feature_from_image','confidence']:s[key]=s[key].repeat(1,2,*([1]*(s[key].ndim-2)))
    s['features'][:,1] += 20.; s['confidence'][:,1] *= 3.
    add('weighted_two_views','followup','Second view has an offset20 and confidence3: fused image offset15.',s)
    s=base_scene()
    for key in ['points','point_valid','point_features','K','T_camera_from_ego','image_hw','camera_valid','features','feature_from_image','confidence']:s[key]=s[key].repeat(2,*([1]*(s[key].ndim-1)))
    s['point_features'][1]+=10.
    add('separate_batches','followup','Identical geometric cells in two batches do not mix.',s)
    s=base_scene();s['feature_from_image'][:,:,0,0]=.5;s['feature_from_image'][:,:,1,1]=.5
    add('feature_affine_crop','followup','Explicit image-to-feature resize changes sampling; tensor shapes do not determine alignment.',s)
    s=base_scene();s['points'][0,0,0]=4.
    add('half_open_grid','followup','Image-supported point on BEV maximum boundary remains inspectable but is excluded from pooling.',s)
    s=base_scene();s['point_valid'].zero_();s['points'].fill_(float('nan'));s['point_features'].fill_(float('nan'));s['features'].fill_(float('nan'));s['confidence'].fill_(float('nan'))
    add('all_masked_points','production','All padding: zero/empty outputs with safe zero gradients, no NaN contamination.',s)
    s=base_scene();s['camera_valid'].zero_();s['features'].fill_(float('nan'));s['confidence'].fill_(float('nan'))
    add('no_camera_nan','production','Unavailable camera NaNs cannot erase LiDAR occupancy or poison gradients.',s)
    s=base_scene();s['point_valid'][:,1]=False;s['points'][:,1]=float('nan');s['point_features'][:,1]=float('nan')
    add('masked_point_nan','production','Point padding is sanitized before any transform/concatenation.',s)
    s=base_scene()
    for key in ['points','point_valid','point_features']:s[key]=s[key][:,:0]
    s['confidence']=s['confidence'][:,:,:0]
    add('empty_scene','production','Empty sensor point set: dense zeros and empty differentiable intermediates.',s)
    s=base_scene(torch.float32);s['features']=s['features'].transpose(-1,-2);s['image_hw']=torch.tensor([[[3,4]]]);s['chunk_size']=1
    add('strided_chunk_scene','production','Storage layout differs from declared feature shape; process one point at a time.',s)
    s=base_scene(torch.float32)
    for key in ['K','T_camera_from_ego','image_hw','camera_valid','features','feature_from_image','confidence']:s[key]=s[key].repeat(1,2,*([1]*(s[key].ndim-2)))
    s['features'][:,1]+=20.;s['confidence'].fill_(3e38)
    add('extreme_confidence','production','Raw confidence numerator overflows; scaled view normalization must preserve means.',s)
    for dtype in (torch.float32,torch.float64):
        s=base_scene(dtype);tiny=torch.finfo(dtype).tiny
        s['points']=torch.tensor([[[-tiny,-.5,-.5],[-.5,-.5,-.5],[0.,-.5,-.5],[tiny,-.5,-.5]]],dtype=dtype)
        s['point_features']=torch.tensor([[[2.],[4.],[8.],[16.]]],dtype=dtype)
        s['point_valid']=torch.ones(1,4,dtype=torch.bool);s['camera_valid'].zero_()
        s['confidence']=torch.full((1,1,4),float('nan'),dtype=dtype);s['features'].fill_(float('nan'))
        s['origin']=-torch.ones(3,dtype=dtype);s['voxel_size']=torch.ones(3,dtype=dtype);s['grid_shape']=torch.ones(3,dtype=torch.int64)
        add('near_zero_upper_'+str(dtype).split('.')[-1],'followup','Both interior LiDAR rows land in one final BEV cell; exact/positive zero edge rows are outside.',s)
    s=base_scene();s['features'].requires_grad_();s['point_features'].requires_grad_()
    add('live_feature_gradients','production','Backpropagate through both eligible camera maps and LiDAR attributes.',s)
    return cases


def verify_baseline_all_intermediates(perception_bev):
    case=make_eval_data()['hand_scene']
    errors=case.comparison(perception_bev(**case.inputs),case.expected)
    assert not errors,errors


def verify_followup_near_zero_upper_literal(perception_bev):
    for dtype in (torch.float32,torch.float64):
        case=make_eval_data()['near_zero_upper_'+str(dtype).split('.')[-1]]
        got=perception_bev(**case.inputs)
        assert torch.equal(got['indices'],torch.tensor([[0,0,0,0]]))
        assert torch.equal(got['counts'],torch.tensor([2]))
        assert torch.equal(got['inverse'],torch.tensor([[0,0,-1,-1]]))
        assert torch.equal(got['point_counts'],torch.tensor([[[[2]]]]))
        torch.testing.assert_close(got['bev'],torch.tensor([[[[3.]],[[0.]],[[0.]]]],dtype=dtype))
        assert not case.comparison(got,case.expected)


def verify_followup_scenes(perception_bev):
    for name in ['duplicate_unequal_z','weighted_two_views','separate_batches','feature_affine_crop','half_open_grid']:
        case=make_eval_data()[name]
        errors=case.comparison(perception_bev(**case.inputs),case.expected)
        assert not errors,(name,errors)


def verify_production_scenes(perception_bev):
    for name,case in make_eval_data().items():
        if case.stage=='production':
            errors=case.comparison(perception_bev(**case.inputs),case.expected)
            assert not errors,(name,errors)

def verify_production_camera_lidar_gradients(perception_bev):
    scene=base_scene();maps=scene['features'].requires_grad_();lidar=scene['point_features'].requires_grad_()
    assert torch.autograd.gradcheck(lambda f,l:perception_bev(scene|dict(features=f,point_features=l))['bev'],(maps,lidar),eps=1e-6,atol=1e-5,rtol=1e-4)
    result=perception_bev(scene)
    gm,gl=torch.autograd.grad(result['bev'].sum(),(maps,lidar))
    expected=maps.new_tensor([[[[[.25,.25,0.,0.],[.25,.75,0.,0.],[0.,0.,0.,0.]]]]])
    torch.testing.assert_close(gm,expected)
    torch.testing.assert_close(gl,lidar.new_tensor([[[.5],[.5],[1.]]]))
    for name in ['no_camera_nan','empty_scene','masked_point_nan','all_masked_points']:
        scene=make_eval_data()[name].inputs['scene']
        maps=scene['features'].detach().clone().requires_grad_();lidar=scene['point_features'].detach().clone().requires_grad_()
        result=perception_bev(scene|dict(features=maps,point_features=lidar))
        gm,gl=torch.autograd.grad(result['bev'].sum(),(maps,lidar))
        assert torch.isfinite(gm).all() and torch.isfinite(gl).all()
        if name in ['no_camera_nan','empty_scene','all_masked_points']:assert gm.count_nonzero()==0
        if name=='all_masked_points':assert gl.count_nonzero()==0
        if name=='masked_point_nan':assert gl[:,1].count_nonzero()==0


def verify_production_validation_and_no_mutation(perception_bev):
    for name,case in make_eval_data().items():
        scene=case.inputs['scene'];saved={k:v.clone() for k,v in scene.items() if isinstance(v,torch.Tensor)}
        perception_bev(scene)
        for k,original in saved.items():torch.testing.assert_close(scene[k],original,equal_nan=True)
    scene=base_scene()
    bad=[None,{k:v for k,v in scene.items() if k!='features'},scene|dict(chunk_size=0),
         scene|dict(point_features=scene['point_features'].float()),scene|dict(confidence=-scene['confidence']),
         scene|dict(grid_shape=torch.tensor([0,3,3]))]
    for changed in bad:
        try:perception_bev(changed)
        except ValueError:continue
        raise AssertionError('Expected documented ValueError')


def verify_production_generated_permutations(perception_bev):
    @settings(max_examples=16,derandomize=True,deadline=None)
    @given(rows=st.lists(st.tuples(st.integers(1,2),st.booleans(),st.integers(-10,10)),max_size=8),weight=st.integers(0,4))
    def trial(rows,weight):
        scene=base_scene();N=len(rows)
        scene['points']=torch.tensor([[[x,-1.,.5 if invisible else -1.] for x,invisible,l in rows]],dtype=torch.float64).reshape(1,N,3)
        scene['point_features']=torch.tensor([[[l] for x,invisible,l in rows]],dtype=torch.float64).reshape(1,N,1)
        scene['point_valid']=torch.ones(1,N,dtype=torch.bool)
        for k in ['K','T_camera_from_ego','image_hw','camera_valid','features','feature_from_image']:
            scene[k]=scene[k].repeat(1,2,*([1]*(scene[k].ndim-2)))
        scene['features'][:,1]+=10.
        scene['confidence']=torch.ones(1,2,N,dtype=torch.float64);scene['confidence'][:,1]*=weight
        got=perception_bev(scene);expected=scalar_pipeline(scene)
        errors=compare_values(got,expected,atol=2e-6,rtol=2e-6);assert not errors,errors
        perm=torch.arange(N-1,-1,-1)
        reordered=scene.copy()
        for k in ['points','point_valid','point_features']:reordered[k]=scene[k][:,perm]
        reordered['confidence']=scene['confidence'][:,:,perm]
        for k in ['K','T_camera_from_ego','image_hw','camera_valid','features','feature_from_image','confidence']:
            reordered[k]=reordered[k].flip(1)
        other=perception_bev(reordered)
        torch.testing.assert_close(other['bev'],got['bev'])
        assert torch.equal(other['point_counts'],got['point_counts'])
        assert torch.equal(other['occupied'],got['occupied'])
    trial()

# %%
if not evaluate('0109_perception_bev_eval.py',perception_bev):
    raise SystemExit(1)
