# %%
# 0109 · Follow a feature from camera and LiDAR into BEV
# Baseline timebox: 30–45 minutes. This is a composition workout: supplied
# local kernels let you concentrate on data flow, masks and intermediate shapes.
# Inspect or revise any helper in the setup cell; saved-candidate checks execute
# that setup too. There are no imports from another numbered exercise.

# %%
import math
import sys
import torch
from torch_judge import check, get_eval_data, compare, replay

# %%
# scene is a dictionary with exactly these required concepts (extra optional
# min_depth=0.001 and chunk_size=None control geometric cutoff and point work):
# points [B,N,3] ego x-forward,y-left,z-up;
# K [B,M,3,3], T_camera_from_ego [B,M,4,4], image_hw int64 [B,M,2];
# point_valid bool [B,N], camera_valid bool [B,M];
# features [B,M,C,Hf,Wf] camera maps, feature_from_image [B,M,3,3];
# confidence [B,M,N] nonnegative on sampled support;
# point_features [B,N,F] LiDAR attributes; origin/voxel_size [3];
# grid_shape int64 [3] x,y,z. Floating inputs share float32/64 dtype/device.
# N may be zero; all other sensor/feature dimensions are positive.
# Camera x-right,y-down,z-forward. Transform names specify direction; affine
# image-to-feature mapping never comes from shape ratios. Image support means
# projectable/samplable, NOT independently verified unoccluded.
# Camera pixel centers have inclusive bounds; voxel bounds are half-open.
# Finite metadata is required even for unavailable cameras. Entire camera maps
# may be NaN when no projected point is supported; valid point padding and
# unsupported confidence may be NaN. Sanitize padding before concatenation.
# Require finite attributes at valid points. Stage helpers validate all inputs.
# Return these named intermediates:
# uv/depth/projected_valid: [B,M,N,2]/[B,M,N]/bool[B,M,N]
# samples/sampled_valid: [B,M,N,C]/bool[B,M,N]
# fused/image_supported: [B,N,C]/bool[B,N]
# point_features: [B,N,F+C+1], channel order LiDAR, fused image, image support.
# indices/counts/means/inverse: [V,4]/[V]/[V,F+C+1]/[B,N], inverse=-1 excluded.
# bev/point_counts/occupied: [B,F+C+1,Y,X]/int64[B,1,Y,X]/bool[B,1,Y,X].
# Points without image support still contribute LiDAR occupancy/features.
# The support channel in BEV is a fraction of image-supported points, not a
# boolean occupancy map. Point decoration is an optional later input branch.

# %% [setup]
# One stable local kernel per stage. Their implementation scope is inherited
# from the previous workouts: fixed discrete membership, smooth interior
# feature/coordinate gradients, float64 conditioned reductions, and explicit
# finite/shape validation. These are supplied tools, not hidden judge code.
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

def _validate_scene(scene):
    required={'points','K','T_camera_from_ego','image_hw','point_valid','camera_valid','features','feature_from_image','confidence','point_features','origin','voxel_size','grid_shape'}
    if not isinstance(scene,dict) or not required.issubset(scene):
        raise ValueError('scene must contain the documented tensor keys')
    p,attributes=scene['points'],scene['point_features']
    if not isinstance(p,torch.Tensor) or p.ndim!=3 or p.shape[-1]!=3:
        raise ValueError('scene points must be [B,N,3]')
    if not isinstance(attributes,torch.Tensor) or attributes.ndim!=3 or attributes.shape[:2]!=p.shape[:2] or attributes.shape[2]<1 or attributes.dtype!=p.dtype or attributes.device!=p.device:
        raise ValueError('point_features must be [B,N,F], F>0 with point dtype/device')


project_points=project_broadcast
sample_features=sample_features_vectorized
fuse_views=fuse_stable
pool_points=pool_unique_scatter
scatter_bev=scatter_bev_index_add

# %% [answer]
def perception_bev(scene):
    pass  # Compose supplied local helpers; retain every named intermediate.

# %%
# Baseline: trace one no-image point. It still has a voxel inverse ID, a LiDAR
# feature and a contribution to BEV counts. Which mask must NOT gate pooling?
check('0109_perception_bev_eval.py',stage='baseline')

# %%
eval_data=get_eval_data('0109_perception_bev_eval.py')
case=eval_data['hand_scene']
case.inputs['scene'],case.expected['bev'],case.expected['image_supported']

# %%
actual=perception_bev(**case.inputs)
result=compare(case,actual)
result.passed,result.diagnostics,result.actual,result.expected

# %%
# Find the FIRST disagreeing intermediate. A wrong BEV can begin with uv,
# feature alignment, confidence, counts or x/y flattening. Inspect all of them
# before changing the last scatter stage.
{name:(result.actual[name],result.expected[name]) for name in result.expected}

# %%
# Follow-up: repeat a point in one z voxel. One z-cell mean gets two votes,
# another gets one. Then change camera order and batch membership.
check('0109_perception_bev_eval.py',stage='followup')

# %%
# Faulty-helper challenge, after your composition works: ignore confidence.
# Keep the bad result for inspection; the next cell repairs the helper binding.
original_fuse_views=fuse_views
def equal_weight_bug(samples,valid,confidence,chunk_size=None):
    return original_fuse_views(samples,valid,torch.ones_like(confidence),chunk_size)
fuse_views=equal_weight_bug
broken_helper_results=replay('0109_perception_bev_eval.py',perception_bev,names=['weighted_two_views'])
bad=broken_helper_results['weighted_two_views']
bad.passed,bad.diagnostics,bad.actual,bad.expected

# %%
# Restore the helper, run fresh inputs, then compare with the retained failure.
fuse_views=original_fuse_views
corrected_helper_results=replay('0109_perception_bev_eval.py',perception_bev,names=['weighted_two_views'])
corrected_helper_results['weighted_two_views'].passed,bad.diagnostics

# %%
# Production: unavailable-camera and point-padding NaNs, empty points, strides,
# extreme confidence, no mutation and backward through both feature sources.
# Hard support/group assignment has no derivative through boundary changes;
# smooth interior gradients remain ordinary. Numerical conditioning keeps
# finite convex means finite, but arbitrary extreme confidence derivatives
# can exceed float64 and are not an arbitrary-precision guarantee. Pooling
# clamp/scale rounding boundaries use their finite algorithm's derivative.
check('0109_perception_bev_eval.py',stage='production')

# %%
# Revise the answer/setup cell, rerun it and replay. Only fresh cases execute;
# your earlier result and bad helper result remain available in the kernel.
results=replay('0109_perception_bev_eval.py',perception_bev,names=['no_camera_nan','masked_point_nan','empty_scene','duplicate_unequal_z'])
{name:r.diagnostics for name,r in results.items() if not r.passed}

# %%
case=get_eval_data('0109_perception_bev_eval.py')['live_feature_gradients']
outputs=perception_bev(**case.inputs)
scene=case.inputs['scene']
image_gradient,lidar_gradient=torch.autograd.grad(outputs['bev'].sum(),(scene['features'],scene['point_features']))
image_gradient,lidar_gradient,outputs['point_counts']

# %%
# Open the eval script: compare complete transparent, vectorized/stable and
# chunked compositions. Their local helpers are ordinary cells, independent
# of this question's editable bindings. Call perception_experiment() explicitly
# after checking correctness. Chunking bounds selected forward work, not the
# required inspectable intermediates or all backward state. Dense BEV storage
# cannot be removed while this output contract remains dense.

# %%
# Boundary precision challenge: -tiny is strictly inside a box ending at zero,
# but (-tiny - -1) may round to 1. Membership must be checked in physical
# coordinates first. Only then clip a rounded cell to grid_shape-1; clipping
# before membership would incorrectly admit zero and positive outside rows.
# The near_zero_upper_float32/float64 eval cases have literal expected groups.
tiny=torch.finfo(torch.float32).tiny
probe=torch.tensor([-tiny,0.,tiny],dtype=torch.float32)
probe<0,torch.floor(probe+1)
