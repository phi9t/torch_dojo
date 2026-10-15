"""BEV queries meet camera features through explicit projection and sampling."""
# %%
import math
from functools import partial
import torch
from torch import nn
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values

# %%
# Local production-safe geometry/sampling helpers: this pair never imports
# another numbered question/reference. Masked maps are sanitized before math.
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

def _project_points(points, K, T_camera_from_ego, image_hw, point_valid,
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

def _sample_features(features, uv, valid, feature_from_image, chunk_size=None):
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
def _validate_attention_inputs(queries,image_features,reference_points):
    if (not isinstance(queries,torch.Tensor) or queries.ndim != 3 or queries.shape[0] < 1 or queries.shape[2] < 1
            or queries.dtype not in (torch.float32,torch.float64) or not torch.isfinite(queries).all()):
        raise ValueError('queries must be finite float32/float64 [B,Q,C] with B,C>0')
    b,q,c=queries.shape
    if (not isinstance(reference_points,torch.Tensor) or reference_points.ndim != 4
            or reference_points.shape[:2] != (b,q) or reference_points.shape[2] < 1 or reference_points.shape[3] != 3
            or reference_points.dtype != queries.dtype or reference_points.device != queries.device):
        raise ValueError('reference_points must share query dtype/device and have [B,Q,A,3], A>0')
    if (not isinstance(image_features,torch.Tensor) or image_features.ndim != 5
            or image_features.shape[0] != b or image_features.shape[2] != c
            or image_features.dtype != queries.dtype or image_features.device != queries.device):
        raise ValueError('image_features must be [B,M,C,Hf,Wf] with query dtype/device')
    return b,q,c


def _prepare(queries,image_features,reference_points,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth,chunk_size):
    b,q,c=_validate_attention_inputs(queries,image_features,reference_points)
    a=reference_points.shape[2]
    points=reference_points.reshape(b,q*a,3)
    uv,depth,projected=_project_points(points,K,T_camera_from_ego,image_hw,
        torch.ones((b,q*a),dtype=torch.bool,device=queries.device),camera_valid,min_depth=min_depth,chunk_size=chunk_size)
    samples,sample_valid=_sample_features(image_features,uv,projected,feature_from_image,chunk_size=chunk_size)
    m=image_features.shape[1]
    geometry=dict(uv=uv.reshape(b,m,q,a,2),depth=depth.reshape(b,m,q,a),projected=projected.reshape(b,m,q,a),
        samples=samples.reshape(b,m,q,a,c),sample_valid=sample_valid.reshape(b,m,q,a))
    values=geometry['samples'].permute(0,2,1,3,4).reshape(b,q,m*a,c).double()
    mask=geometry['sample_valid'].permute(0,2,1,3).reshape(b,q,m*a)
    return geometry,values,mask


def _convex_sum(values,weights):
    # Shared numerical primitive, not another full attention implementation.
    # Scale features so repeated finite maxima cannot overflow a convex sum.
    # Preserve ordinary derivatives when every feature value is exactly zero.
    scale=values.abs().amax(dim=-2,keepdim=True)
    normalized=values/torch.where(scale>0,scale,1)
    total=(normalized*weights[...,None]).sum(dim=-2).clamp(-1,1)*scale.squeeze(-2)
    zero_values=torch.where(scale==0,values,0)
    zero_total=(zero_values*weights[...,None]).sum(dim=-2)
    return torch.where(scale.squeeze(-2)>0,total,zero_total)

# %%
# Reference 1: per-query scores and normalization expose the key set.
# Production qualified float32/64, CPU portable. Keys are camera-major then
# anchor-major. Float64 scores/weights are diagnostics; updates preserve dtype.
# Dot scores must fit float64; nonfinite eligible scores raise ValueError.
def bev_attention_loop(queries,image_features,reference_points,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth=.001,chunk_size=None):
    geometry,values,mask=_prepare(queries,image_features,reference_points,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth,chunk_size)
    b,q,c=queries.shape; m=image_features.shape[1]; a=reference_points.shape[2]
    updates=queries*0+values.reshape(-1)[:1].sum().to(queries.dtype)*0
    scores=values.new_zeros((b,q,m*a)); weights=values.new_zeros((b,q,m*a))
    support=mask.any(dim=-1)
    for batch in range(b):
        for point in range(q):
            raw=torch.stack([torch.dot(queries[batch,point].double()/math.sqrt(c),values[batch,point,key]) for key in range(m*a)])
            if not torch.isfinite(raw[mask[batch,point]]).all():
                raise ValueError('eligible dot scores overflow float64')
            scores[batch,point]=torch.where(mask[batch,point],raw,0)
            if not bool(support[batch,point]):
                continue
            maximum=raw[mask[batch,point]].max()
            shifted=torch.where(mask[batch,point],raw-maximum,0)
            numerator=torch.where(mask[batch,point],shifted.exp(),0)
            probability=numerator/numerator.sum()
            weights[batch,point]=probability
            updates[batch,point]=_convex_sum(values[batch,point],probability).to(queries.dtype)
    geometry.update(scores=scores.reshape(b,q,m,a),weights=weights.reshape(b,q,m,a))
    return updates,support,geometry

# %%
# Reference 2: all queries/keys broadcast. Production qualified.
# All-masked maximum is replaced BEFORE subtraction/exponentiation. No learned
# projections, dropout, residual, depth lifting, or occlusion claim is implicit.
# Required diagnostic sample tensors already cost O(B*M*Q*A*C).
def bev_attention_vectorized(queries,image_features,reference_points,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth=.001,chunk_size=None):
    geometry,values,mask=_prepare(queries,image_features,reference_points,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth,chunk_size)
    b,q,c=queries.shape; m=image_features.shape[1]; a=reference_points.shape[2]
    raw=(queries.double().unsqueeze(-2)/math.sqrt(c)*values).sum(dim=-1)
    if not torch.isfinite(raw[mask]).all():
        raise ValueError('eligible dot scores overflow float64')
    scores=torch.where(mask,raw,0)
    supported=mask.any(dim=-1)
    maximum=torch.where(mask,scores,-torch.inf).amax(dim=-1,keepdim=True)
    maximum=torch.where(supported[...,None],maximum,0)
    shifted=torch.where(mask,scores-maximum,0)
    numerator=torch.where(mask,shifted.exp(),0)
    mass=numerator.sum(dim=-1,keepdim=True)
    weights=numerator/torch.where(mass>0,mass,1)
    updates=_convex_sum(values,weights).to(queries.dtype)
    geometry.update(scores=scores.reshape(b,q,m,a),weights=weights.reshape(b,q,m,a))
    return updates,supported,geometry

# %%
# Reference 3: complete query chunks, bounded corner/score work per chunk.
# Production qualified; output/diagnostics and retained backward state still
# scale with all Q. Map sanitization and validation also need spatial map work.
def bev_attention_chunked(queries,image_features,reference_points,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth=.001,chunk_size=None):
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
        raise ValueError('chunk_size must be a positive integer')
    # Validate even Q=0 before building an empty chunk.
    _,q,_=_validate_attention_inputs(queries,image_features,reference_points)
    updates,supports,parts=[],[],[]
    for start in range(0,max(q,1),chunk_size or max(q,1)):
        stop=min(q,start+(chunk_size or max(q,1)))
        query=queries[:,start:stop]; references=reference_points[:,start:stop]
        geometry,values,mask=_prepare(query,image_features,references,K,T_camera_from_ego,image_hw,camera_valid,feature_from_image,min_depth,chunk_size)
        b,n,c=query.shape; m=image_features.shape[1]; a=references.shape[2]
        raw=(query.double().unsqueeze(-2)/math.sqrt(c)*values).sum(dim=-1)
        if not torch.isfinite(raw[mask]).all():
            raise ValueError('eligible dot scores overflow float64')
        scores=torch.where(mask,raw,0)
        supported=mask.any(dim=-1)
        maximum=torch.where(mask,scores,-torch.inf).amax(dim=-1,keepdim=True)
        maximum=torch.where(supported[...,None],maximum,0)
        shifted=torch.where(mask,scores-maximum,0)
        numerator=torch.where(mask,shifted.exp(),0)
        mass=numerator.sum(dim=-1,keepdim=True)
        weights=numerator/torch.where(mass>0,mass,1)
        updates.append(_convex_sum(values,weights).to(queries.dtype)); supports.append(supported)
        geometry.update(scores=scores.reshape(b,n,m,a),weights=weights.reshape(b,n,m,a))
        parts.append(geometry)
    joined={name:torch.cat([part[name] for part in parts],dim=1 if name in ('weights','scores') else 2) for name in parts[0]}
    return torch.cat(updates,dim=1),torch.cat(supports,dim=1),joined

# %%
def benchmark_attention(inputs,chunk_size=16):
    """Opt-in shared-input forward timings and analytical temporary sizes."""
    from torch_judge.measurement import measure
    b,q,c=inputs['queries'].shape; m=inputs['image_features'].shape[1]; a=inputs['reference_points'].shape[2]
    expected=bev_attention_loop(**inputs)[0]
    reports={}
    for name,fn in [('loop',bev_attention_loop),('vectorized',bev_attention_vectorized),('chunked',bev_attention_chunked)]:
        actual=fn(**inputs,chunk_size=chunk_size)[0]
        query_work=1 if name=='loop' else min(q,chunk_size) if name=='chunked' else q
        reports[name]={'max_update_error':(actual-expected).abs().max().item() if actual.numel() else 0.,
            'analytical_score_work_bytes':b*query_work*m*a*8,
            'required_sample_output_bytes':b*m*q*a*c*inputs['queries'].element_size(),
            'storage_note':'Score work only; required diagnostics, geometry/gathers, map sanitization, validation and retained backward state excluded. Not allocator peaks.',
            'timing':measure(lambda:fn(**inputs,chunk_size=chunk_size),device=inputs['queries'].device,
                workload={'B':b,'Q':q,'M':m,'A':a,'C':c,'feature_hw':tuple(inputs['image_features'].shape[-2:]),
                    'dtype':str(inputs['queries'].dtype),'chunk_size':chunk_size,'mode':'forward with caller gradient flags'})}
    return reports

# %%
bev_attention=bev_attention_chunked

# %% [verifiers]
def _fixture():
    dtype=torch.float64
    rotation=torch.tensor([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]],dtype=dtype)
    pose=torch.eye(4,dtype=dtype).reshape(1,1,4,4).clone(); pose[0,0,:3,:3]=rotation
    u=torch.arange(4,dtype=dtype)[None,:].expand(3,4)
    v=torch.arange(3,dtype=dtype)[:,None].expand(3,4)
    features=torch.stack((u+2*v,3*u-v)).reshape(1,1,2,3,4)
    return dict(queries=torch.tensor([[[1.,.5]]],dtype=dtype),image_features=features,
        reference_points=torch.tensor([[[[2.,-.5,-.5]]]],dtype=dtype),
        K=torch.eye(3,dtype=dtype).reshape(1,1,3,3),T_camera_from_ego=pose,
        image_hw=torch.tensor([[[3,4]]],dtype=torch.int64),camera_valid=torch.ones(1,1,dtype=torch.bool),
        feature_from_image=torch.eye(3,dtype=dtype).reshape(1,1,3,3))


def _scalar_expected(inputs):
    qu,features,points=inputs['queries'],inputs['image_features'],inputs['reference_points']
    b,q,c=qu.shape; m=features.shape[1]; a=points.shape[2]; h,w=features.shape[-2:]
    uv=qu.new_zeros(b,m,q,a,2); depth=qu.new_zeros(b,m,q,a)
    projected=torch.zeros(b,m,q,a,dtype=torch.bool,device=qu.device)
    samples=qu.new_zeros(b,m,q,a,c); sampled=projected.clone()
    scores=torch.zeros(b,q,m,a,dtype=torch.float64,device=qu.device)
    weights=torch.zeros_like(scores); updates=torch.zeros_like(qu)
    support=torch.zeros(b,q,dtype=torch.bool,device=qu.device)
    for batch in range(b):
        for point in range(q):
            eligible=[]
            for camera in range(m):
                if not bool(inputs['camera_valid'][batch,camera]):
                    continue
                T=inputs['T_camera_from_ego'][batch,camera]; K=inputs['K'][batch,camera]
                Fimage=inputs['feature_from_image'][batch,camera]
                for anchor in range(a):
                    p=[float(x) for x in points[batch,point,anchor].detach()]
                    xyz=[math.fsum(float(T[row,col])*p[col] for col in range(3))+float(T[row,3]) for row in range(3)]
                    if xyz[2] <= inputs.get('min_depth',.001):
                        continue
                    image=[math.fsum(float(K[row,col])*xyz[col] for col in range(3)) for row in range(3)]
                    u,v=image[0]/image[2],image[1]/image[2]
                    ih,iw=inputs['image_hw'][batch,camera].tolist()
                    if not (0<=u<=iw-1 and 0<=v<=ih-1):
                        continue
                    projected[batch,camera,point,anchor]=True
                    uv[batch,camera,point,anchor]=qu.new_tensor([u,v]); depth[batch,camera,point,anchor]=xyz[2]
                    fx=float(Fimage[0,0])*u+float(Fimage[0,1])*v+float(Fimage[0,2])
                    fy=float(Fimage[1,0])*u+float(Fimage[1,1])*v+float(Fimage[1,2])
                    if not (0<=fx<=w-1 and 0<=fy<=h-1):
                        continue
                    sampled[batch,camera,point,anchor]=True
                    x0,y0=math.floor(fx),math.floor(fy); x1,y1=min(x0+1,w-1),min(y0+1,h-1)
                    dx,dy=fx-x0,fy-y0
                    for channel in range(c):
                        value=math.fsum(float(features[batch,camera,channel,yy,xx].detach())*weight
                            for xx,yy,weight in [(x0,y0,(1-dx)*(1-dy)),(x1,y0,dx*(1-dy)),(x0,y1,(1-dx)*dy),(x1,y1,dx*dy)])
                        samples[batch,camera,point,anchor,channel]=value
                    score=math.fsum(float(qu[batch,point,channel].detach())/math.sqrt(c)*float(samples[batch,camera,point,anchor,channel]) for channel in range(c))
                    if not math.isfinite(score):
                        raise ValueError('scalar eligible score overflow')
                    scores[batch,point,camera,anchor]=score
                    eligible.append((camera,anchor,score))
            if eligible:
                support[batch,point]=True
                maximum=max(score for _,_,score in eligible)
                masses=[math.exp(score-maximum) for _,_,score in eligible]
                total=math.fsum(masses)
                for (camera,anchor,_),mass in zip(eligible,masses):
                    weights[batch,point,camera,anchor]=mass/total
                for channel in range(c):
                    updates[batch,point,channel]=math.fsum(float(samples[batch,camera,point,anchor,channel])*float(weights[batch,point,camera,anchor]) for camera,anchor,_ in eligible)
    return updates,support,dict(uv=uv,depth=depth,projected=projected,samples=samples,sample_valid=sampled,scores=scores,weights=weights)


def _multi_scene():
    inputs=_fixture()
    inputs['queries']=torch.tensor([[[1.,.5],[-.5,1.]]],dtype=torch.float64)
    inputs['reference_points']=torch.tensor([[[[2.,-.5,-.5],[2.,-1.5,-2.5]],[[2.,-2.5,-.5],[-2.,-.5,-.5]]]],dtype=torch.float64)
    for name in ('K','T_camera_from_ego','image_hw','camera_valid','feature_from_image'):
        inputs[name]=inputs[name].repeat(1,2,*([1]*(inputs[name].ndim-2)))
    inputs['T_camera_from_ego'][0,1,0,3]=.5
    inputs['image_features']=torch.cat((inputs['image_features'],inputs['image_features']+2),dim=1)
    return inputs


@torch.enable_grad()
def _adapted_map_fixture():
    # Local frozen token backbone + one deterministic supervised adapter update.
    base=_fixture()['image_features'][:,0]
    tokens=base.flatten(2).transpose(1,2)
    backbone=nn.Linear(2,2,bias=False).double()
    with torch.no_grad():
        backbone.weight.copy_(torch.eye(2,dtype=torch.float64))
    backbone.eval().requires_grad_(False)
    features=backbone(tokens)
    adapter=nn.Linear(2,2).double()
    with torch.no_grad():
        adapter.weight.zero_(); adapter.bias.zero_()
    target=features @ features.new_tensor([[1.,-1.],[2.,.5]])+features.new_tensor([.25,-.5])
    loss=(adapter(features)-target).square().mean()
    loss.backward()
    with torch.no_grad():
        adapter.weight.sub_(.01*adapter.weight.grad); adapter.bias.sub_(.01*adapter.bias.grad)
    adapter.eval()
    maps=adapter(features).transpose(1,2).reshape(1,2,3,4).detach()
    lineage={'backbone':'local frozen identity Linear token backbone','adapter':'local Linear equivalent to 1x1 Conv',
        'training_steps':1,'learning_rate':.01,'loss':'mean squared feature target error',
        'adapter_weight':adapter.weight.detach().clone(),'adapter_bias':adapter.bias.detach().clone()}
    return maps[:,None],lineage


def make_eval_data():
    examples=[]
    examples.append(('one_key','baseline',_fixture(),'One key has weight one, regardless of query.',None))
    examples.append(('camera_height_keys','followup',_multi_scene(),'Camera-major / height-major key ordering, with one behind-camera anchor.',None))
    adapted=_fixture(); adapted['image_features'],lineage=_adapted_map_fixture()
    examples.append(('adapted_maps','followup',adapted,'Features come from a locally trained adapter on a frozen token backbone.',lineage))
    absent=_multi_scene(); absent['camera_valid'][0,1]=False; absent['image_features'][0,1]=float('nan')
    examples.append(('missing_camera_nan','production',absent,'Unavailable maps are sanitized before sampling and gradients.',None))
    masked=_fixture(); masked['camera_valid'].zero_(); masked['image_features'].fill_(float('nan'))
    examples.append(('all_masked','followup',masked,'All-masked queries produce zero update/false support/zero weights.',None))
    outside=_fixture(); outside['reference_points'][0,0,0]=torch.tensor([2.,-20.,-.5])
    examples.append(('outside_image','followup',outside,'Image support is geometric, not occlusion.',None))
    empty=_multi_scene(); empty['queries']=empty['queries'][:,:0]; empty['reference_points']=empty['reference_points'][:,:0]
    examples.append(('empty_queries','production',empty,'Preserve every named zero-length diagnostic axis.',None))
    huge=_fixture()
    for name,value in list(huge.items()):
        if value.is_floating_point():
            huge[name]=value.float()
    huge['queries'].fill_(1e20); huge['image_features'].fill_(3e20)
    examples.append(('wide_dot_scores','production',huge,'Float32 input dot products accumulate in float64.',None))
    cases={}
    for name,stage,inputs,purpose,generation in examples:
        cases[name]=EvalCase(name=name,stage=stage,purpose=purpose,inputs=inputs,expected=_scalar_expected(inputs),
            comparison=partial(compare_values,atol=2e-6,rtol=2e-6),generation=generation)
    invalid=_fixture(); invalid['queries'].fill_(float('nan'))
    cases['invalid_query']=EvalCase(name='invalid_query',stage='production',purpose='Queries must be finite.',inputs=invalid,expected_exception=ValueError)
    overflow=_fixture(); overflow['queries'].fill_(1e308); overflow['image_features'].fill_(1e308)
    cases['dot_overflow']=EvalCase(name='dot_overflow',stage='production',purpose='Eligible dot scores that exceed float64 are explicitly rejected.',inputs=overflow,expected_exception=ValueError)
    return cases


def _assert_case(implementation,case):
    errors=case.comparison(implementation(**case.inputs),case.expected)
    assert not errors,'\n'.join(errors)


def verify_baseline_one_key(bev_attention):
    inputs=_fixture()
    actual=bev_attention(**inputs)
    assert isinstance(actual,tuple) and len(actual)==3
    torch.testing.assert_close(actual[0],torch.tensor([[[.75,.5]]],dtype=torch.float64))
    assert torch.equal(actual[1],torch.ones(1,1,dtype=torch.bool))

def verify_baseline_full_diagnostics(bev_attention):
    _assert_case(bev_attention,make_eval_data()['one_key'])


def verify_followup_masks_heights_and_adaptation(bev_attention):
    for name in ('camera_height_keys','adapted_maps','all_masked','outside_image'):
        _assert_case(bev_attention,make_eval_data()[name])


def verify_followup_permutations_and_batches(bev_attention):
    @settings(max_examples=8,deadline=None,derandomize=True)
    @given(st.lists(st.integers(-4,4),min_size=4,max_size=4))
    def trial(numbers):
        inputs=_multi_scene(); inputs['queries']=torch.tensor(numbers,dtype=torch.float64).reshape(1,2,2)
        expected=_scalar_expected(inputs); actual=bev_attention(**inputs)
        assert not compare_values(actual,expected,atol=1e-10,rtol=1e-10)
        for name,value in list(inputs.items()):
            inputs[name]=value.repeat(2,*([1]*(value.ndim-1)))
        original=bev_attention(**inputs)
        perm=torch.tensor([1,0])
        for name in ('image_features','K','T_camera_from_ego','image_hw','camera_valid','feature_from_image'):
            inputs[name]=inputs[name][:,perm]
        swapped=bev_attention(**inputs)
        torch.testing.assert_close(swapped[0],original[0]); assert torch.equal(swapped[1],original[1])
        torch.testing.assert_close(swapped[2]['weights'],original[2]['weights'][:,:,perm])
        anchors=torch.tensor([1,0]); inputs['reference_points']=inputs['reference_points'][:,:,anchors]
        flipped=bev_attention(**inputs)
        torch.testing.assert_close(flipped[0],swapped[0])
        torch.testing.assert_close(flipped[2]['weights'],swapped[2]['weights'][:,:,:,anchors])
    trial()


def _scalar_gradient_update(inputs):
    # Independent differentiable scalar traversal. No production geometry,
    # gathers, broadcast attention, or other reference implementation is called.
    queries=inputs['queries']; maps=inputs['image_features']; references=inputs['reference_points']
    b,q,c=queries.shape; m=maps.shape[1]; a=references.shape[2]; h,w=maps.shape[-2:]
    batches=[]
    for batch in range(b):
        rows=[]
        for point in range(q):
            values=[]
            for camera in range(m):
                if not bool(inputs['camera_valid'][batch,camera]):
                    continue
                T=inputs['T_camera_from_ego'][batch,camera]; K=inputs['K'][batch,camera]; Fimage=inputs['feature_from_image'][batch,camera]
                for anchor in range(a):
                    p=references[batch,point,anchor]
                    xyz=torch.stack([sum(T[r,col]*p[col] for col in range(3))+T[r,3] for r in range(3)])
                    if float(xyz[2].detach()) <= inputs.get('min_depth',.001):
                        continue
                    image=torch.stack([sum(K[r,col]*xyz[col] for col in range(3)) for r in range(3)])
                    u,v=image[0]/image[2],image[1]/image[2]
                    ih,iw=inputs['image_hw'][batch,camera].tolist()
                    if not (0<=float(u.detach())<=iw-1 and 0<=float(v.detach())<=ih-1):
                        continue
                    fx=Fimage[0,0]*u+Fimage[0,1]*v+Fimage[0,2]; fy=Fimage[1,0]*u+Fimage[1,1]*v+Fimage[1,2]
                    if not (0<=float(fx.detach())<=w-1 and 0<=float(fy.detach())<=h-1):
                        continue
                    x0,y0=math.floor(float(fx.detach())),math.floor(float(fy.detach()))
                    x1,y1=min(x0+1,w-1),min(y0+1,h-1); dx,dy=fx-x0,fy-y0
                    value=sum(maps[batch,camera,:,yy,xx]*weight for xx,yy,weight in
                        [(x0,y0,(1-dx)*(1-dy)),(x1,y0,dx*(1-dy)),(x0,y1,(1-dx)*dy),(x1,y1,dx*dy)])
                    values.append(value)
            if values:
                score=torch.stack([sum(queries[batch,point,k]/math.sqrt(c)*value[k] for k in range(c)) for value in values])
                mass=(score-score.max()).exp(); probabilities=mass/mass.sum()
                rows.append(sum(value*probability for value,probability in zip(values,probabilities)))
            else:
                rows.append(queries[batch,point]*0)
        batches.append(torch.stack(rows))
    return torch.stack(batches)


def verify_production_backward_and_masked_zero(bev_attention):
    base=_multi_scene(); base['queries']=base['queries'][:,:1]; base['reference_points']=base['reference_points'][:,:1]
    names=('queries','image_features','reference_points','K','T_camera_from_ego','feature_from_image')
    actual_inputs={name:(value.detach().clone().requires_grad_(name in names) if value.is_floating_point() else value.clone()) for name,value in base.items()}
    oracle_inputs={name:(value.detach().clone().requires_grad_(name in names) if value.is_floating_point() else value.clone()) for name,value in base.items()}
    actual=bev_attention(**actual_inputs)[0]; expected=_scalar_gradient_update(oracle_inputs)
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
    pattern=torch.tensor([[[1.,-2.]]],dtype=torch.float64)
    actual_grad=torch.autograd.grad((actual*pattern).sum(),tuple(actual_inputs[name] for name in names))
    oracle_grad=torch.autograd.grad((expected*pattern).sum(),tuple(oracle_inputs[name] for name in names))
    for name,got,wanted in zip(names,actual_grad,oracle_grad):
        assert torch.isfinite(got).all(),name
        torch.testing.assert_close(got,wanted,atol=1e-9,rtol=1e-9,msg=lambda msg:f'{name}: {msg}')
    assert torch.autograd.gradcheck(lambda qu,fe,po:bev_attention(**dict(base,queries=qu,image_features=fe,reference_points=po))[0],
        tuple(base[name].detach().requires_grad_() for name in ('queries','image_features','reference_points')))
    inputs=make_eval_data()['all_masked'].inputs
    for name in names:
        inputs[name]=inputs[name].detach().requires_grad_()
    output=bev_attention(**inputs)[0]
    gradients=torch.autograd.grad(output.sum(),tuple(inputs[name] for name in names),allow_unused=True)
    for gradient in gradients:
        if gradient is not None:
            assert torch.isfinite(gradient).all() and torch.equal(gradient,torch.zeros_like(gradient))
    # Zero-valued VALID features still need ordinary nonzero value derivatives.
    inputs=_fixture(); inputs['image_features']=torch.zeros_like(inputs['image_features'],requires_grad=True)
    output=bev_attention(**inputs)[0]
    grad=torch.autograd.grad(output.sum(),inputs['image_features'])[0]
    assert torch.isfinite(grad).all() and torch.allclose(grad.sum(),torch.tensor(2.,dtype=grad.dtype))


def verify_production_numerics_strides_and_chunks(bev_attention):
    for name in ('missing_camera_nan','empty_queries','wide_dot_scores'):
        _assert_case(bev_attention,make_eval_data()[name])
    inputs=_multi_scene()
    for name in ('queries','reference_points'):
        value=inputs[name]
        inputs[name]=torch.stack((value,value),dim=-1)[...,0]
    inputs['image_features']=inputs['image_features'].transpose(-1,-2)
    inputs['image_hw']=inputs['image_hw'].flip(-1)
    expected=_scalar_expected(inputs)
    originals={name:value.clone() for name,value in inputs.items()}
    for chunk_size in (None,1,3):
        assert not compare_values(bev_attention(**inputs,chunk_size=chunk_size),expected,atol=1e-10,rtol=1e-10)
    for name,value in inputs.items():
        assert torch.equal(value,originals[name]),name


def verify_production_validation(bev_attention):
    inputs=_fixture()
    replacements=[('queries',torch.full_like(inputs['queries'],float('nan'))),
        ('reference_points',torch.zeros(1,1,0,3,dtype=torch.float64)),
        ('image_features',inputs['image_features'].float()),('camera_valid',torch.ones(1,1)),
        ('K',torch.zeros_like(inputs['K'])),('feature_from_image',torch.zeros_like(inputs['feature_from_image'])),
        ('chunk_size',0),('chunk_size',True),('min_depth',0)]
    for key,value in replacements:
        try:
            bev_attention(**dict(inputs,**{key:value}))
        except ValueError:
            continue
        raise AssertionError(f'invalid {key} must raise ValueError')
    overflow=make_eval_data()['dot_overflow'].inputs
    try:
        bev_attention(**overflow)
    except ValueError:
        return
    raise AssertionError('eligible float64 dot overflow must raise ValueError')


# %%
from torch_judge import evaluate
if not evaluate('0113_bev_cross_attention_eval.py',bev_attention):
    raise SystemExit(1)
