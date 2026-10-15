"""Multi-camera projection: complete references and independent verifiers."""
# %%
import math
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values, evaluate
from torch_judge.measurement import measure

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
# Reference 1: camera loops show each row-point transform. Qualified production
# for float32/64; Python loops make camera scaling expensive. Temporary O(B*N),
# required outputs O(B*M*N). Every version shares validation, not algorithms.
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
# Reference 2: broadcast cameras and points. O(B*M*N) forward temporary work.
# Qualified production; vectorization reduces Python dispatch but does not
# improve asymptotic output size. No universal speed advantage is promised.
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
# Reference 3: point chunks bound forward geometric temporaries O(B*M*chunk).
# Output and autograd-retained state still scale with B*M*N. Qualified
# production; full-size default, choose chunk_size explicitly for scale.
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
project_points = project_chunked

# %%
def benchmark_projection(B=2,M=4,N=4096,chunk_size=512,device='cpu',dtype=torch.float32):
    """Opt-in wall-clock experiment; includes validation and Python dispatch."""
    p = torch.zeros(B,N,3,device=device,dtype=dtype); p[...,2] = 2
    inputs = dict(points=p,K=torch.eye(3,device=device,dtype=dtype).expand(B,M,3,3),
                  T_camera_from_ego=torch.eye(4,device=device,dtype=dtype).expand(B,M,4,4),
                  image_hw=torch.full((B,M,2),32,device=device,dtype=torch.int64),
                  point_valid=torch.ones(B,N,device=device,dtype=torch.bool),
                  camera_valid=torch.ones(B,M,device=device,dtype=torch.bool),chunk_size=chunk_size)
    expected = project_per_camera(**inputs)
    reports = {}
    for name,fn in [('per_camera',project_per_camera),('broadcast',project_broadcast),('chunked',project_chunked)]:
        actual = fn(**inputs)
        errors = compare_values(actual,expected)
        assert not errors,errors
        reports[name] = {'timing':measure(lambda:fn(**inputs),device=device,
                        workload={'B':B,'M':M,'N':N,'chunk_size':chunk_size,'dtype':str(dtype)}),
                        'max_uv_error':(actual[0]-expected[0]).abs().max().item() if N else 0,
                        'storage_scope':'forward geometry only; outputs and retained backward state excluded'}
    return reports

# %% [verifiers]
def _scene(dtype=torch.float64):
    # Ego x-forward/y-left/z-up -> camera x-right/y-down/z-forward.
    R = torch.tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]], dtype=dtype)
    T = torch.eye(4, dtype=dtype).reshape(1, 1, 4, 4)
    T[0, 0, :3, :3] = R
    return dict(points=torch.tensor([[[2., 0., 0.], [2., -2., -2.],
                                     [-1., 0., 0.], [2., 2., 0.], [0.001, 0., 0.]]], dtype=dtype),
                K=torch.eye(3, dtype=dtype).reshape(1, 1, 3, 3),
                T_camera_from_ego=T, image_hw=torch.tensor([[[2, 2]]]),
                point_valid=torch.ones(1, 5, dtype=torch.bool),
                camera_valid=torch.ones(1, 1, dtype=torch.bool))


def _scalar_expected(inputs):
    # Scalar pinhole equations, deliberately independent of tensor broadcasting.
    p, K, T = (inputs[k].detach() for k in ('points', 'K', 'T_camera_from_ego'))
    B, N, _ = p.shape
    M = K.shape[1]
    uv = torch.zeros(B, M, N, 2, dtype=p.dtype, device=p.device)
    depth = torch.zeros(B, M, N, dtype=p.dtype, device=p.device)
    valid = torch.zeros(B, M, N, dtype=torch.bool, device=p.device)
    for b in range(B):
        for m in range(M):
            for n in range(N):
                if not (inputs['point_valid'][b,n] and inputs['camera_valid'][b,m]):
                    continue
                xyz = [sum(float(T[b,m,i,j]) * float(p[b,n,j]) for j in range(3)) + float(T[b,m,i,3]) for i in range(3)]
                if xyz[2] <= float(p.new_tensor(inputs.get('min_depth', .001))):
                    continue
                h = [sum(float(K[b,m,i,j]) * xyz[j] for j in range(3)) for i in range(3)]
                u, v = h[0]/h[2], h[1]/h[2]
                height, width = inputs['image_hw'][b,m].tolist()
                if 0 <= u <= width-1 and 0 <= v <= height-1:
                    uv[b,m,n] = torch.tensor([u,v], dtype=p.dtype, device=p.device)
                    depth[b,m,n] = xyz[2]
                    valid[b,m,n] = True
    return uv, depth, valid


def make_eval_data():
    examples = [('optical_axis_and_edges', 'baseline', _scene())]
    multi = _scene()
    for key in multi:
        multi[key] = multi[key].repeat(2, 2 if key in ('K','T_camera_from_ego','image_hw','camera_valid') else 1, *([1]*(multi[key].ndim-2)))
    multi['T_camera_from_ego'][1,1,0,3] = 1
    multi['camera_valid'][0,1] = False
    examples.append(('batches_translation_missing_camera', 'followup', multi))
    rotated = _scene()
    turn = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    rotated['T_camera_from_ego'][0,0,:3,:3] = turn @ rotated['T_camera_from_ego'][0,0,:3,:3]
    rotated['K'][0,0,0,2] = 1
    rotated['K'][0,0,1,2] = 1
    rotated['image_hw'].fill_(3)
    examples.append(('ninety_degree_camera_turn', 'followup', rotated))
    missing = _scene(); missing['camera_valid'].fill_(False)
    examples.append(('all_cameras_missing', 'followup', missing))
    empty = _scene()
    empty['points'] = empty['points'][:,:0]
    empty['point_valid'] = empty['point_valid'][:,:0]
    examples.append(('empty_points', 'production', empty))
    masked = _scene()
    masked['points'][0,1] = float('nan')
    masked['point_valid'][0,1] = False
    examples.append(('masked_nan_padding', 'production', masked))
    strides = _scene(torch.float32)
    storage = torch.zeros(1,5,6)
    storage[...,::2] = strides['points']
    strides['points'] = storage[...,::2]
    strides['chunk_size'] = 2
    examples.append(('strided_point_chunks', 'production', strides))
    return {name: EvalCase(name=name, stage=stage, purpose=name.replace('_',' '),
                          inputs=inputs, expected=_scalar_expected(inputs),
                          comparison=partial(compare_values, atol=1e-6, rtol=1e-6))
            for name,stage,inputs in examples}


def _assert_case(project_points, case):
    actual = project_points(**case.inputs)
    errors = compare_values(actual, case.expected, atol=1e-6, rtol=1e-6)
    assert not errors, errors


def verify_baseline_hand_projection(project_points):
    inputs = _scene()
    actual = project_points(**inputs)
    expected = (torch.tensor([[[[0.,0.],[1.,1.],[0.,0.],[0.,0.],[0.,0.]]]], dtype=torch.float64),
                torch.tensor([[[2.,2.,0.,0.,0.]]], dtype=torch.float64),
                torch.tensor([[[True,True,False,False,False]]]))
    assert not compare_values(actual, expected)


def verify_followup_batch_support(project_points):
    for name in ('batches_translation_missing_camera','ninety_degree_camera_turn','all_cameras_missing'):
        _assert_case(project_points, make_eval_data()[name])


def verify_production_fixtures(project_points):
    for case in make_eval_data().values():
        _assert_case(project_points, case)


def verify_production_calibration_and_metadata(project_points):
    bad_inputs = []
    for key,index,value in [('K',(0,0,0,0),0.), ('K',(0,0,2,0),.1),
                            ('T_camera_from_ego',(0,0,0,0),2.),
                            ('T_camera_from_ego',(0,0,3,3),0.),
                            ('image_hw',(0,0,0),0), ('points',(0,0,0),float('nan'))]:
        inputs = _scene()
        inputs[key][index] = value
        bad_inputs.append(inputs)
    for kwargs in ({'min_depth':0}, {'min_depth':float('nan')}, {'chunk_size':0}, {'chunk_size':1.5}):
        bad_inputs.append(dict(_scene(), **kwargs))
    wrong = _scene(); wrong['T_camera_from_ego'][0,0,:3,:3] *= -1; bad_inputs.append(wrong)
    wrong = _scene(); wrong['K'][0,0,0,:] = wrong['K'][0,0,1,:]; wrong['K'][0,0,0,0] = 1; wrong['K'][0,0,1,0] = 1; bad_inputs.append(wrong)
    wrong = _scene(); wrong['points'] = wrong['points'][0]; bad_inputs.append(wrong)
    wrong = _scene(); wrong['point_valid'] = wrong['point_valid'].float(); bad_inputs.append(wrong)
    wrong = _scene(); wrong['K'] = wrong['K'].float(); bad_inputs.append(wrong)
    wrong = _scene(); wrong['image_hw'] = wrong['image_hw'].float(); bad_inputs.append(wrong)
    for inputs in bad_inputs:
        try:
            project_points(**inputs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid metadata/calibration must raise ValueError')


def verify_production_gradients_and_no_mutation(project_points):
    inputs = _scene()
    inputs['points'] = torch.tensor([[[2.,-.5,-.6]]], dtype=torch.float64, requires_grad=True)
    inputs['point_valid'] = torch.ones(1,1,dtype=torch.bool)
    inputs['K'].requires_grad_(); inputs['T_camera_from_ego'].requires_grad_()
    originals = {key:value.detach().clone() for key,value in inputs.items()}
    def smooth(p,k,t):
        u,d,_ = project_points(**dict(inputs, points=p,K=k,T_camera_from_ego=t))
        return torch.cat((u.flatten(),d.flatten()))
    assert torch.autograd.gradcheck(smooth, (inputs['points'], inputs['K'], inputs['T_camera_from_ego']),
                                    eps=1e-6, atol=1e-5, rtol=1e-4)
    for key,value in inputs.items():
        assert torch.equal(value.detach(), originals[key])
    # A masked NaN must also have finite zero derivatives.
    masked = _scene(); masked['points'][0,1] = float('nan'); masked['point_valid'][0,1] = False
    masked['points'].requires_grad_()
    u,d,_ = project_points(**masked)
    (u.sum()+d.sum()).backward()
    assert torch.isfinite(masked['points'].grad).all()
    assert torch.equal(masked['points'].grad[0,1], torch.zeros(3,dtype=torch.float64))


def verify_production_generated_translation(project_points):
    @settings(max_examples=15, deadline=None, derandomize=True)
    @given(st.floats(min_value=-.25,max_value=.25,allow_nan=False,allow_infinity=False))
    def trial(translation):
        inputs = _scene(); inputs['T_camera_from_ego'][0,0,0,3] = translation
        actual = project_points(**inputs)
        assert not compare_values(actual, _scalar_expected(inputs), atol=1e-6, rtol=1e-6)
    trial()

# %%
if not evaluate('0103_multi_camera_project_eval.py', project_points):
    raise SystemExit(1)
