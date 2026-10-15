"""Bilinear camera features: independent oracles and staged practice checks."""
# %%
import math
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values, evaluate

# %%
# Shared contract validation is separate from interpolation. Invalid metadata
# raises ValueError; out-of-grid points are ordinary unsupported observations.
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
# Version 1: explicit camera/point loops and four corners. Qualified production
# float32/64 CPU correctness (including gradients), but per-point dispatch and
# scalar support decisions synchronize on accelerators. O(B*M*N*C) output and
# graph state, with O(C) interpolation temporaries. chunk_size validates but
# adds no benefit to this already point-at-a-time version.
def sample_features_four_corners(features, uv, valid, feature_from_image, chunk_size=None):
    active = validate_sampling(features,uv,valid,feature_from_image,chunk_size)
    B,M,C,H,W = features.shape
    N = uv.shape[2]
    safe_f = torch.where(active[:,:,None,None,None],features,0.)
    safe_uv = torch.where(valid[...,None],uv,0.)
    zero = safe_f.reshape(-1)[:1].sum()*0 + safe_uv.reshape(-1)[:1].sum()*0 + feature_from_image.reshape(-1)[:1].sum()*0
    rows, masks = [], []
    for b in range(B):
        camera_rows, camera_masks = [], []
        for m in range(M):
            values, flags = [], []
            for n in range(N):
                if not valid[b,m,n]:
                    flags.append(False); values.append(features.new_zeros(C)+zero)
                    continue
                xy = feature_from_image[b,m,:2,:2] @ safe_uv[b,m,n] + feature_from_image[b,m,:2,2]
                x,y = xy.unbind()
                inside = bool(torch.isfinite(xy).all()) and bool(0 <= x <= W-1 and 0 <= y <= H-1)
                flags.append(inside)
                if not inside:
                    values.append(features.new_zeros(C)+zero)
                    continue
                x0,y0 = int(torch.floor(x)),int(torch.floor(y))
                x1,y1 = min(x0+1,W-1),min(y0+1,H-1)
                dx,dy = x-x0,y-y0
                values.append((1-dy)*((1-dx)*safe_f[b,m,:,y0,x0]+dx*safe_f[b,m,:,y0,x1])
                              +dy*((1-dx)*safe_f[b,m,:,y1,x0]+dx*safe_f[b,m,:,y1,x1]))
            camera_rows.append(torch.stack(values) if N else features.new_empty(0,C)+zero)
            camera_masks.append(torch.tensor(flags,dtype=torch.bool,device=features.device))
        rows.append(torch.stack(camera_rows)); masks.append(torch.stack(camera_masks))
    return torch.stack(rows),torch.stack(masks)

# %%
# Version 2: vectorized four-corner gathers. Qualified production float32/64,
# any storage strides; no per-point Python calls. Required O(B*M*N*C) output,
# four corner tensors of that size plus O(B*M*N) coordinates/indices. Backward
# retains gather/weight state. It accepts chunk_size but processes all N at once;
# choose version 3 when point-temporary storage matters.
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
# Version 3: complete chunked gathers, same boundary/stability semantics.
# Qualified production float32/64. Corner work O(B*M*chunk_size*C), plus output
# O(B*M*N*C). Safe map sanitization costs O(B*M*C*H*W) independent of N.
# Concatenation temporarily duplicates output; autograd retains work for all
# chunks, so this is a forward temporary bound, not a constant-memory backward.
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
# Opt in after correctness: workload-specific latency and scalar-oracle error.
# measure reports device, warmup/repeats, synchronization and variability.
# Analytic corner value bytes describe the logical corners in one step;
# teaching corners are views. This is not measured peak or backward memory.
def sampling_experiment(N=256,C=16,chunk_size=64,device='cpu',dtype=torch.float32):
    from torch_judge.measurement import measure
    gen = torch.Generator().manual_seed(104)
    features = torch.randn(1,2,C,8,12,generator=gen,dtype=dtype).to(device)
    uv = torch.rand(1,2,N,2,generator=gen,dtype=dtype).to(device)*features.new_tensor([11.,7.])
    inputs = dict(features=features,uv=uv,valid=torch.ones(1,2,N,dtype=torch.bool,device=device),feature_from_image=torch.eye(3,dtype=dtype,device=device)[None,None].expand(1,2,3,3))
    expected = scalar_expected(**inputs)[0]
    reports = {}
    for name,fn in [('four_corners',sample_features_four_corners),('vectorized',sample_features_vectorized),('chunked',sample_features_chunked)]:
        actual = fn(**inputs,chunk_size=chunk_size)[0]
        reports[name] = dict(max_abs_error=(actual-expected).abs().max().item() if N else 0.,
                            timing=measure(lambda:fn(**inputs,chunk_size=chunk_size),device=device,workload={'B':1,'M':2,'N':N,'C':C,'H':8,'W':12,'dtype':str(dtype),'seed':104}),
                            analytic_corner_value_bytes=4*(1 if name=='four_corners' else 2*(min(N,chunk_size) if name=='chunked' else N))*C*features.element_size())
    return reports

# %%
sample_features = sample_features_chunked

# %% [verifiers]
def scalar_expected(features, uv, valid, feature_from_image):
    """Detached Python arithmetic, weighted corner list, independent of gathers."""
    B, M, C, H, W = features.shape
    N = uv.shape[2]
    values = features.new_zeros(B, M, N, C)
    supported = torch.zeros(B, M, N, dtype=torch.bool, device=features.device)
    for b in range(B):
        for m in range(M):
            a = feature_from_image[b,m].detach().tolist()
            for n in range(N):
                if not valid[b,m,n]:
                    continue
                u, v = uv[b,m,n].detach().tolist()
                x = a[0][0]*u + a[0][1]*v + a[0][2]
                y = a[1][0]*u + a[1][1]*v + a[1][2]
                if not (0 <= x <= W-1 and 0 <= y <= H-1):
                    continue
                supported[b,m,n] = True
                ix, iy = math.floor(x), math.floor(y)
                wx, wy = x-ix, y-iy
                for c in range(C):
                    total = 0.
                    for yy, weight_y in [(iy, 1-wy), (min(iy+1,H-1), wy)]:
                        for xx, weight_x in [(ix, 1-wx), (min(ix+1,W-1), wx)]:
                            total += weight_x*weight_y*features[b,m,c,yy,xx].item()
                    values[b,m,n,c] = total
    return values, supported


def make_eval_data():
    cases = {}
    def add(name, stage, purpose, features, uv, valid=None, affine=None, expected=None):
        B,M = features.shape[:2]
        valid = torch.ones(uv.shape[:-1], dtype=torch.bool, device=features.device) if valid is None else valid
        affine = torch.eye(3, dtype=features.dtype, device=features.device).expand(B,M,3,3).clone() if affine is None else affine
        inputs = dict(features=features, uv=uv, valid=valid, feature_from_image=affine)
        cases[name] = EvalCase(name=name, stage=stage, purpose=purpose, inputs=inputs,
                              expected=scalar_expected(**inputs) if expected is None else expected,
                              comparison=partial(compare_values, atol=1e-7, rtol=1e-6))
    f = torch.tensor([[[[[0.,1.,2.],[10.,11.,12.]]]]], dtype=torch.float64)
    add('numbered_grid', 'baseline', 'x+10*y: corners, fractional center, final row/column, outside.', f,
        torch.tensor([[[[0.25,0.5],[2.,1.],[3.,0.]]]], dtype=f.dtype),
        expected=(torch.tensor([[[[5.25],[12.],[0.]]]], dtype=f.dtype), torch.tensor([[[True,True,False]]])))
    affine = torch.tensor([[[[0.5,0.,-1.],[0.,0.25,-0.5],[0.,0.,1.]]]], dtype=f.dtype)
    add('crop_resize', 'baseline', 'Image (3,4) maps to feature (0.5,0.5); no shape-ratio inference.', f,
        torch.tensor([[[[3.,4.],[6.,6.]]]],dtype=f.dtype), affine=affine,
        expected=(torch.tensor([[[[5.5],[12.]]]],dtype=f.dtype),torch.ones(1,1,2,dtype=torch.bool)))
    for H,W in [(1,4),(3,1),(1,1)]:
        add(f'singleton_{H}x{W}', 'followup', 'Collapsed axis contributes once; last center is supported.',
            torch.arange(H*W,dtype=torch.float64).reshape(1,1,1,H,W),
            torch.tensor([[[[0.,0.],[float(W-1),float(H-1)],[-0.1,0.]]]],dtype=torch.float64))
    f2 = torch.arange(2*2*3*3*5,dtype=torch.float64).reshape(2,2,3,3,5)
    uv2 = torch.tensor([0.4,1.3,4.,2.,-0.1,1.],dtype=torch.float64).reshape(1,1,3,2).expand(2,2,3,2).clone()
    add('rectangular_views', 'followup', 'Two batches, two views, three channels: separate axes and cameras.', f2,uv2)
    masked = f2.clone(); masked[:,1] = float('nan')
    uv_masked = uv2.clone(); uv_masked[:,1] = float('nan')
    valid = torch.ones(2,2,3,dtype=torch.bool); valid[:,1] = False
    add('masked_camera_nan', 'production', 'Unavailable camera values and coordinates never enter interpolation.',masked,uv_masked,valid)
    add('empty_points', 'production', 'Empty output retains shape and graph connection.',f2,torch.empty(2,2,0,2,dtype=f2.dtype))
    add('strided_maps', 'production', 'Transposed feature storage is still indexed in declared y,x order.',f2.transpose(-1,-2),uv2)
    grad_f = torch.tensor([[[[[0.,1.,2.],[10.,11.,12.],[20.,21.,22.]]]]],dtype=torch.float64,requires_grad=True)
    grad_uv = torch.tensor([[[[0.3,0.7],[1.4,1.3]]]],dtype=torch.float64,requires_grad=True)
    add('interior_gradient', 'production', 'Live tensors: differentiate through features, uv, and affine.',grad_f,grad_uv)
    add('outside_all', 'production', 'All unsupported points produce finite zeros and zero gradients.',grad_f.detach().clone().requires_grad_(),torch.tensor([[[[-2.,-3.],[9.,8.]]]],dtype=torch.float64,requires_grad=True))
    return cases


def assert_cases(implementation, names):
    for name in names:
        case = make_eval_data()[name]
        errors = case.comparison(implementation(**case.inputs), case.expected)
        assert not errors, (name, errors)


def verify_baseline_numbered_grid(sample_features):
    assert_cases(sample_features, ['numbered_grid','crop_resize'])


def verify_followup_shapes_and_singletons(sample_features):
    assert_cases(sample_features, ['rectangular_views','singleton_1x4','singleton_3x1','singleton_1x1'])


def verify_production_masked_empty_strided(sample_features):
    assert_cases(sample_features, ['masked_camera_nan','empty_points','strided_maps','outside_all'])


def verify_production_gradient(sample_features):
    case = make_eval_data()['interior_gradient']
    f,uv,a = (case.inputs[k].detach().clone().requires_grad_() for k in ('features','uv','feature_from_image'))
    # The affine bottom row is a fixed contract; gradcheck only its top two rows.
    top = a[:,:,:2].detach().clone().requires_grad_()
    def run(f,uv,top):
        affine = torch.cat([top, a[:,:,2:].detach()],dim=2)
        return sample_features(f,uv,case.inputs['valid'],affine)[0]
    assert torch.autograd.gradcheck(run,(f,uv,top),eps=1e-6,atol=1e-5,rtol=1e-4)
    out = sample_features(f,uv,case.inputs['valid'],a)[0]
    guv, = torch.autograd.grad(out.sum(),uv)
    torch.testing.assert_close(guv,torch.tensor([[[[1.,10.],[1.,10.]]]],dtype=uv.dtype))
    for name in ['outside_all','empty_points','masked_camera_nan']:
        inputs = make_eval_data()[name].inputs
        f = inputs['features'].detach().clone().requires_grad_()
        uv = inputs['uv'].detach().clone().requires_grad_()
        y,_ = sample_features(f,uv,inputs['valid'],inputs['feature_from_image'])
        gf,gu = torch.autograd.grad(y.sum(),(f,uv))
        assert torch.isfinite(gf).all() and torch.isfinite(gu).all()
        if name != 'masked_camera_nan':
            assert not gf.count_nonzero() and not gu.count_nonzero()
        else:
            assert not gf[:,1].count_nonzero() and not gu[:,1].count_nonzero()


def verify_production_chunks_and_no_mutation(sample_features):
    for name,case in make_eval_data().items():
        snapshots = {k:v.clone() for k,v in case.inputs.items()}
        for size in (1,2,20):
            got = sample_features(**case.inputs,chunk_size=size)
            assert not case.comparison(got,case.expected), name
        for k,v in case.inputs.items():
            torch.testing.assert_close(v,snapshots[k],equal_nan=True)


def verify_production_float32_and_device(sample_features):
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        for name in ['crop_resize','rectangular_views','masked_camera_nan','empty_points']:
            case = make_eval_data()[name]
            inputs = {k:(v.to(device=device,dtype=torch.float32) if v.is_floating_point() else v.to(device)) for k,v in case.inputs.items()}
            expected = scalar_expected(**inputs)
            got = sample_features(**inputs,chunk_size=1)
            assert not compare_values(got,expected,atol=2e-5,rtol=2e-6), name


def verify_production_validation(sample_features):
    inputs = make_eval_data()['numbered_grid'].inputs
    variants = [dict(uv=inputs['uv'].float()), dict(valid=inputs['valid'].float()),
                dict(features=inputs['features'].squeeze(0)), dict(chunk_size=0), dict(chunk_size=True),
                dict(feature_from_image=torch.zeros(1,1,3,3,dtype=torch.float64)),
                dict(uv=torch.full_like(inputs['uv'],float('inf'))),
                dict(features=torch.full_like(inputs['features'],float('nan')))]
    for changes in variants:
        try:
            sample_features(**(inputs|changes))
        except ValueError:
            continue
        raise AssertionError(f'Expected ValueError for {list(changes)}')


def verify_production_affine_property(sample_features):
    @settings(max_examples=18,derandomize=True,deadline=None)
    @given(H=st.integers(2,5),W=st.integers(2,6),ax=st.integers(-5,5),ay=st.integers(-5,5),bias=st.integers(-8,8))
    def trial(H,W,ax,ay,bias):
        y,x = torch.meshgrid(torch.arange(H,dtype=torch.float64),torch.arange(W,dtype=torch.float64),indexing='ij')
        f = (ax*x+ay*y+bias)[None,None,None]
        uv = torch.tensor([[[[0.375*(W-1),0.625*(H-1)]]]],dtype=f.dtype)
        got,mask = sample_features(f,uv,torch.ones(1,1,1,dtype=torch.bool),torch.eye(3,dtype=f.dtype)[None,None])
        expected = ax*uv[...,0]+ay*uv[...,1]+bias
        torch.testing.assert_close(got[...,0],expected)
        assert mask.all()
    trial()

# %%
if not evaluate('0104_bilinear_features_eval.py',sample_features):
    raise SystemExit(1)
