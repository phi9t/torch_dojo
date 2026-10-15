"""Three complete masked-view means: transparent, stable, and point chunks."""
# %%
import math
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values

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
# Reference 1: the equation, preserving input precision. Baseline/followup only:
# modest finite confidence and sample magnitudes. Raw sums/products can overflow.
# O(B*M*N*C) work/storage; masked values are removed BEFORE multiplication.
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
# Reference 2: scale confidence by its per-point max, then normalize BEFORE
# multiplying features. float32 inputs accumulate in float64; output keeps dtype.
# Qualifies production. Float64 remains float64; scaling protects extreme finite
# weights and features. Per-point/channel conditioning plus a convex-bound
# correction keeps the forward mean finite at finfo.max. Conditioning is detached;
# a straight-through rounding correction preserves ordinary mean derivatives.
# Extreme confidence derivatives can overflow even for a finite forward output:
# no arbitrary-precision forward/gradient guarantee for every real ratio.
# Complete algorithm: no call to the transparent reference. O(B*M*N*C) storage.
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
# Reference 3: same stable equation, processing Q points at a time. Production.
# Temporary reduction storage O(B*M*Q*C), required output O(B*N*C). Validation
# scans inputs separately; boolean selection also has an input-sized temporary.
# Autograd may retain per-chunk intermediates across the whole call.
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
def explore_overflow():
    """Opt-in counterexample; same finite inputs, raw weights overflow."""
    values = torch.tensor([[[[2.,-4.]],[[6.,8.]]]],dtype=torch.float32)
    weights = torch.full((1,2,1),3e38,dtype=torch.float32)
    valid = torch.ones_like(weights,dtype=torch.bool)
    return {'inputs':(values,valid,weights),'transparent':fuse_transparent(values,valid,weights),
            'stable':fuse_stable(values,valid,weights), 'expected':torch.tensor([[[4.,2.]]])}

# %%
def benchmark_fusion(shape=(2,6,2048,32), chunk_size=256, device='cpu'):
    from torch_judge.measurement import measure
    generator = torch.Generator(device=device).manual_seed(105)
    samples = torch.randn(shape,device=device,generator=generator)
    weights = torch.rand(shape[:3],device=device,generator=generator)
    valid = torch.rand(shape[:3],device=device,generator=generator) > .2
    expected = fuse_stable(samples,valid,weights)
    reports = {}
    for name, fn in [('transparent',fuse_transparent),('stable',fuse_stable),('chunked',fuse_chunked)]:
        actual = fn(samples,valid,weights,chunk_size=chunk_size)
        reports[name] = {'max_error':(actual[0]-expected[0]).abs().max().item(),
            'measurement':measure(lambda:fn(samples,valid,weights,chunk_size=chunk_size),device=device,
                workload={'shape':shape,'dtype':str(samples.dtype),'chunk_size':chunk_size,'seed':105})}
    return reports

# %%
fuse_views = fuse_chunked

# %% [verifiers]
def _scalar_expected(samples, valid, confidence):
    # Independent Python scalar oracle using math.fsum, never a tensor reduction.
    b,m,n,c = samples.shape
    output = torch.zeros((b,n,c),dtype=samples.dtype,device=samples.device)
    support = torch.zeros((b,n),dtype=torch.bool,device=samples.device)
    for batch in range(b):
        for point in range(n):
            cameras = [view for view in range(m) if bool(valid[batch,view,point]) and float(confidence[batch,view,point]) > 0]
            if not cameras:
                continue
            support[batch,point] = True
            scale = max(float(confidence[batch,view,point]) for view in cameras)
            weights = [float(confidence[batch,view,point])/scale for view in cameras]
            mass = math.fsum(weights)
            for channel in range(c):
                output[batch,point,channel] = math.fsum(float(samples[batch,view,point,channel])*(weight/mass) for view,weight in zip(cameras,weights))
    return output,support


def make_eval_data():
    def case(name,stage,values,weights,flags=None,purpose='',chunk_size=None):
        samples = torch.as_tensor(values,dtype=torch.float64)
        confidence = torch.as_tensor(weights,dtype=samples.dtype)
        valid = torch.ones_like(confidence,dtype=torch.bool) if flags is None else torch.as_tensor(flags,dtype=torch.bool)
        inputs = dict(samples=samples,valid=valid,confidence=confidence,chunk_size=chunk_size)
        return EvalCase(name=name,stage=stage,purpose=purpose,inputs=inputs,
            expected=_scalar_expected(samples,valid,confidence),comparison=partial(compare_values,atol=1e-10,rtol=1e-10))
    cases = [
        case('equal_views','baseline',[[[[2.,4.]],[[6.,8.]]]],[[[1.],[1.]]],purpose='Predict [4,6] before calling the answer.'),
        case('one_view','baseline',[[[[3.,-2.]],[[99.,99.]]]],[[[2.],[7.]]],[[[True],[False]]],purpose='A missing view belongs in neither sum nor denominator.'),
        case('confidence_weights','followup',[[[[2.,4.]],[[6.,8.]]]],[[[1.],[3.]]],purpose='Unnormalized confidence gives [5,7].'),
        case('zero_mass','followup',[[[[3.,4.]],[[5.,6.]]]],[[[0.],[0.]]],purpose='Geometric support alone does not imply positive mass.'),
        case('masked_nan','production',[[[[2.,4.]],[[float('nan'),float('nan')]]]],[[[2.],[float('nan')]]],[[[True],[False]]],purpose='Sanitize before arithmetic; NaN*0 is NaN.'),
        case('weight_overflow','production',[[[[2.,-4.]],[[6.,8.]]]],[[[1e308],[1e308]]],purpose='Raw mass overflows even though the mean is finite.',chunk_size=1),
        case('empty_points','followup',torch.empty(2,3,0,4),torch.empty(2,3,0),purpose='Return [2,0,4] and [2,0] without reductions over empty points.'),
        case('all_missing','followup',[[[[3.,4.]],[[5.,6.]]]],[[[1.],[2.]]],[[[False],[False]]],purpose='Missing support is zero/false, not NaN.'),
    ]
    data = {item.name:item for item in cases}
    maximal=torch.finfo(torch.float64).max
    extreme_samples=torch.full((1,22,1,2),maximal,dtype=torch.float64)
    extreme_samples[...,1]=-maximal
    data['finite_max_convex_mean']=EvalCase(name='finite_max_convex_mean',stage='production',
        purpose='22 identical finite maximum magnitudes expose a rounded mean exceeding its convex bound.',
        inputs=dict(samples=extreme_samples,valid=torch.ones(1,22,1,dtype=torch.bool),
                    confidence=torch.ones(1,22,1,dtype=torch.float64),chunk_size=1),
        expected=(torch.tensor([[[maximal,-maximal]]],dtype=torch.float64),torch.ones(1,1,dtype=torch.bool)),
        comparison=partial(compare_values,atol=0.,rtol=1e-14))
    bad_inputs = dict(data['equal_views'].inputs)
    bad_inputs['confidence'] = torch.tensor([[[-1.],[1.]]],dtype=torch.float64)
    data['negative_confidence'] = EvalCase(name='negative_confidence',stage='production',
        purpose='Invalid eligible confidence is a contract error, not missing support.',
        inputs=bad_inputs,expected_exception=ValueError)
    return data


def _assert_case(implementation,case):
    actual = implementation(**case.inputs)
    diagnostics = case.comparison(actual,case.expected)
    assert not diagnostics, '\n'.join(diagnostics)


def verify_baseline_known_means(fuse_views):
    for name in ('equal_views','one_view'):
        _assert_case(fuse_views,make_eval_data()[name])


def verify_followup_mass_and_empty(fuse_views):
    for name in ('confidence_weights','zero_mass','empty_points','all_missing'):
        _assert_case(fuse_views,make_eval_data()[name])


def verify_followup_camera_permutations(fuse_views):
    @settings(max_examples=12,deadline=None,derandomize=True)
    @given(st.lists(st.integers(-20,20),min_size=6,max_size=6),st.lists(st.integers(0,8),min_size=3,max_size=3))
    def trial(values,weights):
        samples = torch.tensor(values,dtype=torch.float64).reshape(1,3,1,2)
        confidence = torch.tensor(weights,dtype=torch.float64).reshape(1,3,1)
        valid = torch.ones_like(confidence,dtype=torch.bool)
        expected = _scalar_expected(samples,valid,confidence)
        actual = fuse_views(samples,valid,confidence)
        assert not compare_values(actual,expected,atol=1e-10,rtol=1e-10)
        permutation = torch.tensor([2,0,1])
        assert not compare_values(fuse_views(samples[:,permutation],valid[:,permutation],confidence[:,permutation]),expected,atol=1e-10,rtol=1e-10)
    trial()


def verify_production_numerics_and_strides(fuse_views):
    for name in ('masked_nan','weight_overflow','finite_max_convex_mean'):
        _assert_case(fuse_views,make_eval_data()[name])
    samples = torch.arange(48,dtype=torch.float32).reshape(2,3,4,2).transpose(2,3)
    valid = torch.ones(samples.shape[:3],dtype=torch.bool)
    confidence = torch.full(samples.shape[:3],3e38,dtype=samples.dtype)
    expected = _scalar_expected(samples,valid,confidence)
    originals = [value.clone() for value in (samples,valid,confidence)]
    for chunk_size in (None,1,7):
        assert not compare_values(fuse_views(samples,valid,confidence,chunk_size=chunk_size),expected)
    for value,original in zip((samples,valid,confidence),originals):
        assert torch.equal(value,original)


def verify_production_smooth_gradients(fuse_views):
    samples = torch.tensor([[[[2.,4.]],[[6.,8.]]]],dtype=torch.float64,requires_grad=True)
    confidence = torch.tensor([[[1.],[3.]]],dtype=torch.float64,requires_grad=True)
    valid = torch.ones_like(confidence,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda s,w:fuse_views(s,valid,w,chunk_size=1)[0],(samples,confidence))
    # d(sum(feature))/d(samples) is normalized confidence in each channel.
    fused,_ = fuse_views(samples,valid,confidence)
    ds,dw = torch.autograd.grad(fused.sum(),(samples,confidence))
    torch.testing.assert_close(ds,torch.tensor([[[[.25,.25]],[[.75,.75]]]],dtype=ds.dtype))
    torch.testing.assert_close(dw,torch.tensor([[[-1.5],[.5]]],dtype=dw.dtype))
    for flags,weights in [(torch.zeros_like(valid),torch.full_like(confidence,float('nan'))),(valid,torch.zeros_like(confidence))]:
        s = torch.full_like(samples,float('nan'),requires_grad=True)
        w = weights.detach().requires_grad_()
        y,support = fuse_views(s,flags,w)
        assert torch.equal(y,torch.zeros_like(y)) and not support.any()
        gradients = torch.autograd.grad(y.sum(),(s,w))
        assert all(torch.isfinite(g).all() and torch.equal(g,torch.zeros_like(g)) for g in gradients)


def verify_production_feature_gradient_at_finite_max(fuse_views):
    inputs=make_eval_data()['finite_max_convex_mean'].inputs
    inputs['samples'].requires_grad_()
    output,_=fuse_views(**inputs)
    gradient=torch.autograd.grad(output.sum(),inputs['samples'])[0]
    torch.testing.assert_close(gradient,torch.full_like(gradient,1/22))


def verify_production_validation(fuse_views):
    case = make_eval_data()['equal_views']
    for key,value in [('confidence',torch.tensor([[[-1.],[1.]]],dtype=torch.float64)),
                      ('confidence',torch.tensor([[[float('inf')],[1.]]],dtype=torch.float64)),
                      ('samples',torch.full((1,2,1,2),float('nan'),dtype=torch.float64)),
                      ('valid',torch.ones(1,2,1)),('chunk_size',0),('chunk_size',True),
                      ('samples',torch.ones(1,2,1,2,dtype=torch.float16)),
                      ('confidence',torch.ones(1,2,1,dtype=torch.float32))]:
        inputs = dict(case.inputs); inputs[key] = value
        try:
            fuse_views(**inputs)
        except ValueError:
            continue
        raise AssertionError(f'invalid {key} must raise ValueError')

# %%
from torch_judge import evaluate
if not evaluate('0105_masked_view_fusion_eval.py',fuse_views):
    raise SystemExit(1)
