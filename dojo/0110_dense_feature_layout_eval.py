"""Rectangular dense maps from row-major tokens, with explicit prefix count."""
# %%
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values

# %%
def _validate_layout(tokens,grid_hw,prefix_tokens):
    if (not isinstance(tokens,torch.Tensor) or tokens.ndim != 3
            or tokens.shape[-1] < 1 or tokens.dtype not in (torch.float32,torch.float64)):
        raise ValueError('tokens must be float32/float64 [B,L,C] with C>0')
    if (not isinstance(grid_hw,(tuple,list)) or len(grid_hw) != 2
            or any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in grid_hw)):
        raise ValueError('grid_hw must be positive integer (height,width)')
    if isinstance(prefix_tokens,bool) or not isinstance(prefix_tokens,int) or prefix_tokens < 0:
        raise ValueError('prefix_tokens must be a nonnegative integer')
    h,w = grid_hw
    if tokens.shape[1] != prefix_tokens+h*w:
        raise ValueError('token count must equal prefix_tokens + height*width')
    return h,w

# %%
# Reference 1: row-by-row copy exposes which token lands at each pixel.
# Production qualified, O(B*C*H*W) output allocation and Python loop overhead.
# Scalars remain tensor views during assignment, preserving gradients.
def dense_loop(tokens,grid_hw,prefix_tokens=0):
    h,w = _validate_layout(tokens,grid_hw,prefix_tokens)
    b,_,c = tokens.shape
    if b == 0:
        return tokens[:,prefix_tokens:].reshape(b,h,w,c).permute(0,3,1,2)
    output = tokens.new_empty((b,c,h,w))
    for row in range(h):
        for col in range(w):
            output[:,:,row,col] = tokens[:,prefix_tokens+row*w+col,:]
    return output

# %%
# Reference 2: express row-major ordering with reshape then permute.
# Production qualified. Usually a view for contiguous BLC tokens, including
# prefix slices; reshape may copy for other strides. Output may be noncontiguous
# and alias tokens: callers must decide whether downstream mutation is acceptable.
def dense_view(tokens,grid_hw,prefix_tokens=0):
    h,w = _validate_layout(tokens,grid_hw,prefix_tokens)
    b,_,c = tokens.shape
    spatial = tokens[:,prefix_tokens:,:]
    return spatial.reshape(b,h,w,c).permute(0,3,1,2)

# %%
# Reference 3: guarantee contiguous output storage for a downstream operator.
# Production qualified; .contiguous() copies only if needed. This storage choice
# changes neither the values nor their gradients. No blanket speed advantage.
def dense_contiguous(tokens,grid_hw,prefix_tokens=0):
    h,w = _validate_layout(tokens,grid_hw,prefix_tokens)
    b,_,c = tokens.shape
    return tokens[:,prefix_tokens:,:].reshape(b,h,w,c).permute(0,3,1,2).contiguous()

# %%
def inspect_layout_storage():
    tokens = torch.arange(42,dtype=torch.float64).reshape(1,7,6)
    reports = {}
    for name,fn in [('loop',dense_loop),('view',dense_view),('contiguous',dense_contiguous)]:
        output = fn(tokens,(2,3),1)
        reports[name] = {'stride':output.stride(),'contiguous':output.is_contiguous(),
            'aliases_input_storage':output.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr(),
            'logical_output_bytes':output.numel()*output.element_size(),
            'note':'Storage alias/stride observations; not allocator peak measurements.'}
    return reports

# %%
def benchmark_layout(shape=(2,257,32),grid_hw=(16,16),prefix_tokens=1,device='cpu'):
    from torch_judge.measurement import measure
    tokens = torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32,device=device).reshape(shape)
    reports = {}
    expected = dense_view(tokens,grid_hw,prefix_tokens)
    for name,fn in [('loop',dense_loop),('view',dense_view),('contiguous',dense_contiguous)]:
        actual = fn(tokens,grid_hw,prefix_tokens)
        reports[name] = {'max_error':(actual-expected).abs().max().item(),
            'timing':measure(lambda:fn(tokens,grid_hw,prefix_tokens),device=device,
                workload={'shape':shape,'grid_hw':grid_hw,'prefix_tokens':prefix_tokens,'dtype':str(tokens.dtype)})}
    return reports

# %%
dense_features = dense_view

# %% [verifiers]
def _numbered_expected(tokens,grid_hw,prefix_tokens=0):
    # Scalar indexing oracle; no reshape/permute used to compute expectations.
    height,width = grid_hw
    b,_,c = tokens.shape
    result = torch.empty((b,c,height,width),dtype=tokens.dtype,device=tokens.device)
    for batch in range(b):
        for channel in range(c):
            for row in range(height):
                for col in range(width):
                    result[batch,channel,row,col] = float(tokens[batch,prefix_tokens+row*width+col,channel])
    return result


def make_eval_data():
    specifications = [('square_numbered','baseline',1,2,2,2,0),
        ('rectangular_prefix','followup',2,2,3,3,2),
        ('empty_batch','followup',0,2,3,2,1),
        ('noncontiguous','production',2,3,2,2,1)]
    cases = {}
    for name,stage,b,h,w,c,p in specifications:
        if name == 'noncontiguous':
            tokens = torch.arange(b*c*(p+h*w),dtype=torch.float64).reshape(b,c,p+h*w).transpose(1,2)
        else:
            tokens = torch.arange(b*(p+h*w)*c,dtype=torch.float64).reshape(b,p+h*w,c)
        cases[name] = EvalCase(name=name,stage=stage,purpose=f'Explicit {h}x{w} spatial grid; skip exactly {p} prefix tokens.',
            inputs=dict(tokens=tokens,grid_hw=(h,w),prefix_tokens=p),
            expected=_numbered_expected(tokens,(h,w),p),comparison=partial(compare_values,atol=0,rtol=0))
    invalid = dict(cases['rectangular_prefix'].inputs); invalid['grid_hw'] = (3,3)
    cases['invalid_grid'] = EvalCase(name='invalid_grid',stage='production',purpose='Token count disagrees with declared rectangular grid.',inputs=invalid,expected_exception=ValueError)
    return cases


def verify_baseline_numbered_layout(dense_features):
    case = make_eval_data()['square_numbered']
    errors = case.comparison(dense_features(**case.inputs),case.expected)
    assert not errors,'\n'.join(errors)


def verify_followup_rectangle_prefix_and_empty(dense_features):
    for name in ('rectangular_prefix','empty_batch'):
        case = make_eval_data()[name]
        assert not case.comparison(dense_features(**case.inputs),case.expected)


def verify_followup_generated_layouts(dense_features):
    @settings(max_examples=12,deadline=None,derandomize=True)
    @given(st.integers(1,3),st.integers(1,4),st.integers(0,2),st.integers(1,3))
    def trial(h,w,p,c):
        tokens = torch.arange(2*(p+h*w)*c,dtype=torch.float64).reshape(2,p+h*w,c)
        assert not compare_values(dense_features(tokens,(h,w),p),_numbered_expected(tokens,(h,w),p),atol=0,rtol=0)
    trial()


def verify_production_strides_gradients_and_immutability(dense_features):
    case = make_eval_data()['noncontiguous']
    inputs = case.inputs; original = inputs['tokens'].clone()
    inputs['tokens'].requires_grad_()
    actual = dense_features(**inputs)
    assert not case.comparison(actual,case.expected)
    weights = torch.arange(actual.numel(),dtype=actual.dtype).reshape(actual.shape)
    gradient = torch.autograd.grad((actual*weights).sum(),inputs['tokens'])[0]
    expected_gradient = torch.zeros_like(inputs['tokens'])
    for batch in range(actual.shape[0]):
        for channel in range(actual.shape[1]):
            for row in range(actual.shape[2]):
                for col in range(actual.shape[3]):
                    expected_gradient[batch,inputs['prefix_tokens']+row*actual.shape[3]+col,channel] = weights[batch,channel,row,col]
    assert torch.equal(gradient,expected_gradient)
    assert torch.equal(inputs['tokens'].detach(),original)
    assert torch.autograd.gradcheck(lambda tokens:dense_features(tokens,(3,2),1),(inputs['tokens'],))
    for dtype in (torch.float32,torch.float64):
        tokens = torch.arange(12,dtype=dtype).reshape(1,6,2)
        actual = dense_features(tokens,(2,3))
        assert actual.dtype == dtype and actual.device == tokens.device
    empty = torch.empty(0,7,2,dtype=torch.float64,requires_grad=True)
    torch.autograd.grad(dense_features(empty,(2,3),1).sum(),empty)


def verify_production_metadata(dense_features):
    inputs = make_eval_data()['rectangular_prefix'].inputs
    for key,value in [('grid_hw',(0,6)),('grid_hw',(3,3)),('grid_hw',(True,6)),
        ('grid_hw',(2.0,3)),('prefix_tokens',-1),('prefix_tokens',True),
        ('prefix_tokens',1.5),('tokens',torch.zeros(2,8,3,dtype=torch.int64)),
        ('tokens',torch.zeros(2,8,0))]:
        try:
            dense_features(**dict(inputs,**{key:value}))
        except ValueError:
            continue
        raise AssertionError(f'invalid {key} must raise ValueError')

# %%
from torch_judge import evaluate
if not evaluate('0110_dense_feature_layout_eval.py',dense_features):
    raise SystemExit(1)
