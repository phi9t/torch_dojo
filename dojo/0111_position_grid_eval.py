"""Rectangular positional embeddings with explicit half-pixel interpolation."""
# %%
import math
from functools import partial
import torch
import torch.nn.functional as F
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values, evaluate
from torch_judge.measurement import measure

# %%
def _validate_positions(positions,old_hw,new_hw,prefix_tokens):
    if not isinstance(positions,torch.Tensor) or positions.ndim!=3 or positions.shape[-1]<1:
        raise ValueError('positions must be [B,L,C], C positive')
    if positions.dtype not in (torch.float32,torch.float64) or not torch.isfinite(positions).all():
        raise ValueError('finite float32/float64 positions required')
    for name,hw in [('old_hw',old_hw),('new_hw',new_hw)]:
        if not isinstance(hw,(tuple,list)) or len(hw)!=2 or any(not isinstance(v,int) or isinstance(v,bool) or v<=0 for v in hw):
            raise ValueError(name+' must be a pair of positive integers')
    if not isinstance(prefix_tokens,int) or isinstance(prefix_tokens,bool) or prefix_tokens<0:
        raise ValueError('prefix_tokens must be a nonnegative integer')
    H,W=old_hw
    if positions.shape[1]!=prefix_tokens+H*W: raise ValueError('token count does not match prefix+H*W')
    return positions.shape[0],positions.shape[2]

# %%
# Reference 1: four neighbors written explicitly for each target pixel.
# Qualifies production on finite float32/64; output stays in source dtype.
# Half-pixel coordinates, clamp-to-edge, no antialias. O(B*h*w*C) output
# plus Python-cell work. Prefix copied exactly; same grid returns source alias.
def resize_explicit(positions,old_hw,new_hw,prefix_tokens=0):
    B,C=_validate_positions(positions,old_hw,new_hw,prefix_tokens)
    H,W=old_hw;h,w=new_hw
    if (H,W)==(h,w): return positions
    grid=positions[:,prefix_tokens:].reshape(B,H,W,C)
    cells=[]
    for y in range(h):
        sy=min(max((y+.5)*H/h-.5,0),H-1); y0=math.floor(sy); y1=min(y0+1,H-1); dy=sy-y0
        for x in range(w):
            sx=min(max((x+.5)*W/w-.5,0),W-1); x0=math.floor(sx); x1=min(x0+1,W-1); dx=sx-x0
            top=grid[:,y0,x0]*(1-dx)+grid[:,y0,x1]*dx
            bottom=grid[:,y1,x0]*(1-dx)+grid[:,y1,x1]*dx
            cells.append(top*(1-dy)+bottom*dy)
    resized=torch.stack(cells,dim=1)
    return torch.cat((positions[:,:prefix_tokens],resized),1)

# %%
# Reference 2: separable interpolation avoids a Python loop per destination.
# Compute coordinates in float64, cast weights to source dtype. Intermediate
# [B,H,new_W,C], then final [B,new_H,new_W,C]. Qualified production; this
# trades a horizontal intermediate for vectorized dispatch, not sparse memory.
def resize_separable(positions,old_hw,new_hw,prefix_tokens=0):
    B,C=_validate_positions(positions,old_hw,new_hw,prefix_tokens)
    H,W=old_hw;h,w=new_hw
    if (H,W)==(h,w): return positions
    grid=positions[:,prefix_tokens:].reshape(B,H,W,C)
    x=((torch.arange(w,device=positions.device,dtype=torch.float64)+.5)*W/w-.5).clamp(0,W-1)
    x0=x.floor().long();x1=(x0+1).clamp(max=W-1);dx=(x-x0).to(positions.dtype)[None,None,:,None]
    horizontal=grid[:,:,x0,:]*(1-dx)+grid[:,:,x1,:]*dx
    y=((torch.arange(h,device=positions.device,dtype=torch.float64)+.5)*H/h-.5).clamp(0,H-1)
    y0=y.floor().long();y1=(y0+1).clamp(max=H-1);dy=(y-y0).to(positions.dtype)[None,:,None,None]
    resized=horizontal[:,y0,:,:]*(1-dy)+horizontal[:,y1,:,:]*dy
    return torch.cat((positions[:,:prefix_tokens],resized.reshape(B,h*w,C)),1)

# %%
# Reference 3: framework interpolation after an explicit layout conversion.
# Same semantics, align_corners=False, antialias=False, source/output dtype.
# O(B*h*w*C) output; framework workspace depends on backend. Qualifies
# production with floating tolerance, not bit parity with a different backend.
# No separate 'stability' version is invented: this track demonstrates layout
# and dispatch improvements rather than claiming a numerical gain.
def resize_framework(positions,old_hw,new_hw,prefix_tokens=0):
    B,C=_validate_positions(positions,old_hw,new_hw,prefix_tokens)
    H,W=old_hw;h,w=new_hw
    if (H,W)==(h,w): return positions
    grid=positions[:,prefix_tokens:].reshape(B,H,W,C).permute(0,3,1,2)
    resized=F.interpolate(grid,size=(h,w),mode='bilinear',align_corners=False,antialias=False)
    tokens=resized.permute(0,2,3,1).reshape(B,h*w,C)
    return torch.cat((positions[:,:prefix_tokens],tokens),1)

# %%
resize_positions=resize_separable

# %%
def benchmark_positions(old_hw=(8,12),new_hw=(12,20),C=16,device='cpu',dtype=torch.float32):
    """Opt-in seeded timings including layout/validation, no speed gate."""
    generator=torch.Generator(device=device).manual_seed(111)
    inputs=dict(positions=torch.randn(2,1+old_hw[0]*old_hw[1],C,device=device,dtype=dtype,generator=generator),old_hw=old_hw,new_hw=new_hw,prefix_tokens=1)
    expected=resize_explicit(**inputs);reports={}
    for name,fn in [('explicit',resize_explicit),('separable',resize_separable),('framework',resize_framework)]:
        actual=fn(**inputs);errors=compare_values(actual,expected,atol=2e-5,rtol=2e-5)
        assert not errors,errors
        reports[name]={'timing':measure(lambda:fn(**inputs),device=device,
                       workload={'B':2,'old_hw':old_hw,'new_hw':new_hw,'C':C,'prefix':1,'seed':111,'dtype':str(dtype)}),
                       'max_absolute_error':(actual-expected).abs().max().item(),
                       'storage_scope':'outputs and forward interpolation intermediates; no measured allocator peak'}
    return reports

# %% [verifiers]
def _scalar_positions(inputs):
    # Literal half-pixel coordinates with scalar interpolation, not a reference.
    p=inputs['positions'].detach(); B,L,C=p.shape
    H,W=inputs['old_hw']; h,w=inputs['new_hw']; prefix=inputs.get('prefix_tokens',0)
    if (H,W)==(h,w): return p.clone()
    out=p.new_empty(B,prefix+h*w,C)
    out[:,:prefix]=p[:,:prefix]
    for b in range(B):
        for y in range(h):
            sy=min(max((y+.5)*H/h-.5,0),H-1); y0=math.floor(sy); y1=min(y0+1,H-1); dy=sy-y0
            for x in range(w):
                sx=min(max((x+.5)*W/w-.5,0),W-1); x0=math.floor(sx); x1=min(x0+1,W-1); dx=sx-x0
                for c in range(C):
                    row0=(1-dx)*float(p[b,prefix+y0*W+x0,c])+dx*float(p[b,prefix+y0*W+x1,c])
                    row1=(1-dx)*float(p[b,prefix+y1*W+x0,c])+dx*float(p[b,prefix+y1*W+x1,c])
                    out[b,prefix+y*w+x,c]=(1-dy)*row0+dy*row1
    return out


def make_eval_data():
    examples=[]
    ramp=torch.tensor([[[0.],[2.]]],dtype=torch.float64)
    examples.append(('one_by_two_ramp','baseline',dict(positions=ramp,old_hw=(1,2),new_hw=(1,4))))
    rectangle=torch.arange(2*7*3,dtype=torch.float64).reshape(2,7,3)
    examples.append(('prefix_rectangular_upsample','followup',dict(positions=rectangle,old_hw=(2,3),new_hw=(3,5),prefix_tokens=1)))
    examples.append(('rectangular_downsample','followup',dict(positions=torch.arange(24.,dtype=torch.float64).reshape(1,12,2),old_hw=(3,4),new_hw=(2,2))))
    examples.append(('singleton_source','followup',dict(positions=torch.tensor([[[5.,-2.]]]),old_hw=(1,1),new_hw=(3,4))))
    examples.append(('exact_identity','production',dict(positions=rectangle,old_hw=(2,3),new_hw=(2,3),prefix_tokens=1)))
    examples.append(('strided_tokens','production',dict(positions=torch.arange(48.,dtype=torch.float64).reshape(2,6,4)[...,::2],old_hw=(2,3),new_hw=(4,2))))
    examples.append(('empty_batch','production',dict(positions=torch.empty(0,7,3,dtype=torch.float64),old_hw=(2,3),new_hw=(3,2),prefix_tokens=1)))
    return {name:EvalCase(name=name,stage=stage,purpose=name.replace('_',' '),inputs=inputs,
                        expected=_scalar_positions(inputs),comparison=partial(compare_values,atol=1e-6,rtol=1e-5))
            for name,stage,inputs in examples}


def verify_baseline_half_pixel_ramp(resize_positions):
    case=make_eval_data()['one_by_two_ramp']
    expected=torch.tensor([[[0.],[.5],[1.5],[2.]]],dtype=torch.float64)
    assert not compare_values(resize_positions(**case.inputs),expected,atol=1e-12,rtol=1e-12)


def verify_followup_rectangles_prefix_and_singletons(resize_positions):
    for name in ('prefix_rectangular_upsample','rectangular_downsample','singleton_source'):
        case=make_eval_data()[name]; actual=resize_positions(**case.inputs)
        assert not compare_values(actual,case.expected,atol=1e-6,rtol=1e-5)
        prefix=case.inputs.get('prefix_tokens',0)
        assert torch.equal(actual[:,:prefix],case.inputs['positions'][:,:prefix])


def verify_production_cases_identity_and_no_mutation(resize_positions):
    for case in make_eval_data().values():
        before=case.inputs['positions'].clone(); actual=resize_positions(**case.inputs)
        assert not compare_values(actual,case.expected,atol=1e-6,rtol=1e-5)
        assert torch.equal(before,case.inputs['positions'])
        if case.inputs['old_hw']==case.inputs['new_hw']: assert torch.equal(actual,before)


def verify_production_validation(resize_positions):
    inputs=make_eval_data()['one_by_two_ramp'].inputs
    invalid=[dict(inputs,old_hw=(0,2)),dict(inputs,new_hw=(2,)),dict(inputs,new_hw=(2,2.5)),
             dict(inputs,old_hw=(True,2)),dict(inputs,prefix_tokens=-1),dict(inputs,prefix_tokens=True),
             dict(inputs,prefix_tokens=1),dict(inputs,positions=torch.zeros(1,2,1,dtype=torch.int64)),
             dict(inputs,positions=torch.full((1,2,1),float('nan'))),dict(inputs,positions=torch.empty(1,2,0))]
    for case in invalid:
        try: resize_positions(**case)
        except ValueError: pass
        else: raise AssertionError('malformed position/grid metadata must raise ValueError')


def verify_production_gradient_and_framework_parity(resize_positions):
    case=make_eval_data()['prefix_rectangular_upsample'].inputs
    p=case['positions'].clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x:resize_positions(**dict(case,positions=x)),(p,))
    actual=resize_positions(**dict(case,positions=p))
    source=p[:,1:].reshape(2,2,3,3).permute(0,3,1,2)
    ref=torch.cat((p[:,:1],F.interpolate(source,size=(3,5),mode='bilinear',align_corners=False,antialias=False).permute(0,2,3,1).reshape(2,15,3)),1)
    torch.testing.assert_close(actual,ref,atol=1e-12,rtol=1e-12)
    weights=torch.arange(actual.numel(),dtype=p.dtype).reshape_as(actual)
    grad=torch.autograd.grad((actual*weights).sum(),p,retain_graph=True)[0]
    expected=torch.autograd.grad((ref*weights).sum(),p)[0]
    torch.testing.assert_close(grad,expected,atol=1e-10,rtol=1e-10)


def verify_production_generated_constants(resize_positions):
    @settings(max_examples=15,deadline=None,derandomize=True)
    @given(st.integers(1,4),st.integers(1,5),st.integers(1,4),st.integers(1,5),st.integers(0,2))
    def trial(H,W,h,w,prefix):
        p=torch.full((2,prefix+H*W,2),3.25,dtype=torch.float64)
        p[:,:prefix]=9
        inputs=dict(positions=p,old_hw=(H,W),new_hw=(h,w),prefix_tokens=prefix)
        actual=resize_positions(**inputs)
        assert not compare_values(actual,_scalar_positions(inputs),atol=1e-12,rtol=1e-12)
    trial()

# %%
if not evaluate('0111_position_grid_eval.py',resize_positions): raise SystemExit(1)
