"""Sparse voxel/pillar pooling: complete reference progressions."""
# %%
import math
from decimal import Decimal, localcontext, ROUND_FLOOR
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values, evaluate
from torch_judge.measurement import measure

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
# Reference 1: explicit occupied-cell loop. All versions qualify production
# with common CPU float32/64 inputs; float32 mean accumulates in float64 and
# returns original dtype. Normalize each channel by its group max magnitude;
# sum bounded contributions, then constrain the normalized mean to [-1,1]
# before restoring scale. This convex bound prevents rounding beyond finfo.max.
# At that rounding correction the derivative follows clamp/max, not an ideal
# real-arithmetic mean. Ordinary smooth cases retain 1/count derivatives.
# Group search is O(P*V), storage O(P*C+V*C), never full-grid allocation.
def pool_group_loop(points,features,point_valid,origin,voxel_size,grid_shape,pillar=False):
    rows,locations,selected=_included_points(points,features,point_valid,origin,voxel_size,grid_shape,pillar)
    keys=sorted(set(tuple(row) for row in rows.tolist()))
    inverse=torch.full(points.shape[:2],-1,dtype=torch.int64,device=points.device)
    counts=[];means=[]
    work=selected.double() if selected.dtype==torch.float32 else selected
    for i,key in enumerate(keys):
        match=(rows==rows.new_tensor(key)).all(1)
        group=work[match]
        counts.append(len(group))
        scale=group.abs().amax(0)
        safe=torch.where(scale>0,scale,torch.ones_like(scale))
        unit=(group/safe).sum(0)/len(group)
        means.append(unit.clamp(-1,1)*safe)
        loc=locations[match]; inverse[loc[:,0],loc[:,1]]=i
    indices=torch.tensor(keys,dtype=torch.int64,device=points.device).reshape(-1,4)
    counts=torch.tensor(counts,dtype=torch.int64,device=points.device)
    means=torch.stack(means).to(features.dtype) if means else selected[:0]
    return indices,counts,means,inverse

# %%
# Reference 2: unique rows plus differentiable scatter. O(P log P) grouping,
# O(P*C+V*C) work storage. Repeated points accumulate, including exact zeros.
# Avoids packed-key overflow; indices are lexicographic [batch,z,y,x].
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
# Reference 3: stable lexicographic sort then contiguous segment reductions.
# Four stable integer sorts avoid a packed linear key. Segmentation keeps
# reductions local; GPU reduction rounding can differ from scatter order.
# Storage O(P*C+V*C); discrete assignment has no coordinate gradient promise.
def pool_sorted_segments(points,features,point_valid,origin,voxel_size,grid_shape,pillar=False):
    rows,locations,selected=_included_points(points,features,point_valid,origin,voxel_size,grid_shape,pillar)
    order=torch.arange(len(rows),device=points.device)
    for column in (3,2,1,0): order=order[torch.argsort(rows[order,column],stable=True)]
    ordered=rows[order]
    indices,ids,counts=torch.unique_consecutive(ordered,dim=0,return_inverse=True,return_counts=True)
    inverse=torch.full(points.shape[:2],-1,dtype=torch.int64,device=points.device)
    loc=locations[order]; inverse[loc[:,0],loc[:,1]]=ids
    work=selected[order].double() if selected.dtype==torch.float32 else selected[order]
    if len(indices):
        scale=torch.segment_reduce(work.abs(),'max',lengths=counts)
        safe=torch.where(scale>0,scale,torch.ones_like(scale))
        sums=torch.segment_reduce(work/safe[ids],'sum',lengths=counts)
        means=((sums/counts[:,None]).clamp(-1,1)*safe).to(features.dtype)
    else: means=selected[:0]
    return indices,counts,means,inverse

# %%
pool_points=pool_unique_scatter

# %%
def benchmark_pooling(N=2048,C=8,device='cpu',dtype=torch.float32):
    """Opt-in seeded sparse workload; timings include validation/grouping."""
    generator=torch.Generator(device=device).manual_seed(107)
    inputs=dict(points=torch.rand(1,N,3,device=device,dtype=dtype,generator=generator)*4,
                features=torch.randn(1,N,C,device=device,dtype=dtype,generator=generator),
                point_valid=torch.ones(1,N,device=device,dtype=torch.bool),
                origin=torch.zeros(3,device=device,dtype=dtype),voxel_size=torch.ones(3,device=device,dtype=dtype),
                grid_shape=torch.tensor([4,4,4],device=device))
    expected=pool_group_loop(**inputs);reports={}
    for name,fn in [('loop',pool_group_loop),('scatter',pool_unique_scatter),('segments',pool_sorted_segments)]:
        actual=fn(**inputs); errors=compare_values(actual,expected,atol=1e-6,rtol=1e-5)
        assert not errors,errors
        reports[name]={'timing':measure(lambda:fn(**inputs),device=device,
                       workload={'N':N,'C':C,'seed':107,'dtype':str(dtype),'occupied':len(actual[0])}),
                       'max_mean_error':(actual[2]-expected[2]).abs().max().item() if len(actual[0]) else 0,
                       'storage':'O(points*channels + occupied*channels), excludes retained backward state'}
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


def _pool_scene(dtype=torch.float64):
    return dict(points=torch.tensor([[[.1,.1,.1],[.2,.2,.2],[.1,.1,1.1],[2.,0.,0.],[-.1,0.,0.]]],dtype=dtype),
                features=torch.tensor([[[2.,0.],[4.,0.],[8.,2.],[99.,99.],[99.,99.]]],dtype=dtype),
                point_valid=torch.ones(1,5,dtype=torch.bool),origin=torch.zeros(3,dtype=dtype),
                voxel_size=torch.ones(3,dtype=dtype),grid_shape=torch.tensor([2,2,2]))


def _scalar_mean(rows,channel):
    values=[row[channel] for row in rows]
    scale=max(abs(value) for value in values)
    return math.fsum(value/scale for value in values)/len(values)*scale if scale else 0.


def _scalar_pool(inputs):
    # Python dictionary groups and scalar math establish ordering independently.
    points,features=inputs['points'].detach(),inputs['features'].detach()
    B,N,C=features.shape; groups={}; members={}
    origin=inputs['origin'].tolist(); size=inputs['voxel_size'].tolist(); shape=inputs['grid_shape'].tolist()
    for b in range(B):
        for n in range(N):
            if not inputs['point_valid'][b,n]: continue
            xyz=points[b,n].tolist()
            if not all(origin[a] <= xyz[a] < origin[a]+size[a]*shape[a] for a in range(3)): continue
            cells=[_scalar_cell(xyz[a],origin[a],size[a]) for a in range(3)]
            key=(b,0 if inputs.get('pillar',False) else cells[2],cells[1],cells[0])
            groups.setdefault(key,[]).append(features[b,n].tolist()); members[(b,n)]=key
    keys=sorted(groups); key_ids={key:i for i,key in enumerate(keys)}
    indices=torch.tensor(keys,dtype=torch.int64,device=points.device).reshape(-1,4)
    counts=torch.tensor([len(groups[key]) for key in keys],dtype=torch.int64,device=points.device)
    means=torch.tensor([[_scalar_mean(groups[key],c) for c in range(C)] for key in keys],dtype=features.dtype,device=features.device).reshape(-1,C)
    inverse=torch.full((B,N),-1,dtype=torch.int64,device=points.device)
    for loc,key in members.items(): inverse[loc]=key_ids[key]
    return indices,counts,means,inverse


def make_eval_data():
    examples=[('two_voxels_and_boundaries','baseline',_pool_scene())]
    pillar=_pool_scene(); pillar['pillar']=True
    examples.append(('pillar_collapses_z','followup',pillar))
    batches=_pool_scene()
    for key in ('points','features','point_valid'): batches[key]=batches[key].repeat(2,*([1]*(batches[key].ndim-1)))
    batches['features'][1,:3]*=2
    examples.append(('separate_batches','followup',batches))
    zeros=_pool_scene(); zeros['features'].zero_()
    examples.append(('occupied_zero_features','followup',zeros))
    asymmetric=_pool_scene()
    asymmetric['points']=torch.tensor([[[3.5,.2,.2],[.2,3.,.2],[.2,.2,3.5],[.2,.2,.2],[.3,.3,.3]]],dtype=torch.float64)
    asymmetric['voxel_size']=torch.tensor([2.,1.,2.],dtype=torch.float64)
    asymmetric['grid_shape']=torch.tensor([2,4,2])
    asymmetric['origin']=torch.tensor([-2.,1.,-3.],dtype=torch.float64)
    asymmetric['points']+=asymmetric['origin']
    examples.append(('anisotropic_shifted_sorted_cells','followup',asymmetric))
    excluded=_pool_scene(); excluded['point_valid'].fill_(False)
    examples.append(('all_excluded','production',excluded))
    empty=_pool_scene()
    for key in ('points','features','point_valid'): empty[key]=empty[key][:,:0]
    examples.append(('empty_points','production',empty))
    masked=_pool_scene(); masked['points'][0,1]=float('nan'); masked['features'][0,1]=float('nan'); masked['point_valid'][0,1]=False
    examples.append(('masked_nan','production',masked))
    wide=_pool_scene(torch.float32); wide['features'][0,:3,0]=3e38
    examples.append(('wide_sum','production',wide))
    huge=_pool_scene(); huge['features'][0,:3,0]=1e308
    examples.append(('finite_float64_mean','production',huge))
    maximal=_pool_scene()
    maximal['points']=torch.full((1,17,3),.1,dtype=torch.float64)
    maximal['features']=torch.full((1,17,2),torch.finfo(torch.float64).max,dtype=torch.float64)
    maximal['point_valid']=torch.ones(1,17,dtype=torch.bool)
    examples.append(('finite_max_convex_mean','production',maximal))
    strides=_pool_scene(); storage=torch.zeros(1,5,4,dtype=torch.float64); storage[...,::2]=strides['features']; strides['features']=storage[...,::2]
    examples.append(('strided_features','production',strides))
    for dtype in (torch.float32,torch.float64):
        tiny=torch.finfo(dtype).tiny
        inputs=dict(points=torch.tensor([[[-tiny,-.5,-.5],[-.5,-.5,-.5],[0.,-.5,-.5],[tiny,-.5,-.5]]],dtype=dtype),features=torch.tensor([[[2.],[4.],[8.],[16.]]],dtype=dtype),point_valid=torch.ones(1,4,dtype=torch.bool),origin=-torch.ones(3,dtype=dtype),voxel_size=torch.ones(3,dtype=dtype),grid_shape=torch.ones(3,dtype=torch.int64))
        examples.append(('near_zero_upper_'+str(dtype).split('.')[-1],'followup',inputs))
    result={name:EvalCase(name=name,stage=stage,purpose=name.replace('_',' '),inputs=inputs,
                        expected=_scalar_pool(inputs),comparison=partial(compare_values,atol=1e-6,rtol=1e-6))
            for name,stage,inputs in examples}
    for dtype in (torch.float32,torch.float64):
        case=result['near_zero_upper_'+str(dtype).split('.')[-1]]
        # Literal one voxel, two included rows; exact/positive zero edge outside.
        literal=(torch.tensor([[0,0,0,0]]),torch.tensor([2]),torch.tensor([[3.]],dtype=dtype),torch.tensor([[0,0,-1,-1]]))
        result[case.name]=EvalCase(name=case.name,stage=case.stage,purpose='Negative tiny x is inside the final voxel; zero and positive tiny x are outside.',inputs=case.inputs,expected=literal)
    return result


def verify_followup_near_zero_upper_literal(pool_points):
    for dtype in (torch.float32,torch.float64):
        case=make_eval_data()['near_zero_upper_'+str(dtype).split('.')[-1]]
        assert not compare_values(pool_points(**case.inputs),case.expected),case.name


def verify_baseline_hand_groups(pool_points):
    actual=pool_points(**_pool_scene())
    expected=(torch.tensor([[0,0,0,0],[0,1,0,0]]),torch.tensor([2,1]),
              torch.tensor([[3.,0.],[8.,2.]],dtype=torch.float64),torch.tensor([[0,0,1,-1,-1]]))
    assert not compare_values(actual,expected)


def verify_followup_pillars_batches_and_zeros(pool_points):
    for name in ('pillar_collapses_z','separate_batches','occupied_zero_features','anisotropic_shifted_sorted_cells'):
        case=make_eval_data()[name]; assert not compare_values(pool_points(**case.inputs),case.expected)


def verify_production_cases(pool_points):
    for case in make_eval_data().values():
        assert not compare_values(pool_points(**case.inputs),case.expected,atol=1e-6,rtol=1e-6)


def verify_production_validation(pool_points):
    examples=[]
    for key,index,value in [('voxel_size',0,0.),('origin',0,float('nan')),('grid_shape',0,0),('features',(0,0,0),float('nan'))]:
        inputs=_pool_scene(); inputs[key][index]=value; examples.append(inputs)
    for key in ('grid_shape','point_valid'):
        inputs=_pool_scene(); inputs[key]=inputs[key].float(); examples.append(inputs)
    inputs=_pool_scene(); inputs['features']=inputs['features'].float(); examples.append(inputs)
    inputs=_pool_scene(); inputs['pillar']=1; examples.append(inputs)
    for inputs in examples:
        try: pool_points(**inputs)
        except ValueError: pass
        else: raise AssertionError('invalid metadata must raise ValueError')


def verify_production_gradients_no_mutation(pool_points):
    inputs=_pool_scene(); inputs['features'].requires_grad_()
    originals={k:v.detach().clone() for k,v in inputs.items()}
    assert torch.autograd.gradcheck(lambda f:pool_points(**dict(inputs,features=f))[2],(inputs['features'],))
    _,_,means,_=pool_points(**inputs); means.sum().backward()
    expected=torch.tensor([[[.5,.5],[.5,.5],[1.,1.],[0.,0.],[0.,0.]]],dtype=torch.float64)
    torch.testing.assert_close(inputs['features'].grad,expected)
    for key,value in inputs.items(): assert torch.equal(value.detach(),originals[key])
    inputs=make_eval_data()['masked_nan'].inputs; inputs['features'].requires_grad_()
    pool_points(**inputs)[2].sum().backward()
    assert torch.isfinite(inputs['features'].grad).all()
    extreme=make_eval_data()['finite_max_convex_mean'].inputs
    extreme['features'].requires_grad_()
    pool_points(**extreme)[2].sum().backward()
    torch.testing.assert_close(extreme['features'].grad,torch.full_like(extreme['features'],1/17))


def verify_production_permutation_and_conservation(pool_points):
    @settings(max_examples=12,deadline=None,derandomize=True)
    @given(st.permutations((0,1,2,3,4)))
    def trial(order):
        inputs=_pool_scene(); perm=torch.tensor(order)
        before=pool_points(**inputs)
        for key in ('points','features','point_valid'): inputs[key]=inputs[key][:,perm]
        after=pool_points(**inputs)
        assert not compare_values(after[:3],before[:3])
        assert torch.equal(after[3],before[3][:,perm])
        assert int(after[1].sum())==int((after[3]>=0).sum())==3
    trial()

# %%
if not evaluate('0107_voxel_pillar_pool_eval.py',pool_points): raise SystemExit(1)
