"""Collision-safe BEV: weighted cells, count conservation, and live gradients."""
# %%
import math
import sys
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values, evaluate

# %%
# Validate before allocating dense outputs or accumulating integer counts.
# Python integer cell totals avoid wraparound; only per-pillar totals must fit
# int64 (the sum across distinct pillars may be larger). No arbitrary memory
# budget is imposed: callers remain responsible for feasible dense workloads.
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
# Version 1: teach numerator/denominator accumulation explicitly. Highest
# qualified stage: followup, float32/64 finite values whose count products and
# intermediate float64 sums remain finite. Production huge-float64 case lies
# outside this domain and deliberately demonstrates overflow. O(K*C) work,
# required dense output O(B*C*Y*X), dense float64 numerator temporary.
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
# Version 2: flatten destinations and normalize each count BEFORE multiplying
# features. Qualified production float32/64. Float64 working reductions avoid
# float32 overflow and normalized convex weights avoid huge raw products even
# in float64. Per-cell scaling and detached convex-range rounding corrections
# keep finite extreme means finite; gradients retain the linear weighted mean.
# Central differences at float64 maximum are not a meaningful gradient oracle. Floating sums have tolerance-based parity; GPU index_add order
# may vary. Exact integer total validation intentionally synchronizes metadata.
# O(K*C) work, O(K*C) source plus dense float64 scale and sum buffers. The
# mandatory dense output cannot be described as sparse-memory processing.
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
# Version 3: sort/group before reducing, then write each occupied pillar once.
# Qualified production float32/64. Sorting adds O(K log K), but avoids a dense
# float64 accumulation buffer: working floating state O(K*C+occupied*C), plus
# required dense output O(B*C*Y*X). No universal speed win; dense output and
# autograd saved sorting/reduction state still have costs. Same normalized
# weights and exact count contract as version 2, implemented independently.
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
# Opt in: the numerator loop overflows, the two normalized versions stay
# finite. This is a declared limitation demonstration, not a parity domain.
def explore_weight_product_overflow():
    case=make_eval_data()['huge_float64']
    return {'expected':case.expected[0],
            'explicit':scatter_bev_explicit(**case.inputs)[0],
            'index_add':scatter_bev_index_add(**case.inputs)[0],
            'grouped':scatter_bev_grouped(**case.inputs)[0]}

# %%
# Opt-in workload timing; exact-count preflight remains part of every call.
# CPU wall clock or synchronized CUDA timing; report numerical error and
# variability, never enforce a speed threshold. Dense/work byte counts here
# are analytic and exclude allocator peaks and autograd-retained state.
def bev_experiment(K=128,C=8,grid_shape=(12,8,3),device='cpu',dtype=torch.float32):
    from torch_judge.measurement import measure
    generator=torch.Generator().manual_seed(108)
    X,Y,Z=grid_shape
    indices=torch.stack([torch.zeros(K,dtype=torch.int64),torch.randint(Z,(K,),generator=generator),torch.randint(Y,(K,),generator=generator),torch.randint(X,(K,),generator=generator)],dim=1).to(device)
    features=torch.randn(K,C,generator=generator,dtype=dtype).to(device)
    counts=torch.randint(1,6,(K,),generator=generator).to(device)
    inputs=dict(indices=indices,features=features,counts=counts,batch_size=1,grid_shape=torch.tensor(grid_shape,device=device))
    expected=scalar_expected(**inputs)[0]
    reports={}
    for name,fn in [('explicit',scatter_bev_explicit),('index_add',scatter_bev_index_add),('grouped',scatter_bev_grouped)]:
        actual=fn(**inputs)[0]
        reports[name]={'max_abs_error':(actual-expected).abs().max().item(),
                       'timing':measure(lambda:fn(**inputs),device=device,workload={'K':K,'C':C,'grid_xyz':grid_shape,'dtype':str(dtype),'seed':108}),
                       'analytic_required_bev_bytes':C*Y*X*features.element_size(),
                       'analytic_dense_float64_accumulator_bytes':0 if name=='grouped' else C*Y*X*8,
                       'analytic_dense_float64_scale_bytes':C*Y*X*8 if name=='index_add' else 0}
    return reports

# %%
scatter_bev=scatter_bev_index_add

# %% [verifiers]
def scalar_expected(indices,features,counts,batch_size,grid_shape):
    """Python destination groups with exact integer totals and math.fsum means."""
    X,Y,Z=grid_shape.tolist()
    C=features.shape[1]
    values=features.new_zeros(batch_size,C,Y,X)
    totals=torch.zeros(batch_size,1,Y,X,dtype=torch.int64,device=features.device)
    groups={}
    for i,(b,z,y,x) in enumerate(indices.tolist()):
        groups.setdefault((b,y,x),[]).append(i)
    for (b,y,x),rows in groups.items():
        total=sum(counts[i].item() for i in rows)
        totals[b,0,y,x]=total
        for c in range(C):
            values[b,c,y,x]=math.fsum(features[i,c].item()*(counts[i].item()/total) for i in rows)
    return values,totals,totals>0


def make_eval_data():
    cases={}
    def add(name,stage,purpose,indices,features,counts,batch_size=1,grid_shape=(3,2,2),expected=None):
        inputs=dict(indices=indices,features=features,counts=counts,batch_size=batch_size,
                    grid_shape=torch.tensor(grid_shape,dtype=torch.int64,device=features.device))
        cases[name]=EvalCase(name=name,stage=stage,purpose=purpose,inputs=inputs,
                             expected=scalar_expected(**inputs) if expected is None else expected,
                             comparison=partial(compare_values,atol=1e-7,rtol=1e-6))
    idx=torch.tensor([[0,0,0,1],[0,1,0,1],[0,0,1,2]])
    f=torch.tensor([[2.],[8.],[0.]],dtype=torch.float64)
    totals=torch.tensor([[[[0,4,0],[0,0,2]]]])
    add('unequal_z_collision','baseline','Three points at2, one at8:3.5; another cell with zero stays occupied.',idx,f,torch.tensor([3,1,2]),
        expected=(torch.tensor([[[[0.,3.5,0.],[0.,0.,0.]]]],dtype=f.dtype),totals,totals>0))
    add('asymmetric_xy','baseline','X=4,Y=2: y is row, x is column. Last cell contains17.',torch.tensor([[0,0,1,3]]),torch.tensor([[17.,-2.]],dtype=torch.float64),torch.tensor([1]),grid_shape=(4,2,1))
    add('duplicate_indices','followup','Repeated exact voxel indices still accumulate, never overwrite.',torch.tensor([[0,1,1,0],[0,1,1,0],[0,0,1,0]]),torch.tensor([[1.,2.],[5.,-4.],[0.,3.]],dtype=torch.float64),torch.tensor([1,2,3]))
    add('separate_batches','followup','Same xyz in two batches never mixes.',torch.tensor([[0,0,0,0],[1,0,0,0],[1,1,0,0]]),torch.tensor([[3.],[8.],[2.]],dtype=torch.float64),torch.tensor([1,2,1]),batch_size=2)
    add('empty_cells','followup','Empty sparse inputs make dense zero maps with no occupied cells.',torch.empty(0,4,dtype=torch.int64),torch.empty(0,3,dtype=torch.float64,requires_grad=True),torch.empty(0,dtype=torch.int64),batch_size=2)
    add('strided_features','production','Column-sliced tensor keeps channel order and gradients.',idx,torch.arange(18,dtype=torch.float64).reshape(3,6)[:,::2].requires_grad_(),torch.tensor([3,1,2]))
    add('large_float32','production','Raw float32 count multiplication overflows; wider/normalized accumulation stays finite.',torch.tensor([[0,0,0,0],[0,1,0,0]]),torch.tensor([[3e38],[2e38]],dtype=torch.float32),torch.tensor([100,200]))
    add('huge_float64','production','Even float64 numerator can overflow: normalize counts before multiplying.',torch.tensor([[0,0,0,0],[0,1,0,0]]),torch.tensor([[1e308],[5e307]],dtype=torch.float64),torch.tensor([2,3]))
    add('max_finite_mean','production','Seventeen maximum finite values have the same finite mean; normalized rounding must not overflow.',
        torch.zeros(17,4,dtype=torch.int64),torch.full((17,1),torch.finfo(torch.float64).max,dtype=torch.float64),torch.ones(17,dtype=torch.int64),
        expected=(torch.tensor([[[[torch.finfo(torch.float64).max,0.,0.],[0.,0.,0.]]]],dtype=torch.float64),torch.tensor([[[[17,0,0],[0,0,0]]]]),torch.tensor([[[[True,False,False],[False,False,False]]]])))
    add('exact_large_counts','production','Count total2**63-1 remains an exact int64; float weights are approximate.',torch.tensor([[0,0,0,0],[0,1,0,0]]),torch.tensor([[3.],[1.]],dtype=torch.float64),torch.tensor([2**63-2,1]))
    add('feature_gradient','production','Mean derivative for each contributing row is its count/total.',idx,torch.tensor([[2.,3.],[8.,-1.],[0.,4.]],dtype=torch.float64,requires_grad=True),torch.tensor([3,1,2]))
    add('independent_large_totals','production','Distinct cells may each have int64 maximum counts; no global-total restriction.',
        torch.tensor([[0,0,0,0],[0,0,0,1]]),torch.tensor([[2.],[5.]],dtype=torch.float64),torch.tensor([2**63-1,2**63-1]))
    bad=cases['unequal_z_collision'].inputs|dict(counts=torch.tensor([2**63-1,1,2]))
    cases['count_overflow']=EvalCase(name='count_overflow',stage='production',purpose='Overflow is rejected before allocation/accumulation.',inputs=bad,expected_exception=ValueError)
    return cases


def assert_cases(implementation,names):
    for name in names:
        case=make_eval_data()[name]
        assert not case.comparison(implementation(**case.inputs),case.expected),name


def verify_baseline_weighted_collision(scatter_bev):
    assert_cases(scatter_bev,['unequal_z_collision','asymmetric_xy'])


def verify_followup_duplicates_batches_empty(scatter_bev):
    assert_cases(scatter_bev,['duplicate_indices','separate_batches','empty_cells'])


def verify_production_numerical_counts(scatter_bev):
    assert_cases(scatter_bev,['large_float32','huge_float64','max_finite_mean','exact_large_counts','independent_large_totals'])

def verify_production_validation(scatter_bev):
    base=make_eval_data()['unequal_z_collision'].inputs
    changes=[dict(batch_size=0),dict(batch_size=True),dict(grid_shape=torch.tensor([3,0,2])),
             dict(grid_shape=torch.tensor([2**62,3,2])),dict(indices=base['indices'].float()),
             dict(indices=torch.tensor([[0,-1,0,1],[0,1,0,1],[0,0,1,2]])),
             dict(indices=torch.tensor([[1,0,0,1],[0,1,0,1],[0,0,1,2]])),
             dict(counts=torch.tensor([0,1,2])),dict(counts=torch.tensor([2**63-1,1,2])),
             dict(counts=torch.tensor([1.,1.,1.])),dict(features=base['features'].reshape(3)),
             dict(features=torch.full_like(base['features'],float('nan')))]
    for change in changes:
        try:
            scatter_bev(**(base|change))
        except ValueError:
            continue
        raise AssertionError(f'Expected ValueError for {list(change)}')


def verify_production_feature_gradients(scatter_bev):
    case=make_eval_data()['feature_gradient']; inputs=case.inputs
    f=inputs['features']
    assert torch.autograd.gradcheck(lambda x:scatter_bev(**(inputs|dict(features=x)))[0],(f,),eps=1e-6,atol=1e-5)
    y,_,_=scatter_bev(**inputs)
    grad,=torch.autograd.grad(y.sum(),f)
    torch.testing.assert_close(grad,torch.tensor([[.75,.75],[.25,.25],[1.,1.]],dtype=f.dtype))
    # Extreme outputs need analytic gradients, not finite differences beyond
    # representable range. The detached rounding correction preserves 1/17.
    extreme=make_eval_data()['max_finite_mean'].inputs
    extreme_f=extreme['features'].detach().clone().requires_grad_()
    output=scatter_bev(**(extreme|dict(features=extreme_f)))[0]
    gradient,=torch.autograd.grad(output[0,0,0,0],extreme_f)
    torch.testing.assert_close(gradient,torch.full_like(extreme_f,1/17),atol=1e-14,rtol=1e-12)
    empty=make_eval_data()['empty_cells'].inputs
    y,_,_=scatter_bev(**empty)
    grad,=torch.autograd.grad(y.sum(),empty['features'])
    assert grad.shape==empty['features'].shape and grad.numel()==0


def verify_production_strides_dtype_and_immutability(scatter_bev):
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        for name in ['strided_features','duplicate_indices','empty_cells']:
            case=make_eval_data()[name]
            for dtype in (torch.float32,torch.float64):
                inputs={k:(v.to(device=device,dtype=dtype) if v.is_floating_point() else v.to(device)) if isinstance(v,torch.Tensor) else v for k,v in case.inputs.items()}
                snapshots={k:v.clone() for k,v in inputs.items() if isinstance(v,torch.Tensor)}
                expected=scalar_expected(**inputs)
                assert not compare_values(scatter_bev(**inputs),expected,atol=1e-6,rtol=1e-5)
                for k,v in snapshots.items():
                    assert torch.equal(inputs[k],v),k


def verify_production_direct_point_mean_property(scatter_bev):
    @settings(max_examples=20,derandomize=True,deadline=None)
    @given(rows=st.lists(st.tuples(st.integers(0,1),st.integers(0,1),st.integers(0,1),st.integers(0,2),st.integers(1,4),st.integers(-10,10)),max_size=12))
    def trial(rows):
        indices=torch.tensor([r[:4] for r in rows],dtype=torch.int64).reshape(-1,4)
        counts=torch.tensor([r[4] for r in rows],dtype=torch.int64)
        features=torch.tensor([[r[5]] for r in rows],dtype=torch.float64).reshape(-1,1)
        grid=torch.tensor([3,2,2])
        got,total,occupied=scatter_bev(indices,features,counts,2,grid)
        point_groups={}
        # Independently expand the original point multiset, not weighted voxels.
        for b,z,y,x,count,value in rows:
            point_groups.setdefault((b,y,x),[]).extend([value]*count)
        expected=features.new_zeros(2,1,2,3)
        for (b,y,x),points in point_groups.items():
            expected[b,0,y,x]=sum(points)/len(points)
            assert total[b,0,y,x].item()==len(points)
        torch.testing.assert_close(got,expected)
        assert torch.equal(occupied,total>0)
        perm=torch.arange(len(rows)-1,-1,-1)
        reordered=scatter_bev(indices[perm],features[perm],counts[perm],2,grid)
        torch.testing.assert_close(reordered[0],got)
        assert torch.equal(reordered[1],total)
    trial()


# %%
if not evaluate('0108_bev_scatter_eval.py',scatter_bev):
    raise SystemExit(1)
