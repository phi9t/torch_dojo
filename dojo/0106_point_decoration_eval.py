"""Point descriptors with sparse, batch-separated occupied-voxel centroids."""
# %%
import math
from decimal import Decimal, localcontext, ROUND_FLOOR
from functools import partial
import torch
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values

# %%
def _validate(points, attributes, point_valid, origin, voxel_size, grid_shape):
    if not isinstance(points,torch.Tensor) or points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('points must be [B,N,3]')
    if points.shape[0] < 1 or points.dtype not in (torch.float32,torch.float64):
        raise ValueError('positive B and float32/float64 required')
    b,n,_ = points.shape
    if (not isinstance(attributes,torch.Tensor) or attributes.ndim != 3
            or attributes.shape[:2] != (b,n) or attributes.dtype != points.dtype
            or attributes.device != points.device):
        raise ValueError('attributes must be [B,N,F] with point dtype/device')
    if (not isinstance(point_valid,torch.Tensor) or point_valid.shape != (b,n)
            or point_valid.dtype != torch.bool or point_valid.device != points.device):
        raise ValueError('point_valid must be boolean [B,N] on point device')
    for name,value in [('origin',origin),('voxel_size',voxel_size)]:
        if (not isinstance(value,torch.Tensor) or value.shape != (3,)
                or value.dtype != points.dtype or value.device != points.device
                or not torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite [3] with point dtype/device')
    if not (voxel_size > 0).all():
        raise ValueError('voxel_size must be positive')
    if (not isinstance(grid_shape,torch.Tensor) or grid_shape.shape != (3,)
            or grid_shape.dtype != torch.int64 or grid_shape.device != points.device
            or not (grid_shape > 0).all()):
        raise ValueError('grid_shape must be positive int64 [3] on point device')
    if not torch.isfinite(points[point_valid]).all():
        raise ValueError('valid points must be finite')
    upper = origin.double() + voxel_size.double()*grid_shape.double()
    if not torch.isfinite(upper).all():
        raise ValueError('grid upper bounds must be finite in float64')
    inside = point_valid & ((points.double() >= origin.double()) & (points.double() < upper)).all(dim=-1)
    if not torch.isfinite(attributes[inside]).all():
        raise ValueError('inside attributes must be finite')
    return inside

# %%
def _included_cells(points,origin,voxel_size,grid_shape):
    # Call only on physically included points. Avoid undefined int64 casts,
    # then correct a rounded upper cell coordinate without changing membership.
    cells=torch.floor((points-origin)/voxel_size)
    if not (torch.isfinite(cells).all() and (cells>=0).all() and (cells<2**63).all()):
        raise ValueError('Included cell coordinate is not representable as int64')
    return torch.minimum(cells.to(torch.int64),grid_shape-1)

# %%
# Reference 1: visit occupied voxels, then build descriptors point by point.
# Baseline/followup; ordinary finite float32/float64 magnitudes. Mean and range
# use input precision, so squared large coordinates can overflow. Sparse state
# O(B*N); outputs O(B*N*(F+10)); Python grouping adds substantial overhead.
def decorate_loop(points, attributes, point_valid, origin, voxel_size, grid_shape):
    inside = _validate(points,attributes,point_valid,origin,voxel_size,grid_shape)
    groups = {}
    for batch in range(points.shape[0]):
        for row in range(points.shape[1]):
            if bool(inside[batch,row]):
                cell = _included_cells(points[batch,row],origin,voxel_size,grid_shape)
                key = (batch,*cell.tolist())
                groups.setdefault(key,[]).append(row)
    output = torch.zeros((*points.shape[:2],attributes.shape[-1]+10),dtype=points.dtype,device=points.device)
    # Retain zero gradient paths even when every row is excluded or N=0.
    output = output + torch.where(inside[...,None],points,0).sum()*0
    output = output + torch.where(inside[...,None],attributes,0).sum()*0
    for key,rows in groups.items():
        batch = key[0]
        centroid = points[batch,rows].mean(dim=0)
        center = origin+(torch.tensor(key[1:],device=points.device,dtype=points.dtype)+.5)*voxel_size
        for row in rows:
            xyz = points[batch,row]
            output[batch,row] = torch.cat((attributes[batch,row],xyz,xyz-centroid,xyz-center,
                torch.sqrt(xyz.square().sum()).reshape(1)))
    return output,inside

# %%
# Reference 2: sparse unique coordinates and indexed addition, input precision.
# Baseline/followup. No grid-sized allocation or packed integer keys. Grouping
# uses [batch,x,y,z] internally and the final output retains original row order.
def decorate_grouped(points, attributes, point_valid, origin, voxel_size, grid_shape):
    inside = _validate(points,attributes,point_valid,origin,voxel_size,grid_shape)
    locations = inside.nonzero()
    xyz = points[inside]
    cells = _included_cells(xyz,origin,voxel_size,grid_shape)
    keys = torch.cat((locations[:,:1],cells),dim=1)
    unique,inverse,counts = torch.unique(keys,dim=0,sorted=True,return_inverse=True,return_counts=True)
    sums = torch.zeros((unique.shape[0],3),dtype=points.dtype,device=points.device)
    sums = sums.index_add(0,inverse,xyz)
    centroids = sums/counts[:,None]
    centers = origin+(cells.to(points.dtype)+.5)*voxel_size
    descriptor = torch.cat((attributes[inside],xyz,xyz-centroids[inverse],xyz-centers,
        torch.sqrt(xyz.square().sum(dim=1,keepdim=True))),dim=1)
    output = torch.zeros((*points.shape[:2],attributes.shape[-1]+10),dtype=points.dtype,device=points.device)
    output[inside] = descriptor
    return output,inside

# %%
# Reference 3: group once, widen float32 geometry/reductions to float64, and
# divide each contribution BEFORE reduction. Range rescales coordinates before
# squaring, including float64 large-magnitude inputs. Production qualified.
# Working storage O(B*N + occupied*3), never O(product(grid_shape)). Discrete
# floor/grouping/membership have no derivative across voxel boundaries.
def decorate_stable(points, attributes, point_valid, origin, voxel_size, grid_shape):
    inside = _validate(points,attributes,point_valid,origin,voxel_size,grid_shape)
    locations = inside.nonzero()
    xyz = points[inside].to(torch.float64)
    cells = _included_cells(xyz,origin.double(),voxel_size.double(),grid_shape)
    keys = torch.cat((locations[:,:1],cells),dim=1)
    unique,inverse,counts = torch.unique(keys,dim=0,sorted=True,return_inverse=True,return_counts=True)
    contributions = xyz/counts[inverse,None]
    centroids = torch.zeros((unique.shape[0],3),dtype=xyz.dtype,device=points.device).index_add(0,inverse,contributions)
    centers = origin.double()+(cells.double()+.5)*voxel_size.double()
    scale = xyz.abs().amax(dim=1,keepdim=True)
    normalized = xyz/torch.where(scale > 0,scale,1)
    ranges = torch.linalg.vector_norm(normalized,dim=1,keepdim=True)*scale
    descriptor = torch.cat((attributes[inside].double(),xyz,xyz-centroids[inverse],xyz-centers,ranges),dim=1)
    output = torch.zeros((*points.shape[:2],attributes.shape[-1]+10),dtype=points.dtype,device=points.device)
    output[inside] = descriptor.to(points.dtype)
    return output,inside

# %%
def explore_range_overflow():
    """Opt in: range is finite although coordinate squares overflow float32."""
    points = torch.tensor([[[1e20,1e20,1e20]]],dtype=torch.float32)
    inputs = dict(points=points,attributes=torch.ones(1,1,1),point_valid=torch.ones(1,1,dtype=torch.bool),
        origin=torch.zeros(3),voxel_size=torch.full((3,),2e20),grid_shape=torch.ones(3,dtype=torch.int64))
    return {'inputs':inputs,'loop_range':decorate_loop(**inputs)[0][...,-1],
        'stable_range':decorate_stable(**inputs)[0][...,-1],
        'analytic_range':math.hypot(*points[0,0].tolist())}

# %%
def benchmark_decoration(batch=2,points_per_batch=512,device='cpu',dtype=torch.float32):
    from torch_judge.measurement import measure
    generator = torch.Generator(device=device).manual_seed(106)
    points = torch.rand((batch,points_per_batch,3),dtype=dtype,device=device,generator=generator)*20
    inputs = dict(points=points,attributes=torch.rand((batch,points_per_batch,2),dtype=dtype,device=device,generator=generator),
        point_valid=torch.ones((batch,points_per_batch),dtype=torch.bool,device=device),
        origin=torch.zeros(3,dtype=dtype,device=device),voxel_size=torch.ones(3,dtype=dtype,device=device)*2,
        grid_shape=torch.full((3,),10,dtype=torch.int64,device=device))
    expected = decorate_stable(**inputs)[0]
    reports = {}
    for name,fn in [('loop',decorate_loop),('grouped',decorate_grouped),('stable',decorate_stable)]:
        actual = fn(**inputs)[0]
        reports[name] = {'max_error':(actual-expected).abs().max().item(),
            'timing':measure(lambda:fn(**inputs),device=device,
                workload={'batch':batch,'points_per_batch':points_per_batch,'dtype':str(dtype),'seed':106,'grid_xyz':(10,10,10)})}
    return reports

# %%
decorate_points = decorate_stable

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


def _scalar_expected(points,attributes,point_valid,origin,voxel_size,grid_shape):
    b,n,_ = points.shape
    f = attributes.shape[-1]
    output = torch.zeros((b,n,f+10),dtype=points.dtype,device=points.device)
    inside = torch.zeros((b,n),dtype=torch.bool,device=points.device)
    groups = {}
    keys = {}
    for batch in range(b):
        for row in range(n):
            if not bool(point_valid[batch,row]):
                continue
            xyz = [float(v) for v in points[batch,row]]
            cells = tuple(_scalar_cell(xyz[a],origin[a],voxel_size[a]) for a in range(3))
            if not all(0 <= cells[a] < int(grid_shape[a]) for a in range(3)):
                continue
            inside[batch,row] = True
            key = (batch,*cells)
            groups.setdefault(key,[]).append(xyz)
            keys[(batch,row)] = key
    for (batch,row),key in keys.items():
        xyz = [float(v) for v in points[batch,row]]
        members = groups[key]
        centroid = [math.fsum(p[a]/len(members) for p in members) for a in range(3)]
        center = [float(origin[a])+(key[a+1]+.5)*float(voxel_size[a]) for a in range(3)]
        descriptor = [float(v) for v in attributes[batch,row]] + xyz
        descriptor += [xyz[a]-centroid[a] for a in range(3)]
        descriptor += [xyz[a]-center[a] for a in range(3)]
        descriptor += [math.hypot(*xyz)]
        output[batch,row] = torch.tensor(descriptor,dtype=points.dtype,device=points.device)
    return output,inside


def make_eval_data():
    def case(name,stage,points,attributes,valid=None,origin=(0.,0.,0.),size=(2.,2.,2.),grid=(3,3,3),dtype=torch.float64,purpose=''):
        points = torch.as_tensor(points,dtype=dtype)
        attributes = torch.as_tensor(attributes,dtype=dtype)
        valid = torch.ones(points.shape[:2],dtype=torch.bool) if valid is None else torch.as_tensor(valid,dtype=torch.bool)
        inputs = dict(points=points,attributes=attributes,point_valid=valid,
            origin=torch.tensor(origin,dtype=dtype),voxel_size=torch.tensor(size,dtype=dtype),
            grid_shape=torch.tensor(grid,dtype=torch.int64))
        return EvalCase(name=name,stage=stage,purpose=purpose,inputs=inputs,
            expected=_scalar_expected(**inputs),comparison=partial(compare_values,atol=1e-8,rtol=1e-6))
    cases = [
        case('one_voxel','baseline',[[[.5,.5,.5],[1.5,.5,.5]]],[[[2.],[6.]]],purpose='Centroid [1,.5,.5], offsets [-.5,0,0] and [.5,0,0].'),
        case('boundary_rows','baseline',[[[0.,0.,0.],[6.,0.,0.],[-.1,0.,0.]]],[[[1.],[2.],[3.]]],purpose='Origin included; upper boundary and negative x excluded.'),
        case('batch_duplicate','followup',[[[.5,.5,.5],[.5,.5,.5],[1.5,.5,.5]],[[.2,.4,.6],[2.5,.5,.5],[2.5,.5,.5]]],[[[1.],[2.],[3.]],[[4.],[5.],[6.]]],purpose='Duplicates contribute; centroids cannot mix batches.'),
        case('anisotropic','followup',[[[-1.,2.,.5],[.5,3.,2.],[5.,3.,2.]]],[[[1.,2.],[3.,4.],[5.,6.]]],origin=(-2.,1.,0.),size=(2.,3.,4.),grid=(3,2,1),purpose='Different xyz cell widths, translated origin, upper-x exclusion.'),
        case('empty_points','followup',torch.empty(2,0,3),torch.empty(2,0,2),purpose='Descriptor shape remains [2,0,12].'),
        case('masked_padding','production',[[[.5,.5,.5],[float('nan')]*3,[7.,1.,1.]]],[[[3.],[float('nan')],[float('nan')]]],valid=[[True,False,True]],purpose='Padding and outside attributes cannot enter centroid arithmetic.'),
        case('wide_range','production',[[[1e20,1e20,1e20]]],[[[1.]]],size=(2e20,2e20,2e20),grid=(1,1,1),dtype=torch.float32,purpose='A finite range should not overflow when its squared coordinates do.'),
        case('wide_double_range','production',[[[1e200,1e200,1e200]]],[[[1.]]],size=(2e200,2e200,2e200),grid=(1,1,1),purpose='Float64 also needs a scaled norm; widening alone is insufficient.'),
        case('sparse_huge_grid','production',[[[.5,.5,.5],[2.5,.5,.5]]],[[[1.],[2.]]],grid=(1000000000,1000000000,1000000000),purpose='Only occupied voxels matter; no packed int64 product or dense grid.'),
    ]
    for dtype in (torch.float32,torch.float64):
        tiny=torch.finfo(dtype).tiny
        fixture=case('near_zero_upper_'+str(dtype).split('.')[-1],'followup',[[[-tiny,-.5,-.5],[-.5,-.5,-.5],[0.,-.5,-.5],[tiny,-.5,-.5]]],[[[1.],[2.],[3.],[4.]]],origin=(-1.,-1.,-1.),size=(1.,1.,1.),grid=(1,1,1),dtype=dtype)
        literal=torch.zeros(1,4,11,dtype=dtype)
        literal[0,0]=torch.tensor([1.,-tiny,-.5,-.5,.25,0.,0.,.5,0.,0.,math.sqrt(.5)],dtype=dtype)
        literal[0,1]=torch.tensor([2.,-.5,-.5,-.5,-.25,0.,0.,0.,0.,0.,math.sqrt(.75)],dtype=dtype)
        cases.append(EvalCase(name=fixture.name,stage='followup',purpose='Both interior rows share one centroid at x=-.25; zero/positive edge rows are outside.',inputs=fixture.inputs,expected=(literal,torch.tensor([[True,True,False,False]]))))
    result = {item.name:item for item in cases}
    invalid = dict(result['one_voxel'].inputs)
    invalid['voxel_size'] = torch.tensor([0.,2.,2.],dtype=torch.float64)
    result['invalid_size'] = EvalCase(name='invalid_size',stage='production',purpose='Zero voxel width is an API error.',inputs=invalid,expected_exception=ValueError)
    return result


def _assert_case(implementation,case):
    errors = case.comparison(implementation(**case.inputs),case.expected)
    assert not errors,'\n'.join(errors)


def verify_followup_near_zero_upper_literal(decorate_points):
    for dtype in (torch.float32,torch.float64):
        _assert_case(decorate_points,make_eval_data()['near_zero_upper_'+str(dtype).split('.')[-1]])


def verify_baseline_layout_and_bounds(decorate_points):
    for name in ('one_voxel','boundary_rows'):
        _assert_case(decorate_points,make_eval_data()[name])


def verify_followup_batches_and_empty(decorate_points):
    for name in ('batch_duplicate','anisotropic','empty_points'):
        _assert_case(decorate_points,make_eval_data()[name])


def verify_followup_permutations(decorate_points):
    @settings(max_examples=12,deadline=None,derandomize=True)
    @given(st.lists(st.integers(0,15),min_size=9,max_size=9))
    def trial(numbers):
        # Quarter-cell offsets stay strictly away from membership boundaries.
        xyz = torch.tensor(numbers,dtype=torch.float64).reshape(1,3,3)/4+.125
        attributes = torch.tensor([[[1.],[2.],[3.]]],dtype=torch.float64)
        inputs = dict(points=xyz,attributes=attributes,point_valid=torch.ones((1,3),dtype=torch.bool),
            origin=torch.zeros(3,dtype=torch.float64),voxel_size=torch.ones(3,dtype=torch.float64)*2,
            grid_shape=torch.ones(3,dtype=torch.int64)*3)
        expected = _scalar_expected(**inputs)
        actual = decorate_points(**inputs)
        assert not compare_values(actual,expected,atol=1e-10,rtol=1e-10)
        perm = torch.tensor([2,0,1])
        inputs.update(points=xyz[:,perm],attributes=attributes[:,perm])
        reordered = decorate_points(**inputs)
        torch.testing.assert_close(reordered[0],expected[0][:,perm])
        assert torch.equal(reordered[1],expected[1][:,perm])
    trial()


def verify_production_large_range_and_padding(decorate_points):
    for name in ('wide_range','wide_double_range','masked_padding','sparse_huge_grid'):
        _assert_case(decorate_points,make_eval_data()[name])


def verify_production_gradients(decorate_points):
    case = make_eval_data()['one_voxel']
    inputs = case.inputs
    inputs['points'].requires_grad_(); inputs['attributes'].requires_grad_()
    def call(p,a):
        return decorate_points(**dict(inputs,points=p,attributes=a))[0]
    assert torch.autograd.gradcheck(call,(inputs['points'],inputs['attributes']))
    descriptor,_ = decorate_points(**inputs)
    # First point's x-centroid offset has derivative [.5,-.5] across members.
    gradients = torch.autograd.grad(descriptor[0,0,4],inputs['points'],retain_graph=True)[0]
    expected = torch.zeros_like(gradients)
    expected[0,0,0] = .5; expected[0,1,0] = -.5
    torch.testing.assert_close(gradients,expected)
    attr_grad = torch.autograd.grad(descriptor[...,0].sum(),inputs['attributes'])[0]
    assert torch.equal(attr_grad,torch.ones_like(attr_grad))
    # Invalid NaN padding is not part of geometry or its backward graph.
    inputs = make_eval_data()['masked_padding'].inputs
    inputs['points'].requires_grad_(); inputs['attributes'].requires_grad_()
    descriptor,_ = decorate_points(**inputs)
    gp,ga = torch.autograd.grad(descriptor.sum(),(inputs['points'],inputs['attributes']))
    assert torch.isfinite(gp).all() and torch.isfinite(ga).all()
    assert torch.equal(gp[0,1:],torch.zeros_like(gp[0,1:]))
    assert torch.equal(ga[0,1:],torch.zeros_like(ga[0,1:]))
    inputs['point_valid'] = torch.zeros_like(inputs['point_valid'])
    descriptor,inside = decorate_points(**inputs)
    gp,ga = torch.autograd.grad(descriptor.sum(),(inputs['points'],inputs['attributes']))
    assert torch.equal(descriptor,torch.zeros_like(descriptor)) and not inside.any()
    assert torch.equal(gp,torch.zeros_like(gp)) and torch.equal(ga,torch.zeros_like(ga))


def verify_production_strides_and_immutability(decorate_points):
    for dtype in (torch.float32,torch.float64):
        points = torch.arange(24,dtype=dtype).reshape(2,3,4).transpose(1,2)/10+.125
        attributes = torch.arange(16,dtype=dtype).reshape(2,2,4).transpose(1,2)
        valid = torch.ones(points.shape[:2],dtype=torch.bool)
        inputs = dict(points=points,attributes=attributes,point_valid=valid,
            origin=torch.zeros(3,dtype=dtype),voxel_size=torch.ones(3,dtype=dtype)*2,
            grid_shape=torch.ones(3,dtype=torch.int64)*3)
        originals = {name:value.clone() for name,value in inputs.items()}
        expected = _scalar_expected(**inputs)
        assert not compare_values(decorate_points(**inputs),expected,atol=1e-6,rtol=1e-6)
        for name,value in inputs.items():
            assert torch.equal(value,originals[name]),name


def verify_production_metadata(decorate_points):
    inputs = make_eval_data()['one_voxel'].inputs
    replacements = [('voxel_size',torch.tensor([0.,2.,2.],dtype=torch.float64)),
        ('origin',torch.tensor([float('nan'),0.,0.],dtype=torch.float64)),
        ('grid_shape',torch.tensor([2,0,2],dtype=torch.int64)),
        ('grid_shape',torch.tensor([2.,2.,2.],dtype=torch.float64)),
        ('point_valid',torch.ones(1,2)),('points',torch.ones(1,2,2,dtype=torch.float64)),
        ('attributes',torch.ones(1,2,1,dtype=torch.float32)),
        ('points',torch.full((1,2,3),float('nan'),dtype=torch.float64))]
    for name,value in replacements:
        try:
            decorate_points(**dict(inputs,**{name:value}))
        except ValueError:
            continue
        raise AssertionError(f'invalid {name} should raise ValueError')

# %%
from torch_judge import evaluate
if not evaluate('0106_point_decoration_eval.py',decorate_points):
    raise SystemExit(1)
