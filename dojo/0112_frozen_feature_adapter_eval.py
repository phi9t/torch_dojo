"""Frozen weights, controlled model state, and trainable dense feature adapters."""
# %%
from functools import partial
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values

# %%
def _initialize_adapter(module,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image):
    if not isinstance(backbone,nn.Module):
        raise ValueError('backbone must be nn.Module returning [B,L,C]')
    if (not isinstance(grid_hw,(tuple,list)) or len(grid_hw) != 2
            or any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in grid_hw)):
        raise ValueError('grid_hw must be positive integer (height,width)')
    if isinstance(prefix_tokens,bool) or not isinstance(prefix_tokens,int) or prefix_tokens < 0:
        raise ValueError('prefix_tokens must be a nonnegative integer')
    if any(isinstance(v,bool) or not isinstance(v,int) or v < 1 for v in (in_channels,out_channels)):
        raise ValueError('channel counts must be positive integers')
    if (not isinstance(feature_from_image,torch.Tensor) or feature_from_image.shape != (3,3)
            or feature_from_image.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(feature_from_image).all()):
        raise ValueError('feature_from_image must be finite float32/float64 [3,3]')
    if not torch.equal(feature_from_image[2],feature_from_image.new_tensor([0.,0.,1.])):
        raise ValueError('feature_from_image must be affine with last row [0,0,1]')
    if torch.linalg.det(feature_from_image[:2,:2]) == 0:
        raise ValueError('feature_from_image linear block must be invertible')
    module.grid_hw = tuple(grid_hw)
    module.prefix_tokens = prefix_tokens
    module.in_channels = in_channels
    module.out_channels = out_channels
    module.backbone = backbone.requires_grad_(False).eval()
    module.adapter = nn.Conv2d(in_channels,out_channels,kernel_size=1).to(
        device=feature_from_image.device,dtype=feature_from_image.dtype)
    # Calibration is fixed metadata, copied into a buffer for .to()/state_dict.
    module.register_buffer('feature_from_image',feature_from_image.detach().clone())


def _validate_images(module,images):
    if (not isinstance(images,torch.Tensor) or images.ndim != 4 or min(images.shape[1:]) < 1
            or images.dtype != module.adapter.weight.dtype or images.device != module.adapter.weight.device
            or not torch.isfinite(images).all()):
        raise ValueError('images must be finite BCHW matching adapter dtype/device')


def _validate_tokens(module,tokens,images):
    h,w = module.grid_hw
    if (not isinstance(tokens,torch.Tensor) or tokens.shape != (images.shape[0],module.prefix_tokens+h*w,module.in_channels)
            or tokens.dtype != images.dtype or tokens.device != images.device
            or not torch.isfinite(tokens[:,module.prefix_tokens:]).all()):
        raise ValueError('backbone output must match batch, explicit token layout, dtype/device and finite spatial features')

# %%
# Reference 1: explicit dense reconstruction then ordinary 1x1 convolution.
# Production qualified. Freezing PARAMETERS still permits derivatives through
# the backbone with respect to images. The supplied backbone must honor eval
# state (no training dropout or buffer updates); arbitrary custom eval side
# effects are outside this wrapper's purification scope.
class FrozenFeatureAdapterSimple(nn.Module):
    def __init__(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image):
        super().__init__()
        _initialize_adapter(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image)

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self,images):
        _validate_images(self,images)
        tokens = self.backbone(images)
        _validate_tokens(self,tokens,images)
        h,w = self.grid_hw; b = images.shape[0]
        spatial = tokens[:,self.prefix_tokens:]
        # A contiguous dense copy makes the geometry/layout boundary visible.
        dense = spatial.reshape(b,h,w,self.in_channels).permute(0,3,1,2).contiguous()
        maps = self.adapter(dense)
        return maps,self.feature_from_image[None].expand(b,-1,-1)

# %%
# Reference 2: perform the same channel projection on spatial tokens BEFORE
# materializing a dense map. Production qualified. Uses the identical Conv2d
# weights reshaped as a Linear matrix; avoids the explicit B*C*H*W input-map
# copy. Output is a strided view; backend workspace/backward state are separate.
class FrozenFeatureAdapterTokens(nn.Module):
    def __init__(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image):
        super().__init__()
        _initialize_adapter(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image)

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self,images):
        _validate_images(self,images)
        tokens = self.backbone(images)
        _validate_tokens(self,tokens,images)
        spatial = tokens[:,self.prefix_tokens:]
        adapted = F.linear(spatial,self.adapter.weight[:,:,0,0],self.adapter.bias)
        h,w = self.grid_hw; b = images.shape[0]
        maps = adapted.reshape(b,h,w,self.out_channels).permute(0,3,1,2)
        return maps,self.feature_from_image[None].expand(b,-1,-1)

# %%
# Reference 3: deliberately NARROW inference-only domain. No input or adapter
# gradients; rejects requires_grad images rather than silently detaching them.
# Forward-only baseline parity, NOT followup/production training qualification.
# No implicit cache: cached features would need image/model/preprocessing keys.
class FrozenFeatureAdapterInference(nn.Module):
    def __init__(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image):
        super().__init__()
        _initialize_adapter(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image)

    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self,images):
        _validate_images(self,images)
        if images.requires_grad:
            raise ValueError('inference-only reference cannot preserve requested input gradients')
        with torch.no_grad():
            tokens = self.backbone(images)
            _validate_tokens(self,tokens,images)
            h,w = self.grid_hw; b = images.shape[0]
            dense = tokens[:,self.prefix_tokens:].reshape(b,h,w,self.in_channels).permute(0,3,1,2)
            maps = self.adapter(dense)
        return maps,self.feature_from_image[None].expand(b,-1,-1)

# %%
def explore_detached_features(inputs):
    """Opt in with eval_data['numbered_images'].inputs: demonstrate the bug."""
    configuration = {name:inputs[name] for name in ('backbone','grid_hw','prefix_tokens','in_channels','out_channels','feature_from_image')}
    model = FrozenFeatureAdapterSimple(**configuration)
    with torch.no_grad():
        model.adapter.weight.copy_(inputs['adapter_weight'])
        model.adapter.bias.copy_(inputs['adapter_bias'])
    bad_images = inputs['images'].detach().clone().requires_grad_()
    with torch.no_grad():
        detached_tokens = model.backbone(bad_images)
    h,w = model.grid_hw
    dense = detached_tokens[:,model.prefix_tokens:].reshape(bad_images.shape[0],h,w,model.in_channels).permute(0,3,1,2)
    bad = model.adapter(dense)
    bad_input_gradient = torch.autograd.grad(bad.sum(),bad_images,allow_unused=True)[0]
    good_images = inputs['images'].detach().clone().requires_grad_()
    good = model(good_images)[0]
    good_input_gradient = torch.autograd.grad(good.sum(),good_images)[0]
    return {'same_forward':torch.equal(bad,good),'detached_input_gradient':bad_input_gradient,
        'preserved_input_gradient':good_input_gradient,'note':'Frozen parameters do not require no_grad around image features.'}

# %%
def benchmark_adapters(inputs):
    """Shared-weight INFERENCE comparison; supply an inspectable case's inputs."""
    from torch_judge.measurement import measure
    images = inputs['images'].detach()
    reports = {}
    expected = None
    for name,cls in [('dense',FrozenFeatureAdapterSimple),('tokens',FrozenFeatureAdapterTokens),('inference',FrozenFeatureAdapterInference)]:
        configuration = {key:inputs[key] for key in ('grid_hw','prefix_tokens','in_channels','out_channels','feature_from_image')}
        model = cls(backbone=deepcopy(inputs['backbone']),**configuration)
        with torch.no_grad():
            model.adapter.weight.copy_(inputs['adapter_weight'])
            model.adapter.bias.copy_(inputs['adapter_bias'])
            actual = model(images)[0]
            if expected is None:
                expected = actual
        def run():
            with torch.no_grad():
                return model(images)
        h,w = inputs['grid_hw']
        reports[name] = {'max_error':(actual-expected).abs().max().item() if actual.numel() else 0.,
            'explicit_input_map_bytes':images.shape[0]*inputs['in_channels']*h*w*images.element_size() if name == 'dense' else 0,
            'storage_note':'Analytical explicit contiguous input map, not allocator peaks, backend workspace or backward state.',
            'timing':measure(run,device=images.device,workload={'images':tuple(images.shape),'grid_hw':(h,w),
                'input_channels':inputs['in_channels'],'output_channels':inputs['out_channels'],'dtype':str(images.dtype),
                'mode':'all variants eval-state, global no_grad; inference comparison only'})}
    return reports

# %%
FrozenFeatureAdapter = FrozenFeatureAdapterTokens

# %% [verifiers]
class _TinyBackbone(nn.Module):
    """Known features plus deliberate training-only randomness and buffer edits."""
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor([2.,3.],dtype=torch.float64))
        self.register_buffer('training_calls',torch.zeros((),dtype=torch.int64))
        self.dropout = nn.Dropout(.75)

    def forward(self,images):
        if self.training:
            self.training_calls.add_(1)
        scalar = images[:,0].flatten(1)
        tokens = torch.stack((scalar*self.gain[0]+1,scalar*self.gain[1]-2),dim=-1)
        tokens = self.dropout(tokens)
        prefix = images.new_full((images.shape[0],1,2),1000)
        return torch.cat((prefix,tokens),dim=1)


def _literal_expected(images,weight,bias,transform):
    # Independent image-pixel scalar arithmetic; no backbone/ref invocation.
    b,_,h,w = images.shape
    output = images.new_empty((b,weight.shape[0],h,w))
    for batch in range(b):
        for row in range(h):
            for col in range(w):
                pixel = float(images.detach()[batch,0,row,col])
                features = (2*pixel+1,3*pixel-2)
                for channel in range(weight.shape[0]):
                    output[batch,channel,row,col] = float(bias[channel])+sum(
                        features[k]*float(weight[channel,k,0,0]) for k in range(2))
    return output,transform[None].expand(b,-1,-1).clone()


def make_eval_data():
    weight = torch.tensor([[[[1.]],[[2.]]],[[[-1.]],[[.5]]]],dtype=torch.float64)
    bias = torch.tensor([.25,-.5],dtype=torch.float64)
    transform = torch.tensor([[.5,0.,-.25],[0.,.5,-.25],[0.,0.,1.]],dtype=torch.float64)
    cases = {}
    for name,stage,images in [('numbered_images','baseline',torch.arange(6,dtype=torch.float64).reshape(1,1,2,3)),
        ('batched_images','followup',torch.arange(12,dtype=torch.float64).reshape(2,1,2,3)),
        ('strided_images','production',torch.arange(6,dtype=torch.float64).reshape(1,1,3,2).transpose(2,3)),
        ('input_gradient','production',torch.arange(6,dtype=torch.float64).reshape(1,1,2,3).requires_grad_()),
        ('empty_batch','production',torch.empty(0,1,2,3,dtype=torch.float64))]:
        backbone = _TinyBackbone().eval().requires_grad_(False)
        inputs = dict(images=images,backbone=backbone,grid_hw=(2,3),prefix_tokens=1,
            in_channels=2,out_channels=2,feature_from_image=transform.clone(),
            adapter_weight=weight.clone(),adapter_bias=bias.clone())
        cases[name] = EvalCase(name=name,stage=stage,purpose='Construct your adapter, copy these explicit weights, then inspect maps and geometry metadata.',
            inputs=inputs,expected=_literal_expected(images,weight,bias,transform),
            comparison=partial(compare_values,atol=1e-10,rtol=1e-10))
    invalid = dict(cases['numbered_images'].inputs); invalid['grid_hw'] = (3,3)
    cases['wrong_token_layout'] = EvalCase(name='wrong_token_layout',stage='production',
        purpose='Supplied token grid disagrees with backbone output.',inputs=invalid,expected_exception=ValueError)
    singular=dict(cases['numbered_images'].inputs)
    singular['feature_from_image']=torch.tensor([[1.,2.,0.],[2.,4.,0.],[0.,0.,1.]],dtype=torch.float64)
    cases['singular_feature_affine']=EvalCase(name='singular_feature_affine',stage='production',purpose='Finite affine metadata still needs an invertible linear block before sampling.',inputs=singular,expected_exception=ValueError)
    return cases


def _build_candidate(FrozenFeatureAdapter,inputs):
    configuration = {name:inputs[name] for name in ('backbone','grid_hw','prefix_tokens','in_channels','out_channels','feature_from_image')}
    model = FrozenFeatureAdapter(**configuration)
    with torch.no_grad():
        model.adapter.weight.copy_(inputs['adapter_weight'])
        model.adapter.bias.copy_(inputs['adapter_bias'])
    return model


def verify_baseline_frozen_forward(FrozenFeatureAdapter):
    case = make_eval_data()['numbered_images']
    model = _build_candidate(FrozenFeatureAdapter,case.inputs)
    errors = case.comparison(model(case.inputs['images']),case.expected)
    assert not errors,'\n'.join(errors)
    assert not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())


def verify_followup_batched_and_requested_gradients(FrozenFeatureAdapter):
    case = make_eval_data()['batched_images']
    images = case.inputs['images'].requires_grad_()
    model = _build_candidate(FrozenFeatureAdapter,case.inputs)
    maps,metadata = model(images)
    assert not case.comparison((maps,metadata),case.expected)
    maps.sum().backward()
    # Fixed feature/adapter slopes: (2+6)+(-2+1.5)=7.5 per input pixel.
    torch.testing.assert_close(images.grad,torch.full_like(images,7.5))
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.adapter.parameters())
    # Bias gets one contribution per pixel, for each output channel.
    torch.testing.assert_close(model.adapter.bias.grad,torch.full_like(model.adapter.bias,12.))
    pixels = images.detach().flatten()
    sums = torch.stack(((2*pixels+1).sum(),(3*pixels-2).sum()))
    expected_weight_gradient = sums[None,:,None,None].expand_as(model.adapter.weight)
    torch.testing.assert_close(model.adapter.weight.grad,expected_weight_gradient)
    assert torch.autograd.gradcheck(lambda x:model(x)[0],(images.detach().requires_grad_(),))


def verify_followup_state_policy(FrozenFeatureAdapter):
    inputs = make_eval_data()['numbered_images'].inputs
    # Constructor must deliberately turn off training randomness and buffer edits.
    inputs['backbone'].train().requires_grad_(True)
    model = _build_candidate(FrozenFeatureAdapter,inputs)
    initial = {name:value.clone() for name,value in model.backbone.state_dict().items()}
    for mode in (False,True,True,False):
        assert model.train(mode) is model
        assert model.training == mode and model.adapter.training == mode
        assert not model.backbone.training and not model.backbone.dropout.training
        first = model(inputs['images'])[0]
        second = model(inputs['images'])[0]
        assert torch.equal(first,second)
        for name,value in model.backbone.state_dict().items():
            assert torch.equal(value,initial[name]),name
        assert all(not p.requires_grad for p in model.backbone.parameters())


def verify_followup_generated_pixels(FrozenFeatureAdapter):
    @settings(max_examples=8,deadline=None,derandomize=True)
    @given(st.lists(st.integers(-10,10),min_size=6,max_size=6))
    def trial(values):
        inputs = make_eval_data()['numbered_images'].inputs
        inputs['images'] = torch.tensor(values,dtype=torch.float64).reshape(1,1,2,3)
        model = _build_candidate(FrozenFeatureAdapter,inputs)
        expected = _literal_expected(inputs['images'],inputs['adapter_weight'],inputs['adapter_bias'],inputs['feature_from_image'])
        assert not compare_values(model(inputs['images']),expected,atol=1e-10,rtol=1e-10)
    trial()


def verify_production_strides_empty_and_dtype(FrozenFeatureAdapter):
    for name in ('strided_images','empty_batch','input_gradient'):
        case = make_eval_data()[name]
        originals = {key:value.detach().clone() for key,value in case.inputs.items() if isinstance(value,torch.Tensor)}
        model = _build_candidate(FrozenFeatureAdapter,case.inputs)
        state = {key:value.clone() for key,value in model.state_dict().items()}
        assert not case.comparison(model(case.inputs['images']),case.expected)
        for key,value in model.state_dict().items():
            assert torch.equal(value,state[key]),key
        for key,value in originals.items():
            assert torch.equal(case.inputs[key].detach(),value),key
    for dtype in (torch.float32,torch.float64):
        case = make_eval_data()['numbered_images']
        model = _build_candidate(FrozenFeatureAdapter,case.inputs).to(dtype=dtype)
        maps,metadata = model(case.inputs['images'].to(dtype=dtype))
        assert maps.dtype == dtype and metadata.dtype == dtype
        torch.testing.assert_close(maps,case.expected[0].to(dtype=dtype))
        torch.testing.assert_close(metadata,case.expected[1].to(dtype=dtype))


def verify_production_singular_affine(FrozenFeatureAdapter):
    for dtype in (torch.float32,torch.float64):
        inputs=make_eval_data()['singular_feature_affine'].inputs
        inputs['backbone']=inputs['backbone'].to(dtype=dtype)
        inputs['feature_from_image']=inputs['feature_from_image'].to(dtype=dtype)
        try:
            _build_candidate(FrozenFeatureAdapter,inputs)
        except ValueError:
            continue
        raise AssertionError('Singular feature_from_image must be rejected during construction')


def verify_production_reject_layout_and_metadata(FrozenFeatureAdapter):
    inputs = make_eval_data()['numbered_images'].inputs
    configuration = {name:inputs[name] for name in ('backbone','grid_hw','prefix_tokens','in_channels','out_channels','feature_from_image')}
    for name,value in [('grid_hw',(0,3)),('prefix_tokens',-1),('prefix_tokens',True),
        ('in_channels',0),('out_channels',True),('backbone',None),
        ('feature_from_image',torch.eye(2,dtype=torch.float64)),
        ('feature_from_image',torch.full((3,3),float('nan'),dtype=torch.float64)),
        ('feature_from_image',torch.zeros(3,3,dtype=torch.float64))]:
        try:
            FrozenFeatureAdapter(**dict(configuration,**{name:value}))
        except ValueError:
            continue
        raise AssertionError(f'invalid {name} must raise ValueError')
    for config_updates,image in [({'grid_hw':(3,3)},inputs['images']),
        ({'in_channels':3},inputs['images']),({},inputs['images'].float()),
        ({},torch.ones(1,2,3,dtype=torch.float64)),
        ({},torch.full((1,1,2,3),float('nan'),dtype=torch.float64))]:
        model = FrozenFeatureAdapter(**dict(configuration,**config_updates))
        try:
            model(image)
        except ValueError:
            continue
        raise AssertionError('invalid backbone output or image must raise ValueError')

# %%
from torch_judge import evaluate
if not evaluate('0112_frozen_feature_adapter_eval.py',FrozenFeatureAdapter):
    raise SystemExit(1)
