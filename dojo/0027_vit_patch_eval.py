"""ViT patch embedding: explicit patches, unfold, and shared-weight convolution."""
# %%
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from hypothesis import given, settings, strategies as st
from torch_judge import EvalCase, compare_values

# %%
def _initialize(module,img_size,patch_size,in_channels,embed_dim):
    if (any(isinstance(v,bool) or not isinstance(v,int) or v <= 0
            for v in (img_size,patch_size,in_channels,embed_dim)) or img_size % patch_size):
        raise ValueError('positive dimensions and divisible img_size required')
    module.patch_size = patch_size
    module.in_channels = in_channels
    module.num_patches = (img_size//patch_size)**2
    module.proj = nn.Linear(in_channels*patch_size*patch_size,embed_dim)


def _validate_image(module,x):
    if (not isinstance(x,torch.Tensor) or x.ndim != 4 or x.shape[1] != module.in_channels
            or x.shape[2] % module.patch_size or x.shape[3] % module.patch_size):
        raise ValueError('need BCHW with expected channels and patch-divisible H,W')
    if x.dtype != module.proj.weight.dtype or x.device != module.proj.weight.device:
        raise ValueError('image must share projection dtype/device; move the module explicitly')

# %%
# Reference 1: extract each patch with an ordinary slice, then project.
# Production qualified, including rectangular runtime and empty batch. Channels
# are ordered BEFORE patch rows/columns; patches themselves are row-major.
# Explicitly materializes flattened patches and incurs per-patch Python overhead.
class PatchEmbeddingLoop(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,embed_dim):
        super().__init__()
        _initialize(self,img_size,patch_size,in_channels,embed_dim)

    def forward(self,x):
        _validate_image(self,x)
        b,c,h,w = x.shape; p = self.patch_size
        if h == 0 or w == 0:
            return self.proj(x.reshape(b,0,c*p*p))
        patches = []
        for row in range(0,h,p):
            for col in range(0,w,p):
                patches.append(x[:,:,row:row+p,col:col+p].reshape(b,c*p*p))
        return self.proj(torch.stack(patches,dim=1))

# %%
# Reference 2: unfold spatial dimensions as views, then reorder all patches.
# Production qualified. The permuted patch tensor generally needs a copy for
# the final flattened [B,N,C*P*P] matrix; unfold does not make that copy disappear.
class PatchEmbeddingUnfold(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,embed_dim):
        super().__init__()
        _initialize(self,img_size,patch_size,in_channels,embed_dim)

    def forward(self,x):
        _validate_image(self,x)
        b,c,h,w = x.shape; p = self.patch_size
        if h == 0 or w == 0:
            return self.proj(x.reshape(b,0,c*p*p))
        patches = x.unfold(2,p,p).unfold(3,p,p)
        patches = patches.permute(0,2,3,1,4,5).reshape(b,(h//p)*(w//p),c*p*p)
        return self.proj(patches)

# %%
# Reference 3: reshape the SAME Linear weights into a stride-P convolution.
# Production qualified. Keeps .proj and its state_dict compatible with the other
# versions. No explicit patch matrix in Python; backend workspace is separate
# and may still materialize work internally. Floating reduction order can differ.
class PatchEmbeddingConv(nn.Module):
    def __init__(self,img_size,patch_size,in_channels,embed_dim):
        super().__init__()
        _initialize(self,img_size,patch_size,in_channels,embed_dim)

    def forward(self,x):
        _validate_image(self,x)
        b,c,h,w = x.shape; p = self.patch_size
        if h == 0 or w == 0:
            return self.proj(x.reshape(b,0,c*p*p))
        weight = self.proj.weight.reshape(self.proj.out_features,c,p,p)
        maps = F.conv2d(x,weight,self.proj.bias,stride=p)
        return maps.flatten(2).transpose(1,2)

# %%
PatchEmbedding = PatchEmbeddingUnfold

# %%
def benchmark_patch_embedding(shape=(2,3,32,48),patch_size=4,embed_dim=32,device='cpu',dtype=torch.float32):
    from torch_judge.measurement import measure
    generator = torch.Generator(device=device).manual_seed(27)
    x = torch.randn(shape,dtype=dtype,device=device,generator=generator)
    prototype = PatchEmbeddingLoop(patch_size,patch_size,shape[1],embed_dim).to(device=device,dtype=dtype)
    with torch.no_grad():
        prototype.proj.weight.copy_(torch.randn(prototype.proj.weight.shape,dtype=dtype,device=device,generator=generator)/10)
        prototype.proj.bias.zero_()
    reports = {}
    expected = prototype(x)
    for name,cls in [('loop',PatchEmbeddingLoop),('unfold',PatchEmbeddingUnfold),('conv',PatchEmbeddingConv)]:
        model = cls(patch_size,patch_size,shape[1],embed_dim).to(device=device,dtype=dtype)
        model.load_state_dict(prototype.state_dict())
        actual = model(x)
        reports[name] = {'max_error':(actual-expected).abs().max().item(),
            'explicit_patch_matrix_bytes':0 if name == 'conv' else x.numel()*x.element_size(),
            'storage_note':'Analytical Python patch matrix, not measured allocator peak or backend workspace.',
            'timing':measure(lambda:model(x),device=device,workload={'shape':shape,'patch_size':patch_size,
                'embed_dim':embed_dim,'dtype':str(dtype),'seed':27,'mode':'forward, autograd enabled'})}
    return reports

# %% [verifiers]
def _scalar_patch_expected(x,weight,bias,patch_size):
    # Independent scalar loops expose c -> patch-row -> patch-column ordering.
    b,c,h,w = x.shape; p = patch_size; e = weight.shape[0]
    output = x.new_empty((b,(h//p)*(w//p),e))
    for batch in range(b):
        for row in range(h//p):
            for col in range(w//p):
                values = [float(x[batch,channel,row*p+dy,col*p+dx])
                    for channel in range(c) for dy in range(p) for dx in range(p)]
                for channel_out in range(e):
                    output[batch,row*(w//p)+col,channel_out] = float(bias[channel_out])+sum(
                        value*float(weight[channel_out,index]) for index,value in enumerate(values))
    return output


def make_eval_data():
    weight = torch.arange(1,25,dtype=torch.float64).reshape(3,8)/8
    bias = torch.tensor([-1.,0.,1.],dtype=torch.float64)
    images = [('numbered_channels','baseline',torch.arange(32,dtype=torch.float64).reshape(1,2,4,4)),
        ('rectangular_runtime','followup',torch.arange(24,dtype=torch.float64).reshape(1,2,2,6)),
        ('transposed_image','production',torch.arange(48,dtype=torch.float64).reshape(1,2,6,4).transpose(2,3)),
        ('empty_batch','production',torch.empty(0,2,4,4,dtype=torch.float64))]
    cases = {}
    for name,stage,x in images:
        cases[name] = EvalCase(name=name,stage=stage,purpose='Use the supplied shared projection weights; inspect channel and patch order.',
            inputs={'x':x},expected=_scalar_patch_expected(x,weight,bias,2),
            comparison=partial(compare_values,atol=1e-10,rtol=1e-10),
            generation={'constructor':(4,2,2,3),'weight':weight.clone(),'bias':bias.clone()})
    cases['invalid_channels'] = EvalCase(name='invalid_channels',stage='production',
        purpose='Wrong runtime channel count is an API error.',inputs={'x':torch.zeros(1,3,4,4,dtype=torch.float64)},
        expected_exception=ValueError,generation={'constructor':(4,2,2,3),'weight':weight.clone(),'bias':bias.clone()})
    return cases


def _fixture_model(PatchEmbedding,case):
    model = PatchEmbedding(*case.generation['constructor']).to(device=case.inputs['x'].device,dtype=case.inputs['x'].dtype)
    with torch.no_grad():
        model.proj.weight.copy_(case.generation['weight'])
        model.proj.bias.copy_(case.generation['bias'])
    return model


def verify_baseline_shape_and_count(PatchEmbedding):
    model = PatchEmbedding(8, 2, 3, 5)
    assert isinstance(model, nn.Module) and isinstance(model.proj, nn.Linear)
    assert model.num_patches == 16 and model.proj.in_features == 12
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 16, 5)


def verify_followup_numbered_layout_and_gradients(PatchEmbedding):
    model = PatchEmbedding(4, 2, 2, 1).double()
    with torch.no_grad():
        model.proj.weight.copy_(torch.arange(1, 9, dtype=torch.float64).reshape(1, 8))
        model.proj.bias.zero_()
    x = torch.arange(32., dtype=torch.float64).reshape(1, 2, 4, 4).requires_grad_()
    expected = torch.tensor([sum((i + 1) * v for i, v in enumerate(values)) for values in
        ([0, 1, 4, 5, 16, 17, 20, 21], [2, 3, 6, 7, 18, 19, 22, 23],
         [8, 9, 12, 13, 24, 25, 28, 29], [10, 11, 14, 15, 26, 27, 30, 31])], dtype=torch.float64)
    assert torch.equal(model(x).flatten(), expected)
    model(x.transpose(2, 3)).sum().backward()
    assert x.grad is not None and model.proj.weight.grad is not None
    assert model(torch.randn(1, 2, 2, 6, dtype=torch.float64)).shape == (1, 3, 1)


def verify_production_validation(PatchEmbedding):
    for args in ((4, 0, 2, 1), (5, 2, 2, 1), (4, 2, 0, 1)):
        try:
            PatchEmbedding(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid patch configuration accepted")
    model = PatchEmbedding(4, 2, 2, 3)
    for x in (torch.randn(1, 3, 4, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 4, 5)):
        try:
            model(x)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    assert model(torch.empty(0, 2, 4, 4)).shape == (0, 4, 3)


def verify_baseline_inspectable_patches(PatchEmbedding):
    case = make_eval_data()['numbered_channels']
    errors = case.comparison(_fixture_model(PatchEmbedding,case)(**case.inputs),case.expected)
    assert not errors,'\n'.join(errors)


def verify_followup_generated_rectangles(PatchEmbedding):
    @settings(max_examples=8,deadline=None,derandomize=True)
    @given(st.integers(1,3),st.integers(1,3))
    def trial(h,w):
        case = make_eval_data()['numbered_channels']
        model = _fixture_model(PatchEmbedding,case)
        x = torch.arange(2*2*h*2*w*2,dtype=torch.float64).reshape(2,2,h*2,w*2)
        expected = _scalar_patch_expected(x,model.proj.weight.detach(),model.proj.bias.detach(),2)
        assert not compare_values(model(x),expected,atol=1e-10,rtol=1e-10)
    trial()


def verify_production_shared_weight_backward(PatchEmbedding):
    case = make_eval_data()['transposed_image']
    model = _fixture_model(PatchEmbedding,case)
    x = case.inputs['x'].detach().requires_grad_()
    original = x.detach().clone()
    before_weight = model.proj.weight.detach().clone()
    before_bias = model.proj.bias.detach().clone()
    y = model(x)
    assert not case.comparison(y,case.expected)
    assert y.dtype == x.dtype and y.device == x.device
    gx,gw,gb = torch.autograd.grad(y.sum(),(x,model.proj.weight,model.proj.bias))
    expected_x = torch.empty_like(x)
    expected_w = torch.zeros_like(gw)
    b,c,h,w = x.shape; p = 2
    for batch in range(b):
        for channel in range(c):
            for row in range(h):
                for col in range(w):
                    k = channel*p*p+(row%p)*p+col%p
                    expected_x[batch,channel,row,col] = before_weight[:,k].sum()
                    expected_w[:,k] += x.detach()[batch,channel,row,col]
    torch.testing.assert_close(gx,expected_x)
    torch.testing.assert_close(gw,expected_w)
    torch.testing.assert_close(gb,torch.full_like(gb,b*(h//p)*(w//p)))
    assert torch.equal(x.detach(),original)
    assert torch.equal(model.proj.weight.detach(),before_weight)
    assert torch.equal(model.proj.bias.detach(),before_bias)
    tiny = torch.arange(8,dtype=torch.float64).reshape(1,2,2,2).requires_grad_()
    assert torch.autograd.gradcheck(model,(tiny,))
    for dtype in (torch.float32,torch.float64):
        typed = PatchEmbedding(4,2,2,3).to(dtype=dtype)
        typed.load_state_dict(model.state_dict())
        actual = typed(original.to(dtype=dtype))
        assert actual.dtype == dtype
        torch.testing.assert_close(actual,case.expected.to(dtype=dtype),atol=1e-5,rtol=1e-5)

# %%
from torch_judge import evaluate
if not evaluate("0027_vit_patch_eval.py", PatchEmbedding):
    raise SystemExit(1)
