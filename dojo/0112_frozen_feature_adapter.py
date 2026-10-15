# %%
# 0112 · Freeze the weights, keep the learning path
# Baseline timebox: 30–45 minutes. No pretrained checkpoint is required.
# Build a trainable feature adapter around a supplied tiny token backbone.

# %%
import torch
from torch import nn
from torch_judge import check, get_eval_data, compare, replay

# %%
# Constructor: backbone, grid_hw, prefix_tokens, in_channels, out_channels,
# feature_from_image. Backbone returns [B,P+Hf*Wf,Cin], explicitly row-major.
# Forward(images [B,C,H,W]) returns (maps [B,Cout,Hf,Wf], transform [B,3,3]).
# Expose backbone and adapter; adapter is a trainable nn.Conv2d(Cin,Cout,1).
# Freeze backbone parameters and keep backbone in eval even after .train(True).
# Adapter train/eval follows the wrapper. Freezing does NOT detach images:
# preserve requested input gradients and adapter gradients. Calibration is a
# copied fixed buffer, moved with .to(); output metadata expands across batch.
# Core dtype float32/float64 on one device. Metadata is a finite affine [3,3]
# with last row [0,0,1] and an invertible 2x2 linear block;
# Hf/Wf/Cin/Cout positive integers, P nonnegative.
# Reject malformed metadata/output layout/dtype with ValueError. B may be zero.
# Backbone must honor eval state without buffer mutations; this wrapper cannot
# purify arbitrary custom side effects. Input/calibration/weights stay unchanged
# in forward. Constructor deliberately freezes/evaluates the supplied backbone.
eval_data = get_eval_data('0112_frozen_feature_adapter_eval.py')
case = eval_data['numbered_images']
case.inputs, case.expected

# %% [answer]
class FrozenFeatureAdapter(nn.Module):
    def __init__(self,backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image):
        super().__init__()
        pass  # Keep helper definitions and this complete class in the answer cell.

    def train(self,mode=True):
        pass

    def forward(self,images):
        pass

# %%
# Inspectable cases include a fresh supplied backbone and explicit shared
# adapter weights. This helper constructs YOUR class, not a reference class.
# It is exploration code and never runs during saved-candidate evaluation.
def run_case(images,backbone,grid_hw,prefix_tokens,in_channels,out_channels,
             feature_from_image,adapter_weight,adapter_bias):
    model = FrozenFeatureAdapter(backbone,grid_hw,prefix_tokens,in_channels,out_channels,feature_from_image)
    with torch.no_grad():
        model.adapter.weight.copy_(adapter_weight)
        model.adapter.bias.copy_(adapter_bias)
    return model(images)

# %%
actual = run_case(**case.inputs)
first_result = compare(case,actual)
first_result.passed, first_result.diagnostics, first_result.actual, first_result.expected

# %%
check('0112_frozen_feature_adapter_eval.py',stage='baseline')

# %%
# Follow-up: try train(True), eval(), train(True). Backbone dropout must remain
# off and its training_calls buffer must not change. Adapter training state
# follows the wrapper. Guess the gradient through input before running backward:
# toy backbone features are [2*x+1,3*x-2]; supplied adapter sums give slope 7.5.
case = get_eval_data('0112_frozen_feature_adapter_eval.py')['input_gradient']
configuration = {key:case.inputs[key] for key in ('backbone','grid_hw','prefix_tokens','in_channels','out_channels','feature_from_image')}
model = FrozenFeatureAdapter(**configuration)
with torch.no_grad():
    model.adapter.weight.copy_(case.inputs['adapter_weight'])
    model.adapter.bias.copy_(case.inputs['adapter_bias'])
model.train(True)
maps,metadata = model(case.inputs['images'])
maps.sum().backward()
case.inputs['images'].grad, model.adapter.weight.grad, model.backbone.gain.grad

# %%
check('0112_frozen_feature_adapter_eval.py',stage='followup')

# %%
# Edit/rerun the class cell, then run_case constructs new instances. Fresh replay
# also recreates fixture backbones/leaf tensors. Keep first_result as history.
results = replay('0112_frozen_feature_adapter_eval.py',run_case)
failures = {name:r for name,r in results.items() if not r.passed}
failures

# %%
check('0112_frozen_feature_adapter_eval.py',stage='production')

# %%
# Eval references: dense reconstruction + 1x1 convolution, equivalent token-side
# projection, and explicitly inference-only no_grad. The latter cannot qualify
# requested input/adapter gradients. Run explore_detached_features(case.inputs)
# on numbered_images to see identical forward values but missing input gradients.
# benchmark_adapters(inputs) compares ALL versions under inference no_grad;
# workload, dtype/device, error and timing variability are explicit. Its map-byte
# estimate is not allocator peak/backend workspace or training backward storage.
# Real-checkpoint boundary (optional): supply a wrapper with this exact token
# contract, declared prefix count/grid and preprocessing feature-coordinate map.
# Do not infer preprocessing, patch stride, crop, or receptive-field alignment
# from tensor shape. Geometry-guided BEV attention consumes the returned map.

# %%
# Finite does not imply usable calibration: a singular affine destroys a
# spatial axis and cannot satisfy the sampler contract. Inspect the named
# singular_feature_affine case and its expected ValueError before composition.
case=get_eval_data('0112_frozen_feature_adapter_eval.py')['singular_feature_affine']
case.inputs['feature_from_image'],case.expected_exception
