# Torch Dojo

Small PyTorch and NumPy coding exercises. The question and reference eval for every problem live together in [`dojo/`](dojo/). Each eval script owns its Python verifier functions; [`torch_judge/`](torch_judge/) contains only the reusable runner.

Try the [browser dojo](https://phi9t.github.io/torch_dojo/lab/) or use the native workflow below.

Explore [Dojo Fundamentals](https://phi9t.github.io/torch_dojo/fundamentals/), an interactive companion to NumPy workouts #125–131. Predict an outcome, change masks, temperature, kernels or coordinate frames, then implement the full tensor operation in the linked workout. Start with attention, contrastive loss, Conv2D and IoU; use Huber, voxels and ego frames as extensions.

Keep the [NumPy / PyTorch cheatsheet](docs/practice/numpy-torch-cheatsheet.md) nearby for formulas, axis patterns, API translations, edge cases and complexity.

For convolution, use the focused [dense Conv2D](docs/practice/conv2d-cheatsheet.md) and [submanifold Conv2D/Conv3D](docs/practice/submanifold-conv-cheatsheet.md) cheatsheets.

## Start GPU Jupyter

From this checkout, start the GPU Jupyter insula with [mise](https://mise.jdx.dev/):

```bash
mise run jupyter
```

The server runs in a background tmux session and prints an authenticated URL at `http://127.0.0.1:8899/`. Open that URL in your browser, or connect Emacs to the `torch-dojo-gpu` kernel. The checkout is writable inside the insula at `/workspace/torch_dojo`, and that path is on `PYTHONPATH`, so `torch_judge` imports from `dojo/` or any notebook directory without `pip install -e .` (the insula rootfs is read-only).

```bash
mise run jupyter:token  # Show the current token
mise run jupyter:logs   # Follow server output; Ctrl-C stops following
mise run jupyter:stop   # Clean up kernels and stop the server
```

On a new machine, build the CUDA rootfs once with `insula/rootfs/build.sh`; the launcher requires a selected rootfs, Python 3, tmux, bubblewrap, and an NVIDIA GPU with host drivers. If mise asks to trust this checkout, run `mise trust`. Without mise, use `insula/jupyter_rootfs/run_server.sh` directly. See [GPU Jupyter insulas](#gpu-jupyter-insulas) for the build and Emacs attachment workflow.

## Start practicing

Install once from the repository root in a Python environment with PyTorch and NumPy:

```bash
python -m pip install -e .
cd dojo
```

Open `0042_pixel_centers.py` (or any question script) in an editor with a Jupyter kernel. Work through its `# %%` sections: predict the probe, plan the tensor operations, fill in the function, try the example, then run `check(...)`. You can also run a question as a Python script after filling in the answer.

The practice loop is edit, check, understand the failure, and revise. The scripts keep ordinary top-level Python code so Emacs and Jupyter can run cells interactively. The learner's editable cell is marked `# %% [answer]`; add a `# %% [setup]` cell when your answer needs top-level constants or helper definitions during saved-file evaluation. No `main()` function is needed.

The neighboring `_eval.py` file holds the reference implementation and named `verify_*` functions. Some verifiers use Hypothesis to explore generated edge cases and shrink failures to small examples. Run the script from `dojo/` to check the reference, or pass `--candidate` to test the answer you wrote in the question script:

```bash
python 0042_pixel_centers_eval.py
python 0042_pixel_centers_eval.py --candidate
```

The runner loads top-level imports outside drill cells and tagged setup/answer cells from the saved question script, so its teaching probes do not run during `--candidate`. Interactive `check(...)` uses the current kernel definition. It loads the eval script's `# %% [verifiers]` cell and earlier imports without executing the reference implementation. Both commands exit nonzero when a check fails. From the repository root, `make eval` checks every reference implementation and `make test` checks the shared runner. The native verifier functions leave room for future formal verification experiments without adding such a dependency today.

The 131 workouts use three cumulative stages: `baseline`, `followup`, and `production`. Keep editing the same `# %% [answer]` cell as you advance. In a kernel, run `check("0006_multihead_attention_eval.py", stage="baseline")`, then select `followup` and `production` when ready. From the shell, `python 0006_multihead_attention_eval.py --candidate` checks baseline, and `--candidate --stage production` checks every stage. Running the eval script without `--candidate` checks all reference verifiers.

The [staged quality spec](docs/specs/staged-dojo-quality.md) defines the quality bar. A production stage sets a concrete kernel contract for that problem: relevant validation, numerical behavior, shapes, dtype/device semantics, memory, and reproducibility. It does not imply that every exercise needs a service or GPU benchmark.

For the softmax → k-means → geometry path, follow the [workout guide](docs/practice/softmax-to-geometry.md). Start with #2 and #59, including #59's short indexing-and-shapes workout; then practice gathering and accumulating repeated destinations in #61 before voxel pooling and geometry. Each step uses a tiny example → predict shapes/values → implement → adversarial tests. Hints stay separate from the reference solutions.

For the MoE path, start with [top-k routing](dojo/0028_moe.py), then work through [router variants](dojo/0095_router_variants.py) to [one training update](dojo/0102_tiny_moe_update.py) in number order. The [implementation survey](docs/research/moe-implementations-primary-survey.md) explains which model and training-system behavior motivated each original workout.

## Keep the patterns

The browser home page now has a motif library and a spaced-review queue. For native Jupyter or the rootfs, open [motif_review.ipynb](motif_review.ipynb). Sixteen short recall cards connect recurring ideas across workouts: predict, reveal, try a probe, and rate your recall. The [motif review guide](docs/practice/motif-review.md) explains the queue and transferring progress between environments.

## Explore the progressive pilots

Softmax (#2), paged attention (#93), and sparse MoE dispatch (#101) now expose named eval data and complete reference progressions. Retrieve `eval_data = get_eval_data("0002_softmax_eval.py")`, run your current answer on a case, retain `compare(case, actual)`, then use `replay` after an edit. The [interactive eval guide](docs/practice/interactive-eval.md) covers the Emacs/Jupyter loop, reference-version contracts, fresh gradients/modules, authoring cases, and opt-in measurements.

The [autonomy perception practice guide](docs/practice/autonomy-perception.md) connects multi-camera projection, feature sampling/fusion, sparse voxel pooling and collision-safe BEV, with point decoration as an optional branch. Each workout exposes inspectable eval data and complete reference progressions for the Emacs/Jupyter practice loop.

Before the voxel-pooling and BEV workouts, do the seven [gather/scatter fundamentals drills](docs/practice/gather-scatter-drills.md) (#114–120): collisions, group shares, flat addresses, compact and put back, group by key, pillar layout, and the backward of a gather.

For why #111 resizes ViT position embeddings and keeps prefix tokens fixed, see the [position grid background](docs/practice/position-grid.md).

## Problems

| # | Problem | Question | Reference eval |
|---:|---|---|---|
| 1 | Implement ReLU | [0001_relu.py](dojo/0001_relu.py) | [0001_relu_eval.py](dojo/0001_relu_eval.py) |
| 2 | Implement Softmax | [0002_softmax.py](dojo/0002_softmax.py) | [0002_softmax_eval.py](dojo/0002_softmax_eval.py) |
| 3 | Simple Linear Layer | [0003_linear.py](dojo/0003_linear.py) | [0003_linear_eval.py](dojo/0003_linear_eval.py) |
| 4 | Implement LayerNorm | [0004_layernorm.py](dojo/0004_layernorm.py) | [0004_layernorm_eval.py](dojo/0004_layernorm_eval.py) |
| 5 | Softmax Attention | [0005_attention.py](dojo/0005_attention.py) | [0005_attention_eval.py](dojo/0005_attention_eval.py) |
| 6 | Multi-Head Attention | [0006_multihead_attention.py](dojo/0006_multihead_attention.py) | [0006_multihead_attention_eval.py](dojo/0006_multihead_attention_eval.py) |
| 7 | Implement BatchNorm | [0007_batchnorm.py](dojo/0007_batchnorm.py) | [0007_batchnorm_eval.py](dojo/0007_batchnorm_eval.py) |
| 8 | Implement RMSNorm | [0008_rmsnorm.py](dojo/0008_rmsnorm.py) | [0008_rmsnorm_eval.py](dojo/0008_rmsnorm_eval.py) |
| 9 | Causal Self-Attention | [0009_causal_attention.py](dojo/0009_causal_attention.py) | [0009_causal_attention_eval.py](dojo/0009_causal_attention_eval.py) |
| 10 | Grouped Query Attention | [0010_gqa.py](dojo/0010_gqa.py) | [0010_gqa_eval.py](dojo/0010_gqa_eval.py) |
| 11 | Sliding Window Attention | [0011_sliding_window.py](dojo/0011_sliding_window.py) | [0011_sliding_window_eval.py](dojo/0011_sliding_window_eval.py) |
| 12 | Linear Self-Attention | [0012_linear_attention.py](dojo/0012_linear_attention.py) | [0012_linear_attention_eval.py](dojo/0012_linear_attention_eval.py) |
| 13 | GPT-2 Transformer Block | [0013_gpt2_block.py](dojo/0013_gpt2_block.py) | [0013_gpt2_block_eval.py](dojo/0013_gpt2_block_eval.py) |
| 14 | KV Cache Attention | [0014_kv_cache.py](dojo/0014_kv_cache.py) | [0014_kv_cache_eval.py](dojo/0014_kv_cache_eval.py) |
| 15 | SwiGLU MLP | [0015_mlp.py](dojo/0015_mlp.py) | [0015_mlp_eval.py](dojo/0015_mlp_eval.py) |
| 16 | Cross-Entropy Loss | [0016_cross_entropy.py](dojo/0016_cross_entropy.py) | [0016_cross_entropy_eval.py](dojo/0016_cross_entropy_eval.py) |
| 17 | Implement Dropout | [0017_dropout.py](dojo/0017_dropout.py) | [0017_dropout_eval.py](dojo/0017_dropout_eval.py) |
| 18 | Embedding Layer | [0018_embedding.py](dojo/0018_embedding.py) | [0018_embedding_eval.py](dojo/0018_embedding_eval.py) |
| 19 | GELU Activation | [0019_gelu.py](dojo/0019_gelu.py) | [0019_gelu_eval.py](dojo/0019_gelu_eval.py) |
| 20 | Kaiming Initialization | [0020_weight_init.py](dojo/0020_weight_init.py) | [0020_weight_init_eval.py](dojo/0020_weight_init_eval.py) |
| 21 | Gradient Norm Clipping | [0021_gradient_clipping.py](dojo/0021_gradient_clipping.py) | [0021_gradient_clipping_eval.py](dojo/0021_gradient_clipping_eval.py) |
| 22 | 2D Convolution | [0022_conv2d.py](dojo/0022_conv2d.py) | [0022_conv2d_eval.py](dojo/0022_conv2d_eval.py) |
| 23 | Multi-Head Cross-Attention | [0023_cross_attention.py](dojo/0023_cross_attention.py) | [0023_cross_attention_eval.py](dojo/0023_cross_attention_eval.py) |
| 24 | Rotary Position Embedding (RoPE) | [0024_rope.py](dojo/0024_rope.py) | [0024_rope_eval.py](dojo/0024_rope_eval.py) |
| 25 | Flash Attention (Tiled) | [0025_flash_attention.py](dojo/0025_flash_attention.py) | [0025_flash_attention_eval.py](dojo/0025_flash_attention_eval.py) |
| 26 | LoRA (Low-Rank Adaptation) | [0026_lora.py](dojo/0026_lora.py) | [0026_lora_eval.py](dojo/0026_lora_eval.py) |
| 27 | ViT Patch Embedding | [0027_vit_patch.py](dojo/0027_vit_patch.py) | [0027_vit_patch_eval.py](dojo/0027_vit_patch_eval.py) |
| 28 | Mixture of Experts (MoE) | [0028_moe.py](dojo/0028_moe.py) | [0028_moe_eval.py](dojo/0028_moe_eval.py) |
| 29 | Adam Optimizer | [0029_adam.py](dojo/0029_adam.py) | [0029_adam_eval.py](dojo/0029_adam_eval.py) |
| 30 | Cosine LR Scheduler with Warmup | [0030_cosine_lr.py](dojo/0030_cosine_lr.py) | [0030_cosine_lr_eval.py](dojo/0030_cosine_lr_eval.py) |
| 31 | Gradient Accumulation | [0031_gradient_accumulation.py](dojo/0031_gradient_accumulation.py) | [0031_gradient_accumulation_eval.py](dojo/0031_gradient_accumulation_eval.py) |
| 32 | Top-k / Top-p Sampling | [0032_topk_sampling.py](dojo/0032_topk_sampling.py) | [0032_topk_sampling_eval.py](dojo/0032_topk_sampling_eval.py) |
| 33 | Beam Search Decoding | [0033_beam_search.py](dojo/0033_beam_search.py) | [0033_beam_search_eval.py](dojo/0033_beam_search_eval.py) |
| 34 | Speculative Decoding | [0034_speculative_decoding.py](dojo/0034_speculative_decoding.py) | [0034_speculative_decoding_eval.py](dojo/0034_speculative_decoding_eval.py) |
| 35 | Byte-Pair Encoding (BPE) | [0035_bpe.py](dojo/0035_bpe.py) | [0035_bpe_eval.py](dojo/0035_bpe_eval.py) |
| 36 | INT8 Quantized Linear | [0036_int8_quantization.py](dojo/0036_int8_quantization.py) | [0036_int8_quantization_eval.py](dojo/0036_int8_quantization_eval.py) |
| 37 | DPO (Direct Preference Optimization) Loss | [0037_dpo_loss.py](dojo/0037_dpo_loss.py) | [0037_dpo_loss_eval.py](dojo/0037_dpo_loss_eval.py) |
| 38 | GRPO (Group Relative Policy Optimization) Loss | [0038_grpo_loss.py](dojo/0038_grpo_loss.py) | [0038_grpo_loss_eval.py](dojo/0038_grpo_loss_eval.py) |
| 39 | PPO (Proximal Policy Optimization) Clipped Loss | [0039_ppo_loss.py](dojo/0039_ppo_loss.py) | [0039_ppo_loss_eval.py](dojo/0039_ppo_loss_eval.py) |
| 40 | Linear Regression | [0040_linear_regression.py](dojo/0040_linear_regression.py) | [0040_linear_regression_eval.py](dojo/0040_linear_regression_eval.py) |
| 41 | OPD (On-Policy Distillation) Loss | [0041_opd_loss.py](dojo/0041_opd_loss.py) | [0041_opd_loss_eval.py](dojo/0041_opd_loss_eval.py) |
| 42 | Resize Pixel Centers | [0042_pixel_centers.py](dojo/0042_pixel_centers.py) | [0042_pixel_centers_eval.py](dojo/0042_pixel_centers_eval.py) |
| 43 | Masked Semantic IoU | [0043_semantic_iou.py](dojo/0043_semantic_iou.py) | [0043_semantic_iou_eval.py](dojo/0043_semantic_iou_eval.py) |
| 44 | Pinhole Camera Projection | [0044_camera_project.py](dojo/0044_camera_project.py) | [0044_camera_project_eval.py](dojo/0044_camera_project_eval.py) |
| 45 | Rigid Point Alignment | [0045_rigid_align.py](dojo/0045_rigid_align.py) | [0045_rigid_align_eval.py](dojo/0045_rigid_align_eval.py) |
| 46 | BEV Grid Indices | [0046_bev_indices.py](dojo/0046_bev_indices.py) | [0046_bev_indices_eval.py](dojo/0046_bev_indices_eval.py) |
| 47 | Encode 3D Detection Boxes | [0047_encode_boxes.py](dojo/0047_encode_boxes.py) | [0047_encode_boxes_eval.py](dojo/0047_encode_boxes_eval.py) |
| 48 | Finite Difference Gradient | [0048_finite_difference_grad.py](dojo/0048_finite_difference_grad.py) | [0048_finite_difference_grad_eval.py](dojo/0048_finite_difference_grad_eval.py) |
| 49 | Linear Vector-Jacobian Product | [0049_linear_vjp.py](dojo/0049_linear_vjp.py) | [0049_linear_vjp_eval.py](dojo/0049_linear_vjp_eval.py) |
| 50 | Backward Through a Tiny Graph | [0050_backward_graph.py](dojo/0050_backward_graph.py) | [0050_backward_graph_eval.py](dojo/0050_backward_graph_eval.py) |
| 51 | Camera Unprojection | [0051_camera_unproject.py](dojo/0051_camera_unproject.py) | [0051_camera_unproject_eval.py](dojo/0051_camera_unproject_eval.py) |
| 52 | Two-View Triangulation | [0052_triangulate_point.py](dojo/0052_triangulate_point.py) | [0052_triangulate_point_eval.py](dojo/0052_triangulate_point_eval.py) |
| 53 | Invert a Rigid Pose | [0053_pose_inverse.py](dojo/0053_pose_inverse.py) | [0053_pose_inverse_eval.py](dojo/0053_pose_inverse_eval.py) |
| 54 | LiDAR Polar Coordinates | [0054_polar_to_cartesian.py](dojo/0054_polar_to_cartesian.py) | [0054_polar_to_cartesian_eval.py](dojo/0054_polar_to_cartesian_eval.py) |
| 55 | Constant-Velocity Rollout | [0055_constant_velocity_rollout.py](dojo/0055_constant_velocity_rollout.py) | [0055_constant_velocity_rollout_eval.py](dojo/0055_constant_velocity_rollout_eval.py) |
| 56 | Minimum Displacement Errors | [0056_min_displacement_errors.py](dojo/0056_min_displacement_errors.py) | [0056_min_displacement_errors_eval.py](dojo/0056_min_displacement_errors_eval.py) |
| 57 | Joint Trajectory Mode | [0057_joint_trajectory_mode.py](dojo/0057_joint_trajectory_mode.py) | [0057_joint_trajectory_mode_eval.py](dojo/0057_joint_trajectory_mode_eval.py) |
| 58 | First Sampled Collision | [0058_first_collision_step.py](dojo/0058_first_collision_step.py) | [0058_first_collision_step_eval.py](dojo/0058_first_collision_step_eval.py) |

| 59 | Stable Softmax Cross-Entropy | [0059_stable_cross_entropy.py](dojo/0059_stable_cross_entropy.py) | [0059_stable_cross_entropy_eval.py](dojo/0059_stable_cross_entropy_eval.py) |
| 60 | Linear Classifier Step | [0060_classifier_step.py](dojo/0060_classifier_step.py) | [0060_classifier_step_eval.py](dojo/0060_classifier_step_eval.py) |
| 61 | K-Means | [0061_kmeans.py](dojo/0061_kmeans.py) | [0061_kmeans_eval.py](dojo/0061_kmeans_eval.py) |
| 62 | Pairwise Box IoU | [0062_pairwise_iou.py](dojo/0062_pairwise_iou.py) | [0062_pairwise_iou_eval.py](dojo/0062_pairwise_iou_eval.py) |
| 63 | Uniform Oriented-Box Sampling | [0063_oriented_box_sampling.py](dojo/0063_oriented_box_sampling.py) | [0063_oriented_box_sampling_eval.py](dojo/0063_oriented_box_sampling_eval.py) |
| 64 | Point in Rotated Box | [0064_point_in_rotated_box.py](dojo/0064_point_in_rotated_box.py) | [0064_point_in_rotated_box_eval.py](dojo/0064_point_in_rotated_box_eval.py) |
| 65 | Temporal Ego Coordinates | [0065_temporal_transforms.py](dojo/0065_temporal_transforms.py) | [0065_temporal_transforms_eval.py](dojo/0065_temporal_transforms_eval.py) |
| 66 | Voxelization and Mean Pooling | [0066_voxel_pool.py](dojo/0066_voxel_pool.py) | [0066_voxel_pool_eval.py](dojo/0066_voxel_pool_eval.py) |
| 67 | Run-Length Random Access | [0067_rle_index.py](dojo/0067_rle_index.py) | [0067_rle_index_eval.py](dojo/0067_rle_index_eval.py) |
| 68 | Union of Half-Open Intervals | [0068_interval_union.py](dojo/0068_interval_union.py) | [0068_interval_union_eval.py](dojo/0068_interval_union_eval.py) |
| 69 | Multiclass Calibration Metrics | [0069_calibration_metrics.py](dojo/0069_calibration_metrics.py) | [0069_calibration_metrics_eval.py](dojo/0069_calibration_metrics_eval.py) |

| 70 | Anchor to GT Matching | [0070_anchor_match.py](dojo/0070_anchor_match.py) | [0070_anchor_match_eval.py](dojo/0070_anchor_match_eval.py) |
| 71 | Grid Connected Components | [0071_grid_components.py](dojo/0071_grid_components.py) | [0071_grid_components_eval.py](dojo/0071_grid_components_eval.py) |
| 72 | CSV Rows with Quoting | [0072_csv_rows.py](dojo/0072_csv_rows.py) | [0072_csv_rows_eval.py](dojo/0072_csv_rows_eval.py) |
| 73 | Paired Trip Bootstrap | [0073_paired_trip_bootstrap.py](dojo/0073_paired_trip_bootstrap.py) | [0073_paired_trip_bootstrap_eval.py](dojo/0073_paired_trip_bootstrap_eval.py) |
| 74 | Detection Average Precision | [0074_detection_ap.py](dojo/0074_detection_ap.py) | [0074_detection_ap_eval.py](dojo/0074_detection_ap_eval.py) |
| 75 | K-Means++ Initialization | [0075_kmeans_plus_plus.py](dojo/0075_kmeans_plus_plus.py) | [0075_kmeans_plus_plus_eval.py](dojo/0075_kmeans_plus_plus_eval.py) |
| 76 | Max-Pool Coordinates | [0076_max_pool_coordinates.py](dojo/0076_max_pool_coordinates.py) | [0076_max_pool_coordinates_eval.py](dojo/0076_max_pool_coordinates_eval.py) |
| 77 | Sparse COO | [0077_sparse_coo.py](dojo/0077_sparse_coo.py) | [0077_sparse_coo_eval.py](dojo/0077_sparse_coo_eval.py) |
| 78 | Streaming CSV Consumer | [0078_csv_consumer.py](dojo/0078_csv_consumer.py) | [0078_csv_consumer_eval.py](dojo/0078_csv_consumer_eval.py) |
| 79 | Expression Parser | [0079_expression_parser.py](dojo/0079_expression_parser.py) | [0079_expression_parser_eval.py](dojo/0079_expression_parser_eval.py) |
| 80 | Coaster Scoring | [0080_coaster_scoring.py](dojo/0080_coaster_scoring.py) | [0080_coaster_scoring_eval.py](dojo/0080_coaster_scoring_eval.py) |
| 81 | Gated Residual Module | [0081_gated_residual.py](dojo/0081_gated_residual.py) | [0081_gated_residual_eval.py](dojo/0081_gated_residual_eval.py) |
| 82 | Tabular Policy Evaluation | [0082_policy_evaluation.py](dojo/0082_policy_evaluation.py) | [0082_policy_evaluation_eval.py](dojo/0082_policy_evaluation_eval.py) |
| 83 | Sequential Q-Learning | [0083_q_learning.py](dojo/0083_q_learning.py) | [0083_q_learning_eval.py](dojo/0083_q_learning_eval.py) |
| 84 | Episodic REINFORCE Loss | [0084_reinforce_loss.py](dojo/0084_reinforce_loss.py) | [0084_reinforce_loss_eval.py](dojo/0084_reinforce_loss_eval.py) |
| 85 | DPO from Completion Logits | [0085_dpo_token_pairs.py](dojo/0085_dpo_token_pairs.py) | [0085_dpo_token_pairs_eval.py](dojo/0085_dpo_token_pairs_eval.py) |
| 86 | PPO Generalized Advantage Estimation | [0086_ppo_gae.py](dojo/0086_ppo_gae.py) | [0086_ppo_gae_eval.py](dojo/0086_ppo_gae_eval.py) |
| 87 | Token-Level GRPO Objective | [0087_grpo_tokens.py](dojo/0087_grpo_tokens.py) | [0087_grpo_tokens_eval.py](dojo/0087_grpo_tokens_eval.py) |
| 88 | Student-Rollout On-Policy Distillation | [0088_opd_rollout.py](dojo/0088_opd_rollout.py) | [0088_opd_rollout_eval.py](dojo/0088_opd_rollout_eval.py) |
| 89 | Pairwise Reward-Model Loss | [0089_pairwise_reward.py](dojo/0089_pairwise_reward.py) | [0089_pairwise_reward_eval.py](dojo/0089_pairwise_reward_eval.py) |
| 90 | Grouped RLOO Advantages and Policy Loss | [0090_rloo_advantages.py](dojo/0090_rloo_advantages.py) | [0090_rloo_advantages_eval.py](dojo/0090_rloo_advantages_eval.py) |
| 91 | On-Policy MDP Trajectories | [0091_on_policy_trajectories.py](dojo/0091_on_policy_trajectories.py) | [0091_on_policy_trajectories_eval.py](dojo/0091_on_policy_trajectories_eval.py) |
| 92 | One DDPM Reverse Step | [0092_ddpm_reverse_step.py](dojo/0092_ddpm_reverse_step.py) | [0092_ddpm_reverse_step_eval.py](dojo/0092_ddpm_reverse_step_eval.py) |
| 93 | Paged KV Cache Attention | [0093_paged_attention.py](dojo/0093_paged_attention.py) | [0093_paged_attention_eval.py](dojo/0093_paged_attention_eval.py) |
| 94 | Binary Probability Calibration Fitting | [0094_binary_calibration.py](dojo/0094_binary_calibration.py) | [0094_binary_calibration_eval.py](dojo/0094_binary_calibration_eval.py) |
| 95 | Router Variants | [0095_router_variants.py](dojo/0095_router_variants.py) | [0095_router_variants_eval.py](dojo/0095_router_variants_eval.py) |
| 96 | Shared Branch | [0096_shared_branch.py](dojo/0096_shared_branch.py) | [0096_shared_branch_eval.py](dojo/0096_shared_branch_eval.py) |
| 97 | Route Ledger | [0097_route_ledger.py](dojo/0097_route_ledger.py) | [0097_route_ledger_eval.py](dojo/0097_route_ledger_eval.py) |
| 98 | Load Balance | [0098_load_balance.py](dojo/0098_load_balance.py) | [0098_load_balance_eval.py](dojo/0098_load_balance_eval.py) |
| 99 | Bias Feedback | [0099_bias_feedback.py](dojo/0099_bias_feedback.py) | [0099_bias_feedback_eval.py](dojo/0099_bias_feedback_eval.py) |
| 100 | Capacity Admission | [0100_capacity_admission.py](dojo/0100_capacity_admission.py) | [0100_capacity_admission_eval.py](dojo/0100_capacity_admission_eval.py) |
| 101 | Sparse Dispatch | [0101_sparse_dispatch.py](dojo/0101_sparse_dispatch.py) | [0101_sparse_dispatch_eval.py](dojo/0101_sparse_dispatch_eval.py) |
| 102 | Tiny MoE Training Update | [0102_tiny_moe_update.py](dojo/0102_tiny_moe_update.py) | [0102_tiny_moe_update_eval.py](dojo/0102_tiny_moe_update_eval.py) |
| 121 | Submanifold Sparse Convolution | [0121_submanifold_conv.py](dojo/0121_submanifold_conv.py) | [0121_submanifold_conv_eval.py](dojo/0121_submanifold_conv_eval.py) |
| 122 | Flash Attention with Recomputing Backward | [0122_flash_attention_backward.py](dojo/0122_flash_attention_backward.py) | [0122_flash_attention_backward_eval.py](dojo/0122_flash_attention_backward_eval.py) |
| 123 | Split-KV Attention Merge | [0123_split_kv_attention.py](dojo/0123_split_kv_attention.py) | [0123_split_kv_attention_eval.py](dojo/0123_split_kv_attention_eval.py) |
| 124 | Packed Variable-Length Attention | [0124_packed_attention.py](dojo/0124_packed_attention.py) | [0124_packed_attention_eval.py](dojo/0124_packed_attention_eval.py) |
| 125 | Masked Self-Attention (NumPy) | [0125_masked_self_attention.py](dojo/0125_masked_self_attention.py) | [0125_masked_self_attention_eval.py](dojo/0125_masked_self_attention_eval.py) |
| 126 | Symmetric Contrastive Loss (NumPy) | [0126_contrastive_loss.py](dojo/0126_contrastive_loss.py) | [0126_contrastive_loss_eval.py](dojo/0126_contrastive_loss_eval.py) |
| 127 | Conv2D from Scratch (NumPy) | [0127_conv2d_scratch.py](dojo/0127_conv2d_scratch.py) | [0127_conv2d_scratch_eval.py](dojo/0127_conv2d_scratch_eval.py) |
| 128 | Pairwise Box IoU Grid (NumPy) | [0128_box_iou_grid.py](dojo/0128_box_iou_grid.py) | [0128_box_iou_grid_eval.py](dojo/0128_box_iou_grid_eval.py) |
| 129 | Masked Huber Loss and Gradient (NumPy) | [0129_masked_huber_grad.py](dojo/0129_masked_huber_grad.py) | [0129_masked_huber_grad_eval.py](dojo/0129_masked_huber_grad_eval.py) |
| 130 | Point-to-Voxel Feature Mean (NumPy) | [0130_voxel_feature_mean.py](dojo/0130_voxel_feature_mean.py) | [0130_voxel_feature_mean_eval.py](dojo/0130_voxel_feature_mean_eval.py) |
| 131 | History Points to Ego Frame (NumPy) | [0131_history_to_ego_frame.py](dojo/0131_history_to_ego_frame.py) | [0131_history_to_ego_frame_eval.py](dojo/0131_history_to_ego_frame_eval.py) |

Exercises 125–131 are a seven-problem NumPy practice set (masked attention, contrastive loss, Conv2D, box IoU, Huber with a manual gradient, voxel averaging, ego-frame transforms). Do them in order; each question states its hand-checkable example and a time budget, and each eval carries baseline, follow-up and production checks.

Exercise 92 introduces one stochastic denoising step after the earlier tensor and probability exercises. It is an original dojo lesson motivated by [Deep-ML's DDPM reverse sampling problem](https://www.deep-ml.com/problems/396); its one-based index, injected randomness, and validation contract are authored here.

The six Sureal adaptations are #42–47: pixel-center resizing and semantic IoU (vision), camera projection and rigid point alignment (3D), plus BEV indexing and box encoding (autonomy). Each question names its source module. All six use small, deterministic PyTorch examples and require no dataset or GPU.

The tnsr-inspired autograd exercises are #48–50: verify gradients numerically, compute a linear layer's backward rule, and accumulate gradients across a small computation graph. Their question scripts name the corresponding tnsr source files.

The additional Sureal geometry exercises are #51–54: undo camera projection, recover a point from two views, invert a rigid pose, and map LiDAR polar measurements into 3D points.

The motion exercises are #55–58: roll out a constant-velocity forecast, score multimodal displacement, select one shared mode for multiple actors, and check sampled circular-footprint collisions. The scoring and collision exercises are simplified teaching models; they use explicit autonomy teaching contracts rather than reproducing benchmark metrics.

The RL exercises are #82–84: evaluate a fixed policy, replay Q-learning updates, and differentiate a reward-to-go policy loss. They are original dojo problems prompted by topics in the Deep-ML survey.

The modern post-training exercises are #85–88: build masked DPO pair scores from token logits, estimate PPO advantages, compute a completion-normalized GRPO objective, and distill a teacher on student-generated prefixes. The [modern RL source map](docs/research/modern-rl-dojo-sources.md) explains the paper-level concepts and the original dojo contracts.

Exercise #89 trains scalar reward-model scores from chosen and rejected responses, with a margin and strict preference accuracy. It is an original dojo exercise motivated by the [Deep-ML reward-model loss topic](https://www.deep-ml.com/problems/484); unlike #85, its objective trains the reward model rather than a preference policy.

Exercise #90 extends the REINFORCE and GRPO lessons with prompt-grouped leave-one-out baselines and fixed KL-shaped rollout rewards. Its original task was prompted by the [Deep-ML RLOO topic](https://www.deep-ml.com/problems/1266).

Exercise #93 extends dense attention and KV caching with logical-to-physical page mapping and blockwise stable softmax. Its original task was prompted by the [Deep-ML PagedAttention topic](https://www.deep-ml.com/problems/492).

Exercises #122–124 are authored production-attention follow-ups to #25 and #14: a custom backward that saves only the logsumexp and recomputes tiles, a flash-decoding merge of per-split outputs, and packed `cu_seqlens` attention without padding. Their memory contracts are checked by recording the largest tensor any operation produces, so a dense implementation with correct gradients still fails.

Exercise #94 extends the calibration-metrics lesson by fitting Platt or isotonic maps on calibration scores and predicting on separate queries. Its original task was prompted by the [Deep-ML calibration topic](https://www.deep-ml.com/problems/1466).

## Practice bank

The [103-drill practice bank](practice_bank/index.md) includes a [self-contained fieldbook](practice_bank/handbook.html), ten-session queue, technical references and a blank practice log. New NumPy questions #59–74 cover the first diagnostic, clustering, geometry, conventional coding and evaluation. The source ledger links primary API and research references.

The [Deep-ML question survey](docs/research/deep-ml-question-survey.md) links foundational practice questions and identifies topics already covered here. The [live catalog follow-up](docs/research/deep-ml-live-catalog-2026-10.md) adds newer ML/AI questions, including post-training, vision, world models, inference, and evaluation.

## Layout

```text
dojo/             guided questions and reference evals only
torch_judge/      generic runner; no per-problem tasks
pyproject.toml    installable judge dependency
```

To add a problem, create a four-digit `dojo/<num>_<problem>.py` question and its `dojo/<num>_<problem>_eval.py` reference. Mark exactly one answer cell with `# %% [answer]`; optional `# %% [setup]` cells hold top-level code needed by a saved answer. Mark exactly one `# %% [verifiers]` cell in the eval script, containing any shared helpers and named `verify_*` functions. Close it with another `# %%` before the reference runner. Each verifier takes the learner's implementation as its sole parameter; all verifiers use the same parameter name.

Every answer signature is fully annotated: each parameter and the return value carry a type (`torch.Tensor`, `int`, `torch.Tensor | None`, ...), and the docstring or the contract comment above the answer states the shapes and dtypes (`Q [B,Tq,D]`, ...). Immediately before the answer cell, put a plain `# %%` cell of inputs that conform to that signature, one global per parameter under the parameter's own name (`Q, K, V, allowed = ...`). Make them small and deterministic (seeded). A learner can then evaluate the globals, step into the function body line by line, and call `answer(Q, K, V)` in a kernel. Do not add `# %% [setup]` to this cell; the saved-file runner must not need it.

For a staged exercise, name its checks `verify_baseline_*`, `verify_followup_*`, and `verify_production_*`. A later stage includes earlier checks. Keep the stage contract and a `check(..., stage="...")` call in separate top-level teaching cells; the answer remains one editable cell.

## GPU Jupyter insulas

Build and select the Torch Dojo CUDA rootfs once:

```bash
insula/rootfs/build.sh
```

Start the token-protected Jupyter Server in the Jupyter insula, then open terminal Doom Emacs in its separate insula:

```bash
mise run jupyter
insula/emacs/run.sh
```

The server listens at `http://127.0.0.1:8899/`. Its fresh token is in `target/insula/shared/token` (mode `0600`); the launch command prints the authenticated URL. In Emacs, use `M-x jupyter-run-server-repl`, supply that URL, and select the `torch-dojo-gpu` kernel. Both insulas edit this checkout at `/workspace/torch_dojo`; only the Jupyter insula sees NVIDIA devices. Run `insula/jupyter_rootfs/verify.sh` for a CUDA kernel and mutable-workspace check, `insula/jupyter_rootfs/verify_cuda.sh` for the integrated CUDA toolchain checks (nvcc, NVRTC, kernel launches on torch tensors, `__cuda_array_interface__`/DLPack/numpy interop), and `insula/emacs/verify_attach.sh` for a batch attachment check. Stop with `mise run jupyter:stop`.

The rootfs and each role's writable home/runtime state are under ignored `target/insula/`. The existing Docker workflow is separate.

### Step through a GPU Jupyter scenario in Emacs

Run `insula/emacs/run_scenario.sh` from the host checkout. It opens a run-specific copy of `insula/scenarios/default.org` in terminal Doom Emacs and starts Jupyter Server if needed. Both insulas see the checkout at `/workspace/torch_dojo`; the server remains at `http://127.0.0.1:8899/`, with its private token in `target/insula/shared/token`. The scenario reads the token itself and never displays its value.

The Org buffer is the debugger. Press `n` once for each step: **setup → connect → each Python cell → final → cleanup**. Python source blocks are editable before their step. The result appears directly beneath each block. The bundled context result shows rootfs Python and the kernel's working directory; the CUDA block synchronizes and shows `device=cuda:0 sum=10.0`; the workspace block writes `workspace.txt` under the run's `target/insula/scenario/` directory. The host checks that file after Emacs exits.

Press `p` to inspect the previous completed step without executing it. Press `r` to keep your cell edits, clear results, shut down the owned kernel, and restart at setup. Press `q` to clean up the owned kernel and exit. A failed cell pauses the scenario; a timeout is shown as **outcome unknown** and is never resent automatically. The run-specific Org copy and results remain under `target/insula/scenario/`. The server remains available after the cleanup checkpoint; on exit the wrapper stops it only if it started it and no other kernels remain.

To watch both insulas in separate tmux windows, run these from a terminal in the host checkout:

```bash
tmux new-session -d -s torch-dojo-scenario -n emacs -c "$PWD" 'insula/emacs/run_scenario.sh'
tmux new-window -t torch-dojo-scenario -n jupyter -c "$PWD" 'tail -n 40 -F target/insula/shared/server.log'
tmux select-window -t torch-dojo-scenario:emacs
tmux attach -t torch-dojo-scenario
```

Use the `emacs` window to press `n` and the `jupyter` window to watch the separate server's log. Run `insula/emacs/verify_scenario.sh` for the bundled cells in batch with CUDA, host workspace, kernel ID, and cleanup checks. `insula/emacs/verify_attach.sh` remains the shorter connection regression check.

### Full-screen scenario TUI

The separate terminal interface shows Setup, Connect, the selected Org file's Python cells, Final, and Cleanup. Install Textual on the host (the CUDA rootfs stays as it is), then start it:

```bash
python3 -m venv target/insula/tui-venv
target/insula/tui-venv/bin/python -m pip install 'textual>=6,<7'
insula/scenarios/run_tui.sh [path/to/file.org]
```

The top strip shows the execution path, the live Emacs PID, and the kernel ID reported by `emacs-jupyter`. The two lower panes follow output from the Emacs worker in its insula and Jupyter Server in its separate insula; the Jupyter pane starts at the current server session and redacts tokens. Use a terminal of at least 80 columns; a wider terminal gives the log panes more room.

With no path, the bundled `insula/scenarios/default.org` runs. A chosen path must name a readable `.org` file inside this checkout. The file needs at least one Python source block; non-Python blocks are skipped. Named and unnamed blocks under nested headings run in document order. Display-oriented `:results` headers may remain; execution-changing headers such as `:var`, `:session`, `:noweb`, `:dir`, and `:eval` are rejected with a heading and line number. This notebook profile does not evaluate general Org Babel headers or render images and HTML. A run's block structure is fixed: edit source inside a block, not its `#+BEGIN_SRC` or `#+END_SRC` boundaries. Restart after an external structural edit. Press `F10` to run exactly one pending step through the Emacs worker and `emacs-jupyter`. The next cell is then ready to edit; the result panel shows the last completed step. Press `F9` to browse earlier results without rerunning them. Use `Tab` to move among the navigator, editor, scrollable result, Emacs and Jupyter logs, and controls. Type in the Python editor and use `Ctrl+Z` to undo; edits are saved to the run copy before navigation or execution. Editing a completed cell marks its old result stale. `Ctrl+R` restarts while keeping cell edits and clearing results; `Ctrl+Q` cleans up and quits. The result panel separates stdout, displayed text, errors, unsupported MIME labels, and empty output. Output is text only. The chosen source file is never edited. The run copy, private bridge socket, and worker log are under `target/insula/scenario/<run-id>/`.

The TUI communicates only with the persistent Emacs worker in the Emacs insula. That worker runs each cell with `emacs-jupyter` against the kernel in the separate Jupyter insula. The server listens on `http://127.0.0.1:8899/`; its token file is `target/insula/shared/token` (mode `0600`). The token value is not shown in the TUI. Run `insula/scenarios/run_tui.sh --verify` to execute the bundled cells through the same bridge and check GPU output, shared workspace evidence, and kernel cleanup. If the worker connection is lost during a step, the TUI reports an unknown outcome and prints the run path for inspection; it does not resend the cell.

## Origins

Torch Dojo builds on [TorchCode](https://github.com/duoan/TorchCode).
Credit to duoan and the upstream contributors for the original project.
