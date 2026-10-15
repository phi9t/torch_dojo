# Deep-ML live catalog: new ML/AI coding leads

Checked 2026-10-06 against the live [Deep-ML problem catalog](https://www.deep-ml.com/problems), its first-party [catalog endpoint](https://api.deep-ml.com/list-problems), and selected [problem detail pages](https://www.deep-ml.com/problems/382). The catalog endpoint returned **1,516 problem metadata entries** (ID, title, category, difficulty, creation field); 211 carried the `Reinforcement Learning` category. This is a selective list of **coding problems** beyond the [earlier survey](deep-ml-question-survey.md), not a complete catalog. Links below go directly to first-party problem pages; proposed dojo angles are original extensions, not copied contracts. Compare with the current [`dojo/` exercises 0001–0088](../../dojo/), especially the new RL and post-training exercises 0082–0088.

## Strongest new candidates

| Live problem | Distinctive dojo angle / gap |
| --- | --- |
| [484 — Reward Model Loss from Pairwise Human Preferences](https://www.deep-ml.com/problems/484) | Fit reward scores from preference pairs; unlike `0085`, trains the reward model rather than the policy. |
| [486 — Adaptive KL-Penalized Reward Shaping for RLHF](https://www.deep-ml.com/problems/486) | Shape rollout rewards and adjust KL pressure against a target; a new bridge between `0086` advantages and policy updates. |
| [1266 — RLOO: REINFORCE Leave-One-Out Advantages](https://www.deep-ml.com/problems/1266) | Each completion uses the other sampled completions as its baseline; compare variance and grouping with `0084` and `0087`. |
| [1267 — Process Reward Models: Aggregation and Step Loss](https://www.deep-ml.com/problems/1267) | Score intermediate reasoning steps and aggregate them; neither the current pairwise DPO nor group reward exercises models step-level supervision. |
| [529 — On-Policy Trajectory Sampling](https://www.deep-ml.com/problems/529) | Generate state-action trajectories from a stochastic policy and measure visitation; supplies the rollout data missing from the current PPO loss and GAE exercises. |
| [537 — Experience Replay Implementation](https://www.deep-ml.com/problems/537) | Bounded transition store with seeded sampling and terminal flags; needed before a deep Q-learning loop, beyond tabular `0083`. |
| [595 — Trust Region Policy Optimization (TRPO)](https://www.deep-ml.com/problems/595) | Constrained policy update and KL check; distinct from `0039`/`0087` clipped objectives. |
| [396 — Implement DDPM Reverse Sampling Step](https://www.deep-ml.com/problems/396) | Reverse diffusion posterior and stochastic denoising with reproducible noise; no diffusion sampler in dojo. |
| [398 — DDIM Deterministic Sampling Step](https://www.deep-ml.com/problems/398) | Deterministic, skipped-step diffusion trajectory; natural follow-up to 396. |
| [500 — Training a Rectified Flow Diffusion Model](https://www.deep-ml.com/problems/500) | Interpolated noisy states and velocity target; adds a modern generative-model training objective. |
| [405 — Multi-Head Latent Attention (MLA)](https://www.deep-ml.com/problems/405) | Compressed KV latent state with projection shapes; advances beyond `0010` GQA and `0014` KV cache. |
| [492 — Implementing PagedAttention: Block-wise Attention Computation](https://www.deep-ml.com/problems/492) | Read noncontiguous KV blocks through a page table while preserving attention output; inference memory behavior is absent from dojo. |
| [1257 — Simplified GPTQ: Sequential Quantization with Error Compensation](https://www.deep-ml.com/problems/1257) | Quantize weights in sequence and propagate error; more algorithmic than float-format arithmetic. |
| [720 — Implement a Vision Transformer (ViT) Image Encoder](https://www.deep-ml.com/problems/720) | Full class-token and encoder path; specifically extends `0027` patch embedding. |
| [1413 — Feature Pyramid Network: Top-Down Merge and Level Assignment](https://www.deep-ml.com/problems/1413) | Multiscale feature fusion and object-to-level assignment; distinct from `0070` anchor matching. |
| [724 — Frechet Inception Distance (FID) for Image Quality Evaluation](https://www.deep-ml.com/problems/724) | Compare generated and real feature distributions using means and covariances; unlike `0074` detection AP. |
| [712 — Compute Frechet Video Distance (FVD) for Video Quality Evaluation](https://www.deep-ml.com/problems/712) | Extend feature-distribution evaluation to video clips; useful after 724. |
| [1408 — Ranking Metrics: NDCG@k, MRR and MAP@k](https://www.deep-ml.com/problems/1408) | Evaluate ordered retrieval outputs, graded relevance, ties, and empty relevance; absent from current classification/detection metrics. |
| [1466 — Probability Calibration: Platt Scaling and Isotonic Regression](https://www.deep-ml.com/problems/1466) | Fit a held-out score-to-probability map; `0069` measures calibration but does not repair it. |
| [1475 — GraphSAGE: Mean and Pool Aggregators with Neighbor Sampling](https://www.deep-ml.com/problems/1475) | Sample bounded neighborhoods and aggregate node features; opens a graph-neural-network track. |
| [1476 — Graph Attention Layer with Masked Softmax](https://www.deep-ml.com/problems/1476) | Normalize learned attention only over graph neighbors; different sparsity structure from transformer attention. |

## Other worthwhile follow-ups

### Language models, adaptation, and serving

| Live problem | Distinctive dojo angle / gap |
| --- | --- |
| [223 — QLoRA: Quantized Low-Rank Adaptation Forward Pass](https://www.deep-ml.com/problems/223) | Quantized frozen base plus trainable adapter; follows `0026` LoRA and `0036` INT8 rather than repeating either. |
| [383 — Top-p (Nucleus) Sampling](https://www.deep-ml.com/problems/383) | Stable probability ordering, smallest retained nucleus, tie policy, and renormalization. |
| [419 — Combined Token Sampling Pipeline (Temperature + Top-k + Top-p)](https://www.deep-ml.com/problems/419) | Define ordering and edge cases when the filters compose; follows `0032` top-k sampling. |
| [425 — Continuous Batching (In-Flight Batching) Simulator](https://www.deep-ml.com/problems/425) | Admit and retire generation requests step by step under a batch cap. |
| [493 — Building a Virtual Memory System for KV Cache](https://www.deep-ml.com/problems/493) | Allocate, append, and free KV blocks; pairs with 492 but is a distinct state-management exercise. |
| [769 — DPO Loss with NLL Regularization on Chosen Sequences](https://www.deep-ml.com/problems/769) | **Only as a follow-up to `0085`**: add chosen-response likelihood regularization to the established DPO loss. |
| [1069 — Preference Dataset Collate Function for DPO](https://www.deep-ml.com/problems/1069) | Pack chosen/rejected responses with explicit prompt, completion, and padding masks; a data-contract complement to `0085`. |

### Vision, world models, and generative modeling

| Live problem | Distinctive dojo angle / gap |
| --- | --- |
| [397 — Classifier-Free Guidance for Conditional Diffusion](https://www.deep-ml.com/problems/397) | Mix conditional and unconditional predictions in a reverse step; follows DDPM 396. |
| [687 — Perceptual Image Distance from Deep Features](https://www.deep-ml.com/problems/687) | Compare normalized multilevel feature maps, not raw pixels. |
| [690 — Hybrid Vision Transformer for Multi-Task Driving Perception](https://www.deep-ml.com/problems/690) | Shared image backbone with multiple perception heads; broader than current geometry and detection metrics. |
| [185 — Optical Flow EPE with Masks (OmniWorld-style metric)](https://www.deep-ml.com/problems/185) | Evaluate predicted flow with valid-pixel masks; `0056` handles trajectory displacement rather than dense pixel motion. |
| [499 — Novel View Synthesis with Depth Reprojection](https://www.deep-ml.com/problems/499) | Warp pixels through depth and camera transforms with occlusion policy; combines but extends `0044` projection and `0051` unprojection. |
| [689 — Pose Accuracy Evaluation for World Models](https://www.deep-ml.com/problems/689) | Measure world-model pose predictions across a rollout; different from `0045` rigid alignment. |
| [498 — Implementing a Kinematic Vehicle Model for Autonomous Driving](https://www.deep-ml.com/problems/498) | Roll out steering and speed controls with a bicycle-model convention; extends `0055` constant-velocity motion. |
| [704 — Video Tokenizer with Spatiotemporal VQ-VAE](https://www.deep-ml.com/problems/704) | Convert space-time features to codebook indices; advanced video representation task. |
| [785 — CLIP Image-Text Alignment Score Filtering](https://www.deep-ml.com/problems/785) | Cross-modal cosine score matrix and data filtering. |
| [793 — Aspect-Ratio Preserving Image Tiling for Vision Transformers](https://www.deep-ml.com/problems/793) | Choose patch grid while respecting image geometry and token budget. |
| [1406 — ArcFace: Additive Angular Margin Loss for Face Recognition](https://www.deep-ml.com/problems/1406) | Margin applied to normalized class cosine logits; new metric-learning loss. |

### Classical ML, robust evaluation, and privacy

| Live problem | Distinctive dojo angle / gap |
| --- | --- |
| [807 — Random Forest Fit from Scratch](https://www.deep-ml.com/problems/807) | Bootstrap trees, feature subsampling, and vote aggregation; later than survey's single decision-tree lead. |
| [811 — Histogram-Based Gradient Boosting Split Finding](https://www.deep-ml.com/problems/811) | Binned gradient/hessian statistics and best split; more focused than a complete ensemble. |
| [1467 — Correcting Probabilities After Downsampling the Majority Class](https://www.deep-ml.com/problems/1467) | Reconcile training prior and deployment prior after class sampling. |
| [1411 — DP-SGD Step: Per-Example Clipping and Gaussian Noise](https://www.deep-ml.com/problems/1411) | Clip each example gradient before averaging and noise; distinct from ordinary global clipping. |
| [753 — Multi-Needle Long-Context Retrieval Evaluation](https://www.deep-ml.com/problems/753) | Measure whether multiple targets survive a long context; adds an LLM-specific evaluation slice. |
| [316 — MMLU Log-Probability Scoring](https://www.deep-ml.com/problems/316) | Score choices via model log-probabilities rather than parse generated letters. |
| [317 — Rubric-Based LLM Judge Evaluation](https://www.deep-ml.com/problems/317) and [323 — Pairwise Preference Judge for LLM Comparison](https://www.deep-ml.com/problems/323) | Rubric and pairwise evaluation workflows; useful with explicit tie/abstain and order-bias policies. |
| [1337 — Cohen's Kappa and Judge Position Bias](https://www.deep-ml.com/problems/1337) | Agreement and position-swap diagnostics for LLM judges. |
| [1213 — 1D Kalman Filter Predict-Update Step](https://www.deep-ml.com/problems/1213) | State/uncertainty propagation and measurement correction; complements `0055` constant-velocity rollout. |
| [1212 — Quaternion SLERP Interpolation](https://www.deep-ml.com/problems/1212) | Interpolate orientation through the shortest rotation while handling nearly aligned and antipodal quaternions. |

### RL after the current tabular and policy-gradient exercises

| Live problem | Distinctive dojo angle / gap |
| --- | --- |
| [682 — Behavior Cloning: Learning from Expert Demonstrations](https://www.deep-ml.com/problems/682) | Supervised policy from demonstrations, with an explicit rollout distribution-shift check. |
| [978 — Monte Carlo Tree Search with Neural Policy and Value](https://www.deep-ml.com/problems/978) | Search guided by learned priors/values, visit-count targets, and backup logic; substantial advanced exercise. |
| [593 — Rainbow DQN Implementation](https://www.deep-ml.com/problems/593) | Distributional/double/dueling update; only after replay and basic DQN. |
| [504 — Distributed On-Policy Reinforcement Learning](https://www.deep-ml.com/problems/504) | Rollout versioning and synchronous learner updates; systems-oriented extension, not another loss-only question. |

## Overlap and scope cautions

- The earlier survey already lists [18 K-fold](https://www.deep-ml.com/problems/18), [19 PCA](https://www.deep-ml.com/problems/19), [62 BPTT](https://www.deep-ml.com/problems/62), [160 mixed precision](https://www.deep-ml.com/problems/160), [276 ROC](https://www.deep-ml.com/problems/276), [277 AUC](https://www.deep-ml.com/problems/277), [840 stratified K-fold](https://www.deep-ml.com/problems/840), and several other foundational leads. They are omitted from the new-candidate tables.
- [382 DPO loss](https://www.deep-ml.com/problems/382), [487 GAE with episode boundaries](https://www.deep-ml.com/problems/487), [224 group-relative advantage](https://www.deep-ml.com/problems/224), [225 GRPO KL estimator](https://www.deep-ml.com/problems/225), and [485 PPO clipped surrogate](https://www.deep-ml.com/problems/485) are now substantially covered by `0037`–`0039` and `0085`–`0087`. [1395 object-detection IoU and mAP](https://www.deep-ml.com/problems/1395) overlaps `0062` and `0074`; [1253 anchor matching](https://www.deep-ml.com/problems/1253) overlaps `0070`.
- [222 LoRA](https://www.deep-ml.com/problems/222), [380 BPE](https://www.deep-ml.com/problems/380), [381 RoPE](https://www.deep-ml.com/problems/381), [391 GQA](https://www.deep-ml.com/problems/391), [394 speculative verification](https://www.deep-ml.com/problems/394), and [426 per-channel quantization](https://www.deep-ml.com/problems/426) substantially overlap dojo `0026`, `0035`, `0024`, `0010`, `0034`, and `0036` respectively.
- Titles/categories were read from the live catalog API; selected individual pages were checked for problem-specific context. Deep-ML categories are not a reliable relevance filter by themselves (for example, RL and language-model topics also appear under `Machine Learning`, `ML`, `Deep Learning`, and `pytorch`). The catalog can change, and a direct URL does not guarantee free access to every editor or hidden test.
- The site's [open problem repository](https://github.com/Open-Deep-ML/DML-OpenProblem) is much smaller than the live catalog. Its [Educational Use Only license](https://github.com/Open-Deep-ML/DML-OpenProblem/blob/main/LICENSE) is a reason to author fresh dojo prose, contracts, implementations, and verifiers rather than transplanting published problem material.
