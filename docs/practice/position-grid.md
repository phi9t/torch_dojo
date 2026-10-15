# Position grid resize: background and motivation

Background for [0111 position grid](../../dojo/0111_position_grid.py). The
exercise itself holds the contract; this page explains why the contract exists.

## The problem

A Vision Transformer cuts an image into patches and adds a learned position
embedding to each. Train at 224×224 with 16×16 patches and the model owns a
14×14 grid of 196 embeddings, plus a prefix token such as `[CLS]`.

A larger image, say 448×336, yields a 28×21 grid. The checkpoint has no
embeddings for those positions, so the learned table must be resized to match.

## The fix

Reshape the grid embeddings to `[B, C, H, W]`, interpolate to the new size,
flatten back, and put the prefix tokens in front untouched:

```python
# pos_embed: [1, 1 + 14*14, C]
new_pos = resize_positions(pos_embed, old_hw=(14, 14), new_hw=(28, 21), prefix_tokens=1)
# -> [1, 1 + 28*21, C]
```

Prefix tokens (`[CLS]`, a distillation token, register tokens) are not
locations, so they are copied exactly and never blended. That is why
`prefix_tokens` is a count rather than a flag.

## Where this shows up

- Fine-tuning at higher resolution (the original ViT recipe: pretrain at 224,
  fine-tune at 384).
- Loading a checkpoint into a model built with a different `img_size`, as `timm`
  and Hugging Face loaders do.
- Non-square inputs such as documents, screenshots and video frames. This is why
  the exercise takes explicit `(H, W)` instead of inferring `sqrt(token_count)`.
- Video and multimodal models that resize the spatial grid and handle time
  separately.

## Why the details matter

- **The convention must match training.** A checkpoint trained with
  `align_corners=False` bilinear shifts slightly if resized another way, and
  accuracy drops quietly. Parity with a framework call says nothing about how a
  given model was trained. Step 5 asks how an adapter records a checkpoint's
  actual interpolation rules.
- **Square grids hide H/W swaps.** Equal token counts do not expose the error;
  a numbered ramp on a rectangular grid does.
- **Downsampling here does not antialias.** It samples the surface. A
  different contract is needed if low-pass filtering is required.
- **Gradients must reach the source table**, including the prefix positions, or
  fine-tuning silently stops training part of the embedding.
- **Same-size grids take an exact identity path**, so loading a checkpoint at its
  native resolution changes nothing.

## Relation to other workouts

The half-pixel mapping is the same pixel-center convention practiced in
[0042 pixel centers](../../dojo/0042_pixel_centers.py), and the four-neighbor
blend is the one in [0104 bilinear features](../../dojo/0104_bilinear_features.py).
In production you would call
`F.interpolate(..., mode="bilinear", align_corners=False)`; the exercise builds it
by hand so the prefix handling, H/W order and exact identity can be verified.
