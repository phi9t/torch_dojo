# Softmax to k means to geometry workouts

Practice in this order: [softmax](../../dojo/0002_softmax.py) →
[cross-entropy and indexing](../../dojo/0059_stable_cross_entropy.py) →
[k-means](../../dojo/0061_kmeans.py) →
[voxel pooling](../../dojo/0066_voxel_pool.py) and
[pixel-center geometry](../../dojo/0042_pixel_centers.py).
For each workout, inspect a tiny example, predict shapes and values, implement,
then run adversarial checks. Keep your prediction beside the cell before running it.

## Softmax and cross entropy

Start with two equal logits and target class 1. Predict the probabilities,
loss, and each gradient entry. Then use logits around 1000: which exponential
overflows, and why does subtracting the row maximum help?

Integer labels select one logit per example. Derive
`L_i = logsumexp(z_i) - z_i[y_i]`, then differentiate both terms.
For ignored labels, divide by the valid-example count. Predict the zero loss
and correctly shaped zero gradient for empty and all-ignored batches.
The exercise defines this empty-input policy explicitly.

Before the main answer, complete #59's indexing mini-workout. Compare paired
coordinates with deliberately broadcast row and column grids. Explain each
added axis and the effect of `keepdims`. Implement a loss vector and check
each element before taking its mean. The adversarial example shows why an
accidental `(N,N)` subtraction can pass a scalar-mean test.

Adversarial checks include repeated class labels, unequal row losses, ignored
rows, invalid negative labels, constant row shifts, and finite differences.
The reference eval contains the gradient derivation and the shape answers.

## K means and repeated destinations

Use points `[0,1,9,10]` and centers `[0,10]`. Predict assignments, gathered
centers, cluster sums, counts, new centers, and inertia. Implement an update.
Then try sending `[2,3,5]` to one cluster: all three values must contribute.

`centers[labels]` gathers an assigned center for each point. For updates,
`sums[labels] += X` can lose contributions when labels repeat.
Use `np.add.at` to accumulate at repeated destinations; the
[NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
explains this buffered-update distinction.

An embedding lookup gathers vectors by token ID. Repeated tokens accumulate
gradients into the same embedding row. Overlapping point-cloud neighborhoods
likewise accumulate backward contributions at shared points. In cross-entropy,
repeated class labels in different examples address distinct `(row,class)`
coordinates. Compare the full coordinates before deciding that writes collide.

Check repeated destinations, multidimensional points, empty clusters, ties,
zero iterations, and agreement across assignment block sizes. Carry the same
accumulation reasoning into voxel pooling before the geometry exercises.

## Hints

For the loss vector, align one row's normalizer with that row's selected logit.
For gradients, subtract the target contribution once per valid coordinate,
then apply the averaging factor. For cluster updates, accumulate sums and
counts separately; update only occupied centers. Complete these steps before
opening the neighboring `_eval.py` solutions.

## Memory and address probe

Predict the values and memory-sharing results before running:

```python
import numpy as np

a = np.arange(12, dtype=np.float64).reshape(3, 4)
view = a[:, ::2]
rows, cols = np.array([0, 2]), np.array([1, 3])
picked = a[rows, cols]
print(a.strides, view.strides, picked)
print(np.shares_memory(a, view), np.shares_memory(a, a[:, None]))
print(np.shares_memory(a, picked))
```

For a two-dimensional array, the byte address is
`base + row*stride[0] + col*stride[1]`. Use the actual strides for a transpose
or a reversed slice. NumPy strides count bytes; PyTorch strides count elements.
Gathering computes source addresses from coordinates and copies the selected
values. NumPy advanced-index reads copy data; basic slices and added axes are
views. An indexed assignment still writes to the original array.
See [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html).

## Compare with library operations after implementing

For valid targets, compare the selected `(N,)` vector with
`np.take_along_axis(logits, targets[:, None], axis=1)[:, 0]`.
In PyTorch, use `logits.gather(1, targets[:, None]).squeeze(1)`.
Filter ignored labels before either gather. See
[take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
and [torch.gather](https://docs.pytorch.org/docs/stable/generated/torch.gather.html).

Compare the from-scratch stable reduction with
`scipy.special.logsumexp(logits, axis=1)` or
`torch.logsumexp(logits, dim=1)`. These helpers compute the log normalizer;
they do not select targets or define your loss reduction. See
[SciPy logsumexp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html)
and [torch.logsumexp](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html).

For nonempty valid batches, compare loss and gradients with
`torch.nn.functional.cross_entropy`, using integer labels, no class weights,
no label smoothing, and the same `ignore_index`. Use `reduction="none"` to
inspect per-example losses, then explicitly select valid rows before averaging.
Keep the exercise's zero-valid-row branch: a mean with a zero denominator
does not supply its zero-loss policy. See
[PyTorch cross_entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html).
