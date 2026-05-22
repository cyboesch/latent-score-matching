# Conv vs DenseAM Ablation Report

Run date: 2026-05-22 UTC

Notebook: `notebooks/hybrid_ablation_conv_vs_denseam_mnist_jax.ipynb`

Generator: `notebooks/_build_hybrid_ablation_conv_vs_denseam_mnist_jax.py`

## Goal

This experiment asks whether the hybrid model is actually using the convolutional Bernoulli hidden units, or whether the DenseAM prototypes are doing all of the work through memory retrieval.

The ablation compares five regimes:

| mode | conv hidden units | DenseAM prototypes | prototypes train? |
|---|---|---|---|
| `conv_only` | yes | no | no |
| `dense_frozen` | no | yes | no |
| `dense_train_proto` | no | yes | yes |
| `hybrid_frozen_proto` | yes | yes | no |
| `hybrid_train_all` | yes | yes | yes |

The most important controlled comparison is `dense_frozen` vs `hybrid_frozen_proto`. If adding trainable conv hidden units improves performance while the prototypes are frozen, then the conv layer is doing real work.

## Environment And Checks

JAX reported:

- JAX version: `0.4.33`
- devices: `[CudaDevice(id=0)]`
- default backend: `gpu`

Data:

- MNIST digits `0` and `1`
- `12 x 12 x 1`
- train images: `2000`
- validation images: `512`
- class counts: `{0: 943, 1: 1057}`

Conservative score checks passed:

| mode | max score error |
|---|---:|
| `conv_only` | 2.384e-07 |
| `dense_frozen` | 1.788e-07 |
| `hybrid_frozen_proto` | 2.384e-07 |

## Sweep Design

The sweep used:

- `K = [64, 128]`
- `lambda = [0.5, 1.0]`
- prototype init = `random`, `kmeans`
- modes = `dense_frozen`, `dense_train_proto`, `hybrid_frozen_proto`, `hybrid_train_all`
- plus one `conv_only` baseline

Metrics:

- `val`: held-out denoising-score loss; lower is better.
- `sample->train`: median distance from generated samples to nearest training image.
- `train control`: median train-image-to-nearest-train-image distance, `2.540`.
- `sample->sample`: median generated-sample nearest-neighbor distance; near zero means duplicates/collapse.
- `zero_frac`: fraction nearest-labelled as digit `0`; near `0.5` is balanced.
- component norms: RMS score contribution near low noise.

## Final Category Retrains

Each category winner from the sweep was retrained with the longer final schedule.

| mode | init | K | lambda | val | sample->train | train control | sample->sample | zero frac | counts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `conv_only` | random | 64 | 0.0 | 14.231 | 3.998 | 2.540 | 3.584 | 0.11 | `{0: 7, 1: 57}` |
| `dense_frozen` | random | 128 | 1.0 | 27.400 | 0.304 | 2.540 | 2.290 | 0.44 | `{0: 28, 1: 36}` |
| `dense_train_proto` | random | 64 | 1.0 | 21.795 | 1.282 | 2.540 | 0.000 | 0.44 | `{0: 28, 1: 36}` |
| `hybrid_frozen_proto` | random | 128 | 1.0 | 25.267 | 0.738 | 2.540 | 0.647 | 0.50 | `{0: 32, 1: 32}` |
| `hybrid_train_all` | random | 128 | 1.0 | 12.638 | 1.551 | 2.540 | 1.426 | 0.45 | `{0: 29, 1: 35}` |

## Direct Answer

### 1. DenseAM prototypes absolutely provide most of the global coherence.

The pure conv baseline is still poor:

- `conv_only sample->train = 3.998`, worse than the `2.540` train control.
- It is badly class-skewed: `{0: 7, 1: 57}`.

The global DenseAM term is what gives whole-digit attractor basins. This confirms the earlier diagnosis: local conv hidden units alone learn stroke fragments, not global digit shape.

### 2. Strong frozen DenseAM alone can look good by retrieval/memorization.

`dense_frozen` with `K=128, lambda=1.0` gives:

- `sample->train = 0.304`, much smaller than the `2.540` train control.
- no conv hidden units at all.

That means a strong fixed prototype term can pull samples extremely close to stored training images. This is not evidence of generalization; it is memory retrieval. Its held-out DSM loss is also bad: `27.400`.

### 3. Trainable DenseAM prototypes alone are not enough.

`dense_train_proto` gives:

- `val = 21.795`, still much worse than hybrid train-all.
- `sample->sample = 0.000`, indicating severe duplicate/collapse behavior.

So letting prototypes drift does not by itself solve the problem. It tends to collapse into repeated attractors.

### 4. The conv hidden units do help, especially when prototypes are not allowed to drift.

The cleanest evidence is from matched frozen-prototype sweep entries at moderate `lambda = 0.5`.

Random prototypes, `K=128, lambda=0.5`:

| model | val | sample->train | sample->sample | counts |
|---|---:|---:|---:|---|
| `dense_frozen` | 71.538 | 4.586 | 0.843 | `{0: 33, 1: 31}` |
| `hybrid_frozen_proto` | 19.224 | 1.435 | 2.177 | `{0: 30, 1: 34}` |

K-means prototypes, `K=128, lambda=0.5`:

| model | val | sample->train | sample->sample | counts |
|---|---:|---:|---:|---|
| `dense_frozen` | 72.088 | 4.888 | 0.156 | `{0: 31, 1: 33}` |
| `hybrid_frozen_proto` | 16.497 | 2.313 | 2.409 | `{0: 32, 1: 32}` |

This is the strongest result: with prototypes frozen, adding the conv/Bernoulli hidden layer massively improves score loss and sample placement. So the conv layer is not decorative.

### 5. In the best train-all hybrid, conv is dynamically important.

Final `hybrid_train_all` component norms near low noise:

| component | RMS score norm |
|---|---:|
| base | 68.149 |
| conv | 43.884 |
| DenseAM | 26.149 |
| total | 5.880 |

The conv score contribution is larger than the DenseAM contribution in this final model. The total is much smaller because the terms cancel to form a denoising force. This says the conv hidden units are actively shaping the force field, not merely sitting on top of the prototype memory.

## Interpretation

The answer is nuanced:

- If the question is “what makes samples globally digit-like?”, the answer is mostly the DenseAM prototype term.
- If the question is “does the conv/Bernoulli layer help beyond memorizing prototypes?”, the answer is yes, clearly, especially at moderate DenseAM strength and with frozen prototypes.
- If the question is “can DenseAM alone solve it?”, the answer is only by retrieval-like behavior or collapse. Its validation loss and duplicate diagnostics are bad.
- The best score model is the full hybrid train-all model, not DenseAM-only.

## Recommendation

The next clean experiment should avoid the misleading high-lambda retrieval regime:

1. Use frozen k-means prototypes.
2. Focus on `K=64,128` and `lambda=0.4..0.8`.
3. Train conv + biases only.
4. Add duplicate-rate and nearest-neighbor image grids as first-class diagnostics.
5. Optionally use time-dependent `lambda(t)`: weak at high noise, stronger near low noise.

That setup best tests whether local conv hidden units improve a non-memorizing global prior.
