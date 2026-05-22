# Hybrid Conv + DenseAM MNIST Sweep Report

Run date: 2026-05-22 UTC

Notebook: `notebooks/hybrid_conv_denseam_mnist_jax.ipynb`

Generator: `notebooks/_build_hybrid_conv_denseam_mnist_jax.py`

## Question

The pure convolutional latent free-energy model generated local stroke fragments rather than coherent MNIST digits. The hypothesis was that the failure was architectural: a local product-of-experts model can learn patch statistics, but it lacks a global hidden variable that enforces whole-digit shape.

The hybrid model tests that hypothesis by adding a global categorical DenseAM term to the local convolutional free energy:

```math
\log p_\theta(x,t)=\log p_{conv}(x,t)+\lambda\log\sum_a\exp\ell_a(x,t)
```

with

```math
\ell_a(x,t)=\frac{\alpha_t x^T c_a}{\sigma_t^2}-\frac{\alpha_t^2\|c_a\|^2}{2\sigma_t^2}+b_a.
```

The resulting score is

```math
s(x,t)=s_{conv}(x,t)+\lambda\frac{\alpha_t}{\sigma_t^2}\sum_a r_a(x,t)c_a.
```

The model remains conservative because both terms come from one scalar free energy.

## Execution Environment

The notebook was executed end to end with:

```bash
jupyter nbconvert --to notebook --execute notebooks/hybrid_conv_denseam_mnist_jax.ipynb --inplace --ExecutePreprocessor.timeout=3600
```

JAX reported:

- JAX version: `0.4.33`
- devices: `[CudaDevice(id=0)]`
- default backend: `gpu`

The usual non-fatal XLA warning appeared about the NVIDIA driver CUDA version being older than the PTX compiler version. Execution still completed on GPU.

## Data

- Dataset: MNIST digits `0` and `1`
- Image size: `12 x 12 x 1`
- Training images: `2000`
- Validation images: `512`
- Training class counts: `{0: 943, 1: 1057}`
- Training mean/std after `[-1, 1]` normalization: `-0.6701 / 0.6208`

## Conservative Score Check

The analytic hybrid score was compared against `jax.grad` of the hybrid log-density:

| t | sigma^2 | max score error |
|---:|---:|---:|
| 0.080 | 0.1479 | 9.537e-07 |
| 0.500 | 0.6321 | 2.384e-07 |
| 1.500 | 0.9502 | 1.192e-07 |
| 3.000 | 0.9975 | 1.192e-07 |

This validates the hybrid free-energy score implementation.

## Sweep Setup

The sweep varied:

- `K`: number of global prototypes, `[16, 32, 64, 128]`
- `lambda`: global DenseAM strength, `[0.25, 0.5, 1.0]`
- initialization: `random` training images vs `kmeans` centers

Fixed conv layer:

- hidden channels: `32`
- kernel size: `5`
- weight scale: `0.035`

Sweep training:

- `N_TIME_SWEEP = 8`
- `STEPS_PER_TIME_SWEEP = 300`
- `BATCH_SIZE = 128`
- sampler: stochastic reverse SDE plus final Tweedie denoising

## Sweep Results

| init | K | lambda | val DSM | train loss | sample to train | sample to proto | train control | zero frac | counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random | 16 | 0.25 | 19.402 | 20.109 | 2.172 | 4.481 | 2.540 | 0.41 | `{0: 26, 1: 38}` |
| random | 16 | 0.50 | 20.295 | 19.758 | 2.777 | 2.042 | 2.540 | 0.62 | `{0: 40, 1: 24}` |
| random | 16 | 1.00 | 24.603 | 25.007 | 1.445 | 2.136 | 2.540 | 0.38 | `{0: 24, 1: 40}` |
| random | 32 | 0.25 | 19.512 | 18.402 | 2.280 | 4.924 | 2.540 | 0.45 | `{0: 29, 1: 35}` |
| random | 32 | 0.50 | 20.626 | 20.218 | 1.443 | 2.175 | 2.540 | 0.39 | `{0: 25, 1: 39}` |
| random | 32 | 1.00 | 24.889 | 24.123 | 1.889 | 1.922 | 2.540 | 0.48 | `{0: 31, 1: 33}` |
| random | 64 | 0.25 | 18.940 | 19.659 | 1.963 | 5.490 | 2.540 | 0.34 | `{0: 22, 1: 42}` |
| random | 64 | 0.50 | 19.882 | 18.111 | 2.456 | 3.118 | 2.540 | 0.59 | `{0: 38, 1: 26}` |
| random | 64 | 1.00 | 26.784 | 21.072 | 1.430 | 2.198 | 2.540 | 0.48 | `{0: 31, 1: 33}` |
| random | 128 | 0.25 | 18.785 | 18.737 | 1.983 | 5.874 | 2.540 | 0.28 | `{0: 18, 1: 46}` |
| random | 128 | 0.50 | 19.052 | 17.822 | 1.364 | 4.538 | 2.540 | 0.34 | `{0: 22, 1: 42}` |
| random | 128 | 1.00 | 20.662 | 17.933 | 1.243 | 3.333 | 2.540 | 0.45 | `{0: 29, 1: 35}` |
| kmeans | 16 | 0.25 | 17.592 | 18.242 | 3.212 | 4.915 | 2.540 | 0.53 | `{0: 34, 1: 30}` |
| kmeans | 16 | 0.50 | 19.745 | 18.412 | 2.564 | 2.002 | 2.540 | 0.56 | `{0: 36, 1: 28}` |
| kmeans | 16 | 1.00 | 20.113 | 21.575 | 1.713 | 2.326 | 2.540 | 0.45 | `{0: 29, 1: 35}` |
| kmeans | 32 | 0.25 | 18.930 | 17.438 | 2.083 | 5.257 | 2.540 | 0.45 | `{0: 29, 1: 35}` |
| kmeans | 32 | 0.50 | 20.029 | 18.701 | 2.357 | 2.213 | 2.540 | 0.56 | `{0: 36, 1: 28}` |
| kmeans | 32 | 1.00 | 22.860 | 22.956 | 1.588 | 1.929 | 2.540 | 0.39 | `{0: 25, 1: 39}` |
| kmeans | 64 | 0.25 | 19.136 | 18.798 | 2.113 | 5.450 | 2.540 | 0.36 | `{0: 23, 1: 41}` |
| kmeans | 64 | 0.50 | 18.745 | 16.955 | 1.418 | 3.291 | 2.540 | 0.41 | `{0: 26, 1: 38}` |
| kmeans | 64 | 1.00 | 18.382 | 20.030 | 1.474 | 2.444 | 2.540 | 0.38 | `{0: 24, 1: 40}` |
| kmeans | 128 | 0.25 | 18.665 | 18.260 | 1.985 | 6.045 | 2.540 | 0.34 | `{0: 22, 1: 42}` |
| kmeans | 128 | 0.50 | 17.607 | 16.750 | 2.657 | 4.656 | 2.540 | 0.62 | `{0: 40, 1: 24}` |
| kmeans | 128 | 1.00 | 21.706 | 16.827 | 1.862 | 3.258 | 2.540 | 0.59 | `{0: 38, 1: 26}` |

The best validation loss was `17.592` from `kmeans, K=16, lambda=0.25`, but it had weaker sample placement. The selected config used the sample diagnostic among models within 25 percent of the best validation loss:

```python
{"init": "random", "K": 128, "lambda": 1.0}
```

## Final Retrain

Final selected config:

- prototype init: `random`
- `K = 128`
- `lambda = 1.0`
- `N_TIME_FINAL = 12`
- `STEPS_PER_TIME_FINAL = 1200`

Final per-slice losses:

| t | sigma^2 | final loss |
|---:|---:|---:|
| 3.000 | 0.9975 | 5.0921e-02 |
| 2.735 | 0.9958 | 8.3508e-02 |
| 2.469 | 0.9928 | 1.3733e-01 |
| 2.204 | 0.9878 | 2.2924e-01 |
| 1.938 | 0.9793 | 3.8896e-01 |
| 1.673 | 0.9648 | 6.4091e-01 |
| 1.407 | 0.9401 | 9.6937e-01 |
| 1.142 | 0.8981 | 1.6356e+00 |
| 0.876 | 0.8267 | 2.0356e+00 |
| 0.611 | 0.7053 | 3.6669e+00 |
| 0.345 | 0.4989 | 9.8011e+00 |
| 0.080 | 0.1479 | 1.0592e+02 |

Final metrics:

- final held-out DSM loss: `11.746`
- median sample to nearest train: `2.288`
- median train to nearest train control: `2.540`
- nearest-label counts: `{0: 34, 1: 30}`
- zero fraction: `0.531`
- median sample to prototype: `7.607`

## Comparison To Pure Conv

The widened pure-conv run had:

- median sample to nearest train: `3.825`
- train control: `2.540`
- nearest-label counts: `{0: 11, 1: 53}`
- held-out DSM loss: `12.275`

The hybrid final run has:

- median sample to nearest train: `2.288`
- train control: `2.540`
- nearest-label counts: `{0: 34, 1: 30}`
- held-out DSM loss: `11.746`

So the hybrid term fixes the main failure mode in the diagnostics: global coherence and class balance improve substantially. The model is now close to the training manifold by the nearest-neighbor control. The remaining caveat is that `K=128, lambda=1.0` is a fairly strong global memory; it may be closer to a memory-guided sampler than a highly generalizing image model.

## Interpretation

The experiment supports the diagnosis: the pure conv model failed because it was too local. Adding the categorical DenseAM hidden variable gives whole-image attractor basins, while the conv layer provides local stroke corrections.

The interesting regime is not necessarily the lowest validation loss. Low `lambda` and k-means centers score well under DSM, but may not sample as cleanly. Higher `lambda` and larger `K` pull samples closer to coherent digits and improve class balance, but increase memorization risk.

## Suggested Next Sweep

1. Freeze prototypes after k-means initialization and train only conv plus biases. This would isolate retrieval from prototype drift.
2. Sweep `lambda` more finely around `[0.6, 0.8, 1.0, 1.2]` for `K=64` and `K=128`.
3. Add a prototype diversity/memorization diagnostic: sample-to-nearest-train image pairs and duplicate-rate among generated samples.
4. Try a time-dependent `lambda(t)`, weaker at high noise and stronger near low noise.
