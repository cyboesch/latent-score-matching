# Convolutional Latent Free-Energy MNIST Run Report

Run date: 2026-05-22 UTC

Notebook: `notebooks/conv_latent_free_energy_crossbar_mnist_jax.ipynb`

Source note: `notes/conv_latent_free_energy_crossbar.pdf`

## Why The Original Samples Failed

The screenshot from the first run was genuinely bad. There were two separate causes:

1. The notebook displayed the reverse process state at `T_MIN` directly. That state is still noisy. The corrected notebook now applies the final Tweedie denoising estimate `x0_hat = (x_t + sigma_t^2 * score(x_t,t)) / alpha_t` before visualization.
2. The pure local convolutional free-energy architecture is weak for global digit geometry. It learns local stroke fragments and patch statistics, but it has no global latent memory index or multiscale shape variable to enforce a coherent whole digit.

The first issue was an implementation/sampling omission. The second is an architectural limitation of this minimal model.

## Execution Environment

The notebook was executed end to end with:

```bash
jupyter nbconvert --to notebook --execute notebooks/conv_latent_free_energy_crossbar_mnist_jax.ipynb --inplace --ExecutePreprocessor.timeout=3600
```

JAX reported:

- JAX version: `0.4.33`
- devices: `[CudaDevice(id=0)]`
- default backend: `gpu`

A non-fatal XLA warning reported that the NVIDIA driver CUDA version is older than the PTX compiler version, so XLA disabled parallel compilation. Execution still completed on the GPU.

## Data

- Dataset: MNIST digits `0` and `1`
- Image size: `12 x 12 x 1`
- Training images: `1600`
- Validation images: `512`
- Training class counts: `{0: 751, 1: 849}`
- Training mean/std after `[-1, 1]` normalization: `-0.6710 / 0.6199`

## Correctness Tests

### Conservative Score Check

| t | sigma^2 | max score error |
|---:|---:|---:|
| 0.050 | 0.0952 | 9.537e-07 |
| 0.500 | 0.6321 | 1.192e-07 |
| 1.500 | 0.9502 | 1.192e-07 |
| 3.000 | 0.9975 | 1.192e-07 |

### Local Gradient Check

- local loss: `28.355907440185547`
- autodiff loss: `28.355907440185547`

| parameter | max abs error | relative error |
|---|---:|---:|
| `W` | 9.537e-07 | 6.129e-08 |
| `a` | 5.960e-08 | 3.780e-08 |
| `b` | 0.000e+00 | 0.000e+00 |

These checks validate the free-energy score and the local learning rule.

## Widened Hyperparameter Search

Quick sweep settings:

- `N_TIME_SWEEP = 8`
- `STEPS_PER_TIME_SWEEP = 250`
- `BATCH_SIZE = 128`
- `N_REVERSE_STEPS = 300`
- sampler: stochastic reverse SDE plus final Tweedie denoising
- validation loss averaged across all time slices

| hidden channels | kernel | W scale | lr | validation DSM loss | final train loss | median sample to train | train control | nearest-label counts |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 3 | 0.060 | 0.0030 | 22.065 | 23.703 | 4.814 | 2.540 | `{0: 6, 1: 58}` |
| 16 | 3 | 0.050 | 0.0030 | 20.526 | 20.127 | 5.011 | 2.540 | `{0: 13, 1: 51}` |
| 16 | 5 | 0.040 | 0.0020 | 21.049 | 19.673 | 4.503 | 2.540 | `{0: 14, 1: 50}` |
| 24 | 3 | 0.040 | 0.0020 | 23.023 | 23.340 | 4.689 | 2.540 | `{0: 11, 1: 53}` |
| 32 | 3 | 0.035 | 0.0015 | 25.353 | 22.938 | 4.369 | 2.540 | `{0: 7, 1: 57}` |
| 32 | 5 | 0.035 | 0.0010 | 22.467 | 22.594 | 4.442 | 2.540 | `{0: 11, 1: 53}` |
| 32 | 7 | 0.030 | 0.0008 | 23.857 | 22.334 | 4.192 | 2.540 | `{0: 10, 1: 54}` |
| 48 | 5 | 0.030 | 0.0008 | 23.802 | 23.863 | 4.757 | 2.540 | `{0: 16, 1: 48}` |
| 64 | 5 | 0.025 | 0.0006 | 25.271 | 23.950 | 5.363 | 2.540 | `{0: 17, 1: 47}` |

Lowest held-out DSM loss: `20.5256`.

Selected config within 15 percent of best DSM loss by sample diagnostic:

```python
{"hidden_ch": 32, "kernel": 5, "W_scale": 0.035, "lr": 0.001}
```

## Final Retrain

Final settings:

- `N_TIME_FINAL = 12`
- `STEPS_PER_TIME_FINAL = 1500`
- selected config: `hidden_ch=32`, `kernel=5`, `W_scale=0.035`, `lr=0.001`

Per-slice final losses:

| t | sigma^2 | final loss |
|---:|---:|---:|
| 3.000 | 0.9975 | 4.7716e-02 |
| 2.735 | 0.9958 | 9.0470e-02 |
| 2.469 | 0.9928 | 1.4085e-01 |
| 2.204 | 0.9878 | 2.3085e-01 |
| 1.938 | 0.9793 | 4.2072e-01 |
| 1.673 | 0.9648 | 7.1735e-01 |
| 1.407 | 0.9401 | 1.2197e+00 |
| 1.142 | 0.8981 | 2.2360e+00 |
| 0.876 | 0.8267 | 3.7181e+00 |
| 0.611 | 0.7053 | 7.7091e+00 |
| 0.345 | 0.4989 | 1.7382e+01 |
| 0.080 | 0.1479 | 1.1894e+02 |

Final metrics:

- final wall time reported inside the JAX training loop: `4.3s`
- final held-out DSM loss: `12.275`
- median sample to nearest train: `3.8245`
- median train to nearest train control: `2.5404`
- nearest-training-label counts: `{0: 11, 1: 53}`

## Interpretation

The sampler fix and widened search improved the sample diagnostic substantially relative to the original run. The original median sample-to-train distance before final denoising was about `6.64`; the corrected widened run reaches `3.82`.

That is better, but it is still not good enough. The model remains biased toward digit `1`, and generated samples are still farther from the training manifold than independent training images are from each other. This points to an architectural limit of the pure local one-layer convolutional latent free energy.

## Recommended Next Step

The next model should keep the conservative free-energy construction but add global coherence. The most direct option is a hybrid free energy:

```math
F(x,t)=F_{conv}(x,t)-\lambda\log\sum_a\exp\left(\frac{\alpha_t x^T c_a}{\sigma_t^2}-\frac{\alpha_t^2\|c_a\|^2}{2\sigma_t^2}+b_a\right).
```

That adds a categorical DenseAM/prototype hidden variable for whole-image geometry while retaining local convolutional Bernoulli hidden units for stroke statistics. Hyperparameter search alone is unlikely to make the pure local model produce clean MNIST digits.
