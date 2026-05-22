# Convolutional Latent Free-Energy MNIST Implementation

This note explains `notebooks/conv_latent_free_energy_crossbar_mnist_jax.ipynb`, which implements the architecture from `notes/conv_latent_free_energy_crossbar.pdf` in JAX and applies it to downsampled MNIST images.

## Core Model

The notebook uses a Gaussian-visible / Bernoulli-hidden convolutional free-energy model:

```math
F_\theta(x,t)=\frac{\|x\|^2}{2\sigma_t^2}-b^T x-\sum_{u,k}\operatorname{softplus}(a_k+\langle w_k,P_u x\rangle).
```

The score is the negative free-energy gradient:

```math
s_\theta(x,t)=-\nabla_x F_\theta(x,t)=-\frac{x}{\sigma_t^2}+b+W^\dagger * \sigma(W*x+a).
```

The encoder and decoder weights are tied. In code, this is the pairing of `conv_encoder(x, W)` and `conv_transpose_tied(q, W)`. The tied transpose is the conservative-force constraint: the vector field is the gradient of a scalar free energy rather than a generic CNN output.

## Data And Corruption

The default run uses MNIST digits `0` and `1`, downsampled to `12 x 12` and normalized to `[-1, 1]`. The forward noising process is Ornstein-Uhlenbeck:

```math
x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad \alpha_t=e^{-t},\qquad \sigma_t^2=k_BT(1-\alpha_t^2).
```

The target score during training is analytic:

```math
s^\star(x_t,x_0,t)=\frac{\alpha_t x_0-x_t}{\sigma_t^2}.
```

The loss at each time slice is denoising score matching:

```math
\ell=\frac{1}{2}\|s_\theta(x_t,t)-s^\star(x_t,x_0,t)\|^2.
```

## Local Crossbar Gradient

The notebook trains with the local tied-weight gradient from the PDF. Let

```math
\delta=s_\theta-s^\star,\qquad q=\sigma(W*x+a),\qquad \eta=W*\delta.
```

For a shared convolutional filter `w_k`, the gradient is

```math
\frac{\partial \ell}{\partial w_k}=\sum_u\left[q_{uk}P_u\delta+q_{uk}(1-q_{uk})\eta_{uk}P_u x\right].
```

The two terms have the hardware interpretation from the note:

- `q * patch(delta)` is the decoder or force-synthesis update.
- `q * (1 - q) * eta * patch(x)` is the encoder susceptibility correction.

In code this is implemented in `local_conv_grads_from_batch` using `jax.lax.conv_general_dilated_patches` plus two `einsum` reductions. The hidden error current `eta` is computed by the same convolutional crossbar operation applied to the visible force error.

## Sampling Fix

The first version displayed the state at `T_MIN` directly. That was a mistake for image display: it is still a noised state. The notebook now applies the standard final Tweedie estimate:

```math
\hat x_0=\frac{x_t+\sigma_t^2 s_\theta(x_t,t)}{\alpha_t}.
```

This materially improves the samples and diagnostics. The stochastic reverse SDE plus final denoising is the default; `reverse_sample_conv(..., sampler="ode")` is also available for the probability-flow ODE path.

## Correctness Checks

The notebook contains two numerical checks before training:

1. `conv_score` is compared to `jax.grad(conv_log_unnormalized)` at several diffusion times.
2. `local_conv_grads_from_batch` is compared to `jax.value_and_grad` of the denoising loss.

Both checks match to float32 precision in the executed notebook.

## Training And Search

Training is per time slice, high noise to low noise, warm-starting each lower-noise slice from the previous higher-noise solution. The widened quick search now tests nine configurations across hidden-channel count, kernel size, initialization scale, and learning rate.

The final config is not selected by DSM loss alone. Among configurations within 15 percent of the best validation DSM loss, the notebook chooses the one with the best generated-sample nearest-neighbour diagnostic. This avoids choosing a model that scores well under the denoising objective but produces worse samples.

## Current Failure Mode

Even after the sampler fix and wider search, the local-only model is still not a satisfying MNIST generator. It improves from the original garbage image, but it remains biased toward digit `1` and its generated samples are still farther from the training manifold than the train-to-train control.

The likely reason is architectural, not just hyperparameter search. A one-layer local convolutional product-of-experts can learn stroke-like patch statistics, but it has no global latent variable or multiscale shape code that forces a set of strokes to form a coherent `0` or `1`. Widening channels and kernels helps only modestly. A better next model should add a global categorical DenseAM/prototype term or a multiscale/global hidden layer while keeping the whole score conservative.
