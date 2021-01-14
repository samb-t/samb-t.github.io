---
layout: single
classes: wide
author_profile: true
title: "Denoising Diffusion and Score Matching"
categories: generative-models

read_time: true
comments: true
share: true
related: true

excerpt: Denoising Diffusion and Score Matching

sidebar:
  nav: "gentuts"
---

## Summary

1. Song and Ermon Score matching
2. Denoising Derivation
3. Improvements e.g. cosine scheduling
4. SDE approach to speed up

## Sampling with Langevin Dynamic

Langevin dynamics equation.

$$
\begin{equation}
\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_{t-1} + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\tilde{\mathbf{x}}_{t-1}) + \sqrt{\epsilon} \mathbf{z}_t
\end{equation}
$$

Converges to the true distribution as $T \rightarrow \infty$.

## Score Matching

We would like to estimate the gradient of the data's log-likelihod, known as the score, i.e. $\nabla_\mathbf{x} \log p_d(\mathbf{x})$ with a neural network $s_\theta(\mathbf{x})$. Minimising the mean squared error

$$
\begin{equation}
\frac{1}{2} \mathbb{E}_{p_d}[ || s_\theta(\mathbf{x}) - \nabla_\mathbf{x} \log p_d(\mathbf{x}) ||^2_2 ]
\end{equation}
$$

which is equivalent to: (if there is an easy proof, that would be nice)

$$
\begin{equation}
\mathbb{E}_{p_d}(\mathbf{x}) \bigg[ \text{tr}(\nabla_\mathbf{x} s_\theta(\mathbf{x})) + \frac{1}{2} || s_\theta(\mathbf{x}) ||_2^2 \bigg]
\end{equation}
$$

however this won't scale to to large data: trace of the jacobian needs $hw$ grad calls, one for each pixel in image.

### Sliced Score Matching

Use Hutchkinson's trace estimator. Requires around four times more computations due to the auto-differentiation.

## Empirical Bayes

<!-- Check out http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf -->

For a random variable $\mathbf{x} \sim p_\mathbf{x}$ and particular observation $\tilde{\mathbf{x}} \sim p\_{\tilde{\mathbf{x}}}$, empirical Bayes provides an estimator of $\mathbf{x}$ expressed purely in terms of $p({\tilde{\mathbf{x}}})$ that minimises the expected squared error. This estimator can be written as a conditional mean:

$$
\begin{equation}
    \hat{\mathbf{x}}(\tilde{\mathbf{x}}) = \int \mathbf{x} p(\mathbf{x}|\tilde{\mathbf{x}}) d\mathbf{x} = \int \mathbf{x} \frac{p(\mathbf{x},\tilde{\mathbf{x}})}{p(\tilde{\mathbf{x}})} d\mathbf{x}.
    \label{eqn:bayes-estimator}
\end{equation}
$$

Of particular relevance is the case where $\tilde{\mathbf{x}}$ is a noisy observation of $\mathbf{x}$ with covariance $\boldsymbol{\Sigma}$. In this case $p(\tilde{\mathbf{x}})$ can be represented by marginalising out $\mathbf{x}$:

$$
\begin{equation}
    p(\tilde{\mathbf{x}}) = \int \frac{1}{(2\pi)^{d/2}|\det(\boldsymbol{\Sigma})|^{1/2}}\exp \Big( -(\tilde{\mathbf{x}}-\mathbf{x})^T \boldsymbol{\Sigma}^{-1}(\tilde{\mathbf{x}}-\mathbf{x})/2 \Big) p(\mathbf{x}) d\mathbf{x}.
\end{equation}
$$

Differentiating this with respect to $\tilde{\mathbf{x}}$ and multiplying both sides by $\mathbf{\Sigma}$ gives:

$$
\begin{equation}
    \boldsymbol{\Sigma} \nabla_{\tilde{\mathbf{x}}} p(\tilde{\mathbf{x}}) = \int (\mathbf{x} - \tilde{\mathbf{x}}) p(\mathbf{x},\tilde{\mathbf{x}}) d\mathbf{x} = \int \mathbf{x} p(\mathbf{x},\tilde{\mathbf{x}}) d\mathbf{x} - \tilde{\mathbf{x}}p(\tilde{\mathbf{x}}).
\end{equation}
$$

After dividing through by $p(\tilde{\mathbf{x}})$ and combining with Equation \ref{eqn:bayes-estimator} we obtain a closed form estimator of $\mathbf{x}$ (Miyasawa 1961) written in terms of the score function $\nabla \log_{\tilde{\mathbf{x}}} p(\tilde{\mathbf{x}})$ 

$$
\begin{equation}
  \hat{\mathbf{x}}(\tilde{\mathbf{x}}) = \tilde{\mathbf{x}} + \boldsymbol{\Sigma} \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}}).
  \label{eqn:gaussian-empirical-bayes}
\end{equation}
$$

And so we can approximate the score as:

$$
\begin{equation}
  \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}}) = \boldsymbol{\Sigma}^{-1} ( \mathbf{x} - \tilde{\mathbf{x}})
\end{equation}
$$

Which converges as $\boldsymbol{\Sigma}$ converges to $\mathbf{0}$. With our score approximation, we can train using mean squared error as in Equation 1.

Mention that we choose $\Sigma$ to be diagonal.

## Denoising Score Matching

Unlike implicit energy models trained with contrastive divergence that require MCMC sampling during training, score matching allows training without sampling.

Trade offs: training with sampling isn't great, we never truly sample from the energy distribution due to long mixing times. Plus it takes a long time to train due to long sampling times. Have to use tricks like persistent memory or short-run MCMC. However denoising score matching isn't great since our score estimate has large variance.

Training with one $\boldsymbol{\Sigma}$ isn't possible - with large values the accuracy is terrible so we can't get sharp images. With small values the whole space isn't covered so we would have to start our MCMC chain close to true data values therefore we can't sample. 

So train with multiple variance sizes, condition the network on it, and sample ith annealed Langevin dynamics. 

All this seems pretty complicated, right? Let's have a look how it's implemented:

```julia
all_σs = exp.(range(log(1f0), log(0.01f0), length=10))
σs = rand(all_σs, batchsize)
x̃ = x + randn!(similar(x)) .* σs
score = UNet(x̃, σs)
target = (x - x̃) ./ (σs ^ 2)
loss = 0.5f0 .* sumexceptbatch((score - target) .^ 2) .* (σs ^ 2)
```

Wait what? That's it? Pretty much! We choose what standard deviations we want `all_σs`, sample one for each element in our batch `x`, perturb the data points, then compute the loss. 

As the code suggests a UNet is used.


## Denoising Diffusion

Another derivation based on diffusion.

Allows computation of the ELBO.

### Cosine Scheduling

Improved scheduling


## Stochastic Differential Equations

Langevin equation - wait a second that's an SDE. Isn't that nice. 

