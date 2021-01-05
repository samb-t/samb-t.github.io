---
layout: single
classes: wide
author_profile: true
title: "Normalizing Flows"
categories: generative-models

read_time: true
comments: true
share: true
related: true

excerpt: Introduction to Normalizing Flows and how to implement them in Julia.

sidebar:
  nav: "gentuts"

last_modified_at: 2021-01-04
---

{: .notice--warning}
This is a work in progress

# Introduction

Introduction to Normalizing Flows and how to implement them in Julia.


## Normalizing Flows

Change of variables rule (and application of the chain rule) of a random variable $\mathbf{z} \sim q(\mathbf{z})$
through a function $f \colon \mathbb{R}^d \rightarrow \mathbb{R}^d$ to $\mathbf{z}' = f(\mathbf{z})$:

\begin{equation}
q(\mathbf{z}') = q(\mathbf{z}) \bigg| \det \frac{\partial f^{-1}}{\partial \mathbf{z}'} \bigg| = 
                 q(\mathbf{z}) \bigg| \det \frac{\partial f}{\partial \mathbf{z}} \bigg|^{-1}
\end{equation}

Complex densities can be constructed by composing simple maps and applying the above equation. The density 
$q_K(\mathbf{z}_K)$ obtained by successively transforming a random variable $\mathbf{z}_0$ with distribution $q_0$ 
through a chain of $K$ transformations $f_k$ can be defined as:

$$
\begin{align}
\mathbf{z}_K &= f_K \circ \cdots \circ f_2 \circ f_1(\mathbf{z}_0) \\
\ln q_K(\mathbf{z}_K) &= \ln q_0(\mathbf{z}_0) - \sum_{k=1}^K \ln \bigg| \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \bigg|
\end{align}
$$

<!-- What properties should f satisfy? -->

Each transformation therefore must be sufficiently expressive while being easily invertible and have an 
efficient to compute Jacobian determinant.

## Summary

<!-- 1. How to implement flows with Bijectors.jl
2. Simple toy datasets e.g. moons and gaussians. Possibly MNST & Cifar if can be written nicely...
3. SurVAE. Some toy datasets exhibit regularities e.g. with abs flow... Might seem not very useful but talk
about surjective functions, how the space can be simplified, etc. And it decreases improves NLL by using max
pooling rather than standard splitting. VAEs are a special case of Normalising flows: just one layer where
the inverse is approximated.
4. Neural ODEs (ffjord). Show on toy datasets. Talk about augmented neural ODEs. Only beneficial for toy 
datasets though. Or whenever data doesn't lie on a submanifold.
5. RADs, maybe stochastic normalising flows.
6. VAEs with flows in the middle like IAF
7. Maybe some other flows e.g. residual flows where function just needs to be 1-Lipschitz. Also 1-Lipschitz
attention. Splines. -->

1. Implementing flows with Bijectors.jl.
2. Affine Coupling and other popular flows.
3. Beyond bijections: surjection and stochastic layers.
4. Continuous Normalizing Flows.

Probably split this page into a few subpages e.g. basics, advanced, odes. This summary can discuss what is on
each page.

## Bijectors.jl

In PyTorch it is typical for authors to write their own flow modules from scratch (..., ..., and ... to name a few) 
following no set standard. In julia, however, we have [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) 
which offers a standard `Bijector` type with a plethora of methods, simplifying code writing and sharing. That's not
to say this isn't possible in PyTorch, it just hasn't happened and is less likely to[^julia-sharing]. Beyond that, the 
power of multiple dispatch allows us to use syntax we already know e.g. we can sample a flow with `rand(flow)` and 
get the inverse transformation with `inv(flow)`.

To implement a bijector `b` there are three functions we need to write: the forwards pass `b(x)`, the inverse
`inv(b)(x)`, and  $\log | \det \mathbf{J} |$ `logabsdetjac(b, x)`. To see this in action, lets write our own `Exp` 
bijector for vector inputs:

```julia
struct Exp <: Bijector{1} end
(b::Exp)(x) = exp.(x)
(ib::Inverse{<:Exp})(x) = log.(x)
logabsdetjac(b::Exp, x) = sum(x)
```

Here `Exp <: Bijector{1}` says that `Exp` is a subtype of a 1D `Bijector`. Alternatively, we could subtype our 
`Exp` under `ADBijector` which will brute force the jacobian using automatic differentiation so we don't have to
implement `logabsdetjac`. And that's it! All operations defined on bijectors (see home page for list) now work. 
With that said, we often also want to  manually implement `forward(b, x)` which calculates both the forward pass 
and logabsdetjac, `x ↦ b(x), log｜det J(b, x)｜`, in one go for efficiency.

We can easily compose bijectors the same way we compose functions generally in julia: `b1 ∘ b2`, which is 
equivalent to `x ↦ b1(b2(x))`. Alternatively, we can use composel and composer to compose a list of bijectors
left-to-right and right-to-left, respectively.

<!-- composel and composer are really just x ↦ foldl(∘, x) and x ↦ foldr(∘, x) resp. -->

With our new bijector, we want to transform a simple base distribution, e.g. `dist = MvNormal(2, 1)` to a complex 
one. To do this we call `transformed(dist, b)` which wraps our flow as a `TransformedDistribution`, allowing
sampling `rand(flow)`, likelihood evaluation `logpdf(flow, x)`, etc.

# Some Flows

Introduce and implement some commonly used flows. Task is to design functions that are expressive as possible 
while still being efficiently invertible.

## Affine Transformation

A simple example of an invertible function used in normalizing flows is the affine coupling layer which divides 
its input into two sets, $\mathbf{x}^{(1:d)}$ and $\mathbf{x}^{(d+1:D)}$, then passes one set directly to the 
output and performs an affine transformation on the latter:

$$
\begin{align}
    \mathbf{y}^{(1:d)} &= \mathbf{x}^{(1:d)}, \\
    \mathbf{y}^{(d+1:D)} &= \mathbf{x}^{(d+1:D)} \odot \exp(f_\sigma(\mathbf{x}^{(1:d)})) + f_\mu(\mathbf{x}^{(1:d)}),
\end{align}
$$

where $f_\sigma$ and $f_\mu$ are arbitrarily complex functions that do not need to be invertible. This coupling 
layer is easily inverted as

$$
\begin{align}
    \mathbf{x}^{(1:d)} &= \mathbf{y}^{(1:d)}, \\
    \mathbf{x}^{(d+1:D)} &= (\mathbf{y}^{(d+1:D)} - f_\mu(\mathbf{y}^{(1:d)})) \odot \exp(-f_\sigma(\mathbf{y}^{(1:d)})).
\end{align}
$$


<!-- Could merge both of the above into a single line like in lilwengs blog. -->

Since the jacobian matrix is lower triangular, its log-determinant can be efficiently computed as 
$\sum_1^d \ln | f_\sigma(x^{(i)}) |$. Although stacking parameterising affine transformations with neural 
networks can lead to reasonably complex functions, the transformation is still limited and only affects one 
portion of the inputs at a time therefore requiring a substantial number of parameters in total.

$$
\begin{align}
\mathbf{J} = \begin{bmatrix} 
\mathbb{I}_d & \mathbf{0}_{d \times (D-d)} \\ 
\frac{\partial \mathbf{y}_{d+1:D}}{\partial \mathbf{x}_{1:d}} & \text{diag}(\exp(f_\sigma(\mathbf{x}_{1:d}))) \end{bmatrix}
\end{align}
$$

Jacobian is lower triangular so determinant is the trace (sum of elements on the diagonal)

$$
\begin{equation}
\det(\mathbf{J}) = \prod^{D-d}_{j=1} \exp(f_\sigma(\mathbf{x}_{1:d})_j) = \exp( \sum^{D-d}_{j=1} f_\sigma(\mathbf{x}_{1:d})_j)
\end{equation}
$$



```julia
struct AffineCouplingBijection <: Bijector{1}
    coupling_net
    mask::PartitionMask
    AffineCouplingBijection(net, dim::Int) = new(net, PartitionMask(dim, 1:div(dim, 2)))
end

function forward(b::AffineCouplingBijection, x::AbstractArray)
    x1, x2, x3 = partition(b.mask, x)
    shift, scale = chunk(b.coupling_net(x2), 2, dim=1)
    x1 = @. x1 * exp(scale) + shift
    logabsdetjac = reshape(sum(scale, dims=(1)), :)
    return (rv = combine(b.mask, x1, x2, x3), logabsdetjac = logabsdetjac)
end

function (ib::Inverse{<:AffineCouplingBijection})(z::AbstractArray)
    z1, z2, z3 = partition(ib.mask, z)
    shift, scale = chunk(b.coupling_net(x2), 2, dim=1)
    z1 = @. (z1 - shift) / exp(scale)
    return combine(ib.orig.mask, z1, z2, z3)
end
```

{: .notice--info}
Note that `chunk` isn't defined in any packages used. See Helper functions at the bottom of this page for its definition.

{: .notice--info}
Bijectors.jl has a few methods that can be used to build this affine coupling bijection:`Shift`, `Scale`, and `Coupling`,
however, I found it simpler and more parallelisable to implement this in its entirety as here. If anyone has a nicer
implementation I would love to see it.


## Invertible 1x1 Convolution

A clear problem with coupling layers is that a transformation is only applied to one portion of the inputs. A 
simple solution to this is to permute dimensions after each coupling layer so each successive transform is 
applied to opposite parts of the input. More generally, a permutation can be represented as a $1 \times 1$ 
convolution with equal numbers of input and output channels. The forward pass is therefore written as 
$ \forall\_{i, j} : \mathbf{z}\_{i,j} = \mathbf{W}\mathbf{x}\_{i,j} $, and the inverse transform as 
$\forall\_{i, j} : \mathbf{x}\_{i,j} = \mathbf{W}^{-1}\mathbf{z}\_{i,j}$. The jacobian determinant can thus be
calculated as:

$$
\begin{align}
\forall_{i,j} : \mathbf{J}_{i,j} &= \frac{\partial (\mathbf{W}\mathbf{x}_{i,j})}{\partial \mathbf{x}_{i,j}}
 = \frac{\mathbf{W} \partial \mathbf{x}_{i,j}}{\partial \mathbf{x}_{i,j}} = \mathbf{W} \\
\log \bigg| \det \frac{\partial \texttt{conv2d}(\mathbf{x}; \mathbf{W})}{\partial \mathbf{x}} \bigg| &= 
\log(|\det \mathbf{W}|^{h \cdot w}) = h \cdot w \cdot \log |\det \mathbf{W}|
\end{align}
$$

In the 1D case as described below, this is simply equivalent to a standard linear transform.

```julia
struct InvertibleConv1x1{A} <: Bijector{1}
    weight::A
    InvertibleConv1x1(dim::Int) = new(Flux.glorot_uniform(dim,dim))
end

(b::InvertibleConv1x1)(x::AbstractArray) = reshape(conv(unsqueeze(x, 1), unsqueeze(b.weight,1)), size(x))

logabsdetjac(b::InvertibleConv1x1, x::AbstractArray) = fill(logabsdet(b.weight)[1], size(x,2))

(ib::Inverse{<:InvertibleConv1x1})(z::AbstractArray) = reshape(conv(unsqueeze(x, 1), unsqueeze(inv(ib.orig.weight),1)), size(x))
```

{: .notice--warning}
Probably better to initialise the weights more carefully.


## Putting it all together

There is one more tool we need to be able to train a deep normalizing flow: a normalisation layer. Most of the time,
normalisation layers are just affine transformations where the shift and scale parameters are determined by the data.
Batch normalisation, for instance, calculates shift and scale as the mean and standard-deviation respectively, 
per-dimension over a batch. The inverse can thus be calculated using running estimates of the mean and 
standard-deviation. Bijectors.jl has already implemented this for your convenience `InvertibleBatchNorm`.

```julia
numflows = 6
datadim = 2
numfeatures = 64
transforms = Bijector[]

for i in 1:numflows
    # Coupling Layer
    nn = Chain(Dense(div(datadim, 2), numfeatures, relu), Dense(numfeatures, datadim))
    b1 = AffineCouplingBijection(nn, datadim)
    # BatchNorm
    b2 = InvertibleBatchNorm(datadim)
    # Reverse
    b3 = InvertibleConv1x1(datadim)
    push!(transforms, b1, b2, b3)
end

b = Bijectors.composel(transforms...)
d = MvNormal(zeros(Float32, datadim), ones(Float32, datadim))
flow = transformed(d, b)
```

Moons, checkerboard, etc. plots.


## Splines

Steal code from pull request. Again might not loop though. 
Create a new Bijector containing a neural net that is a spline.

```julia
d = 2; K = 3; B = 2;
b_mv = RationalQuadraticSpline(randn(d, K), randn(d, K), randn(d, K - 1), B)
```


## Residual Flow

Need a working spectral norm (i.e. use Zygote branch with Array mutations supported)


## Invertible Convolutions

## Autoregressive Flows

# Beyond Bijections

Maybe discuss SurVAE first as compositions of bijections, surjections and stochastic layers, then RADs and Stochastic NFs as specific implementations.
Or perhaps instead discuss RADs then say that this is one surjection, and discuss SurVAE. Then discuss stochastic layers (vs a stochastic VAE layer).

'A major caveat of sampling with exactly invertible functions forphysical problems are topological constraints... For example, when trying to map a unimodal 
Gaussian distribution to a bimodal distribution with affine coupling layers, a connection between the modes remains.'

Turing.jl already does variational approaches and MCMC sampling approaches. Wonder if they could be composed easily.

## Surjections

Helful to have surjective layers...

'Interestingly, RAD can be seen to implement a class of inference surjections that rely on partitioning of the data space. 
The partitioning is learned during training, thus allowing learning of expressive inference surjections. However, careful parameterization 
is required for stable gradient-based training.'

'Wu et al. (2020) (Stochastic Normalizing Flows) propose an extended flow framework consisting of bijective and stochastic transformations 
using MCMC transition kernels. Their method utilizes the same computation as in the general formulation in Algorithm 1, but does not consider 
surjective maps or an explicit connection to VAEs. Their work shows that MCMC kernels may also be implemented as stochastic transformations in SurVAE Flows.'

Abs surjection.

```julia
abstract type Surjector{N} <: Bijector{N} end

struct AbsSurjection <: Surjector{1} end

(b::AbsSurjection)(x::AbstractArray) = abs.(x)

logabsdetjac(b::AbsSurjection, x::AbstractArray) = fill!(similar(x, size(x)[end]), 1) .* log(2f0) .* prod(size(x)[1:end-1])

function (ib::Inverse{<:AbsSurjection})(z::AbstractArray)
    s = rand!(Bernoulli(0.5), similar(z))
    return @. (2 * s - 1) * z
end
```

## RADs

RADs: 'Combine piecewise invertible functions with discrete auxiliary variables, selecting which invertible function applies, to describe a deep mixture model.''

Choose RADs at cost of losing asymptotically unbiased sampling

## Augmented Normalizing Flows

Augment with noise e; transform e conditioned on x into z; transform x conditioned on z into y (the output). Instead of maximising the marginal likelihood 
of x’s, they maximise p(x, e), allowing use of an augmented data space. $\log p(x) – \log p(x, e) – H(e) = D\_{KL}(q(e)||p(e|x))$. The log marginal likelihood 
$\log p(x)$ can be monte-carlo approximated by sampling es.

$$
\begin{equation}
p_\pi(x, e) = \mathcal{N}(G_\pi(x, e); 0, I) \bigg| \det \frac{\partial G_\pi(x,e)}{\partial(x, e)} \bigg|
\end{equation}
$$

Really, this is as you would expect. Concat x with noise -> affine coupling -> reverse -> affine coupling. Maximise likelihood that output is normal.

```julia
struct Augment <: Surjector{1} end

(s::Augment)(x::AbstractArray) = vcat(x, randn!(similar(x)))

logabsdetjac(s::Augment, x::AbstractArray) = zeros(eltype(x), size(x,2))

(is::Inverse{<:Augment>}) = selectdim(x, 1, 1:div(size(x, 1), 2))
```

<!-- For logabsdetjac check out maxpool surjector since we maximise the likelihood that the noise is noisy. -->
<!-- Probably not 0, probably vcat(zeros(), logpdf(z)) ??? log p(x, e) = log p(x|z) + log p(z). Not sure... -->

Wonder if we could use Turing variational inference to implement affine layer?


## Stochastic Normalizing Flows

Stochastic Normalizing Flows: Composing bijective layers and stochastic layers is a compromise between fully bijective layers (more expressive) and full MCMC
(much faster than full MCMC).

Choose stochastic normalising flows at cost of increasing runtime (and maybe stability?)

Turing.jl is great for probabilistic programming, implementing MCMC samplers etc.

Might just be able to define a Bijector with an energy model. And forwards & backwards passes are both just somthing like sample(energy, MetropolisHastings())

Not tested it. Not a bijector. Would be better to use Turing pre-defined MCMC but not sure how to batch properly.

```julia
struct MetropolisMCFlow <: Bijector{1}
    energy_model
    nsteps
    stepsize
    MetropolisMCFlow(model, nsteps=1, stepsize=0.01f0) = new(model, nsteps, stepsize)
end

function forward(b::MetropolisMCFlow, x::AbstractArray)
    E0 = b.energy_model(x)
    E = E0
    for i in 1:b.nsteps
        xprop = x + b.stepsize .* randn!(similar(x))
        Eprop = b.energy_model(xprop)
        acc = eltype(x).(rand(eltype(x), 1, size(x, 2)) < exp.(-(Eprop - E)))
        x = @. (1f0 - acc) * x + acc * xprop
        E = @. (1f0 - acc) * E + acc * Eprop
    end
    return x, E - E0
end

(ib::Inverse{<:MetropolisMCFlow})(x::AbstractArray) = forward(ib.orig, x)[1]
```

In 1D without batching it's simple
```julia
model = DensityModel(x -> energy(x)[1])
p1 = StaticProposal(MvNormal(zeros(2), ones(2) .* 0.01))
chain = sample(model, MetropolisHastings(p1), 10, chain_type=Chain, init_params=zeros(2))
```
Can somewhat batch by including `MCMCThreads` and batchsize. But bmm isn't used. Probably have to reimplement some parts like [here](https://turing.ml/dev/docs/for-developers/interface).


# Continuous Normalizing Flows

## FFJORD

Build network that concats time with data every layer. Maybe not necessary to include, too explicit.

```julia
struct ConcatDense
    dense
    ConcatDense(in, out, σ=identity) = new(Dense(in+1, out, σ))
end

(l::ConcatDense)(x::AbstractArray, t) = l.dense(vcat(x, fill!(similar(x[1:1,:]), t)))

applychain(::Tuple{}, x, t) = x
applychain(fs::Tuple, x, t) = applychain(tail(fs), first(fs)(x, t), t)
(c::Chain)(x, t) = applychain(c.layers, x, t)
```

Calculate trace

At test time you need the exact trace so probably better to show brute force here.

```julia
function ODEfunc(diffeq, u, t)
    y = u[1:end-1,:]
    e = randn!(similar(y))
    dy, back = pullback(y -> diffeq(y, t), y)
    e_dzdy = back(e)[1]
    approx_tr_dzdy = sum(e_dzdy .* e, dims=1)
    return dy, -approx_tr_dzdy
end
```

Make a Bijector that integrates through.

```julia
struct CNF <: Bijector{1}
    diffeq
    T
    solver
    abstol
    reltol
    CNF(diffeq; T=1f0, solver=Tsit5(), abstol=1f-5, reltol=1f-5) = new(diffeq, T, solver, abstol, reltol)
end

function forward(b::Union{CNF, Inverse{<:CNF}}, x)
    b = b <: CNF ? b : b.orig
    tspan = b <: CNF ? (0f0, b.T) : (b.T, 0f0)
    input = vcat(x, fill!(similar(x, 1, size(x, 2)), 0))
    prob = ODEProblem(func, input, tspan)
    sol = solve(prob, solver, sensealg=InterpolatingAdjoint(), reltol=b.reltol, abstol=b.abstol)
    return (rv = sol.u[end][1:end-1, :], logabsdetjac = sol.u[end][end, :])
end
```

{: .notice--warning}
Maybe having forward and inverse functions would make it clearer? Even if it would be a lot of repeating. Or better, another way to do it.

## Augmented NODEs

Discrete transformations e.g. affine coupling allow flow paths to pass through each other. Continuous flows can't do this.
So add an extra dimension to flow though. Easy.

## OT-Flow

Problems with CNF:
1. Can require large number of evaluations esp. when the parameters lead to a stiff ODE or dynamics that change quickly in time.
2. Building the jacobian to compute the trace is expensive. Trace estimates reduce this but introduce error.

Think I saw that this was already implemented in DiffEqFlux. Excatly calculates the jacobian in O(d) time, same as trace estimate with 1 
Hutchkinson vector.


# Helper functions

```julia
split(x::AbstractArray, splitsize::Int; dim=1) = (selectdim(x, dim, bot:min(bot+splitsize, size(x, dim))) for bot in 1:splitsize:size(x, dim))
chunk(x::AbstractArray, chunks::Int; dim=1) = split(x, div(size(x, dim), chunks), dim=dim)
```

# Footnotes

[^julia-sharing]: See post on intro to Julia for more on this.
