---
layout: single
classes: wide
author_profile: true
title: "Few Shot GANs"
categories: generative-models

read_time: true
comments: true
share: true
related: true

excerpt: Few Shot GANs.

sidebar:
  nav: "gentuts"
---

## Summary

1. Intro: baseline methods e.g. Fine tuning and from scratch
2. Differentiable Augmentations
3. Adversarial attach method
4. Lightweight GAN

## GAN

```julia
function Discriminator(;nc=3, ndf=64)
    act = x -> leakyrelu(x, 0.2f0)
    Chain(
        Conv((4, 4), nc => ndf, act, stride=2, pad=1), # 16x16
        Dropout(0.25),
        Conv((4, 4), ndf => ndf*2, act, stride=2, pad=1), # 8x8
        Dropout(0.25), 
        Conv((4, 4), ndf*2 => ndf*4, act, stride=2, pad=1), # 4x4
        Dropout(0.25), 
        Conv((4, 4), ndf*4 => 1, act, stride=1, pad=0)) # 1x1
end

function Generator(;nc=3, nz=128, ngf=64)
    Chain(
        ConvTranspose((4, 4), nz => ngf*4, stride=1, pad=0), # 4x4
        BatchNorm(ngf*4, relu),
        ConvTranspose((4, 4), ngf*4 => ngf*2, stride=2, pad=1), # 8x8
        BatchNorm(ngf*2, relu),
        ConvTranspose((4, 4), ngf*2 => ngf, stride=2, pad=1), # 16x16
        BatchNorm(ngf, relu),
        ConvTranspose((4, 4), ngf => nc, σ, stride=2, pad=1)) # 32x32
end
```

## Augmentations

```julia
function randbrightness(x)
    magnitude = rand!(similar(x, (1, 1, 1, size(x, 4)))) .- 0.5f0
    return x .+ magnitude
end

function randsaturation(x)
    magnitude = rand!(similar(x, (1, 1, 1, size(x, 4)))) .* 2f0
    μ = mean(x, dims=3) # mean across channels?
    return (x .- μ) .* magnitude .+ μ
end

function randcontrast(x)
    magnitude = rand!(similar(x, (1, 1, 1, size(x, 4)))) .+ 0.5
    μ = mean(x, dims=(1,2,3)) # all but batch dim
    return (x .- μ) .* magnitude .+ μ
end
```
