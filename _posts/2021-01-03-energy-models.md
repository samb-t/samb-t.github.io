---
layout: single
classes: wide
author_profile: true
title: "Implicit Energy Models"
categories: generative-models

read_time: true
comments: true
share: true
related: true

excerpt: Implicit Energy Models.

sidebar:
  nav: "gentuts"
---

## Summary

1. Boltzmann Machines, RBMs, Belief Nets
2. Langevin Sampling
3. Conditional EBM
4. Short Run MCMC

Initalise MCMC chain with network (Xie) isn't great as amortized generation is prone to mode collapse. A Bengio paper applies an entropy term to the network but not sure how great that is. Probably approximated.