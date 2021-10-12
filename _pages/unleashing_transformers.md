---
title: Unleashing Transformers
layout: project
header:
    overlay_image: /assets/images/unleashing_transformers_header.jpg
    overlay_filter: rgba(18,17,20,0.7)
    excerpt: Unleashing Transformers
authors:
    - Sam Bond-Taylor
    - Peter Hessey
    - Hiroshi Sasaki
    - Toby P. Breckon
    - Chris G. Willcocks
contributed_equally:
    - Sam Bond-Taylor
    - Peter Hessey
excerpt: TODO
# abstract: We propose a new generative model capable of synthesising realistic high-resolution images that uses a bidirectional Transformer to provide global context. While Vector Quantized Variational Autoencoders are efficient approachs for representing high resolution images, autoregressive models are used to learn the latent prior which by assuming an ordering over the latents are unable to use global context to approximate conditional probabilities. Our approach solves this problem by instead using an absorbing diffusion model which gradually unmasks the latents over multiple time steps.
links:
    - label: "arXiv"
      icon: "fa fa-book"
      url: ""
    - label: "Code"
      icon: "fa fa-github"
      url: ""
    - label: "Contact"
      icon: "fa fa-envelope-o"
      url: ""
    - label: "Tweet"
      icon: "fa fa-retweet"
      url: ""
permalink: /projects/unleashing-transformers
author_profile: false
---

![image-center](/assets/images/unleashing-transformers-diagram.png){: .align-center}

**Summary:** We propose a new generative model capable of synthesising realistic high-resolution images that uses a bidirectional Transformer to provide global context.

## Abstact
We propose a new generative model capable of synthesising realistic high-resolution images that uses a bidirectional Transformer to provide global context. While Vector Quantized Variational Autoencoders are efficient approachs for representing high resolution images, autoregressive models are used to learn the latent prior which by assuming an ordering over the latents are unable to use global context to approximate conditional probabilities. Our approach solves this problem by instead using an absorbing diffusion model which gradually unmasks the latents over multiple time steps.


![image-left](/assets/images/all_datasets_samples.png){: .align-left .align-left-50}


## Approach
Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Donec mollis. Quisque convallis libero in sapien pharetra tincidunt. Aliquam elit ante, malesuada id, tempor eu, gravida id, odio. Maecenas suscipit, risus et eleifend imperdiet, nisi orci ullamcorper massa, et adipiscing orci velit quis magna. Praesent sit amet ligula id orci venenatis auctor. 

$$
\mathbb{E}_{q(\boldsymbol{x}_{t+1}|\boldsymbol{x}_0)}\big[ D_{KL}[ q(\boldsymbol{x}_t|\boldsymbol{x}_{t+1}, \boldsymbol{x}_0) || p_\theta(\boldsymbol{x}_t|\boldsymbol{x}_{t+1}) ] \big]
$$

Phasellus porttitor, metus non tincidunt dapibus, orci pede pretium neque, sit amet adipiscing ipsum lectus et libero. Aenean bibendum. Curabitur mattis quam id urna. Vivamus dui. Donec nonummy lacinia lorem. Cras risus arcu, sodales ac, ultrices ac, mollis quis, justo. Sed a libero. Quisque risus erat, posuere at, tristique non, lacinia quis, eros.


Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Donec mollis. Quisque convallis libero in sapien pharetra tincidunt. Aliquam elit ante, malesuada id, tempor eu, gravida id, odio. Maecenas suscipit, risus et eleifend imperdiet, nisi orci ullamcorper massa, et adipiscing orci velit quis magna. Praesent sit amet ligula id orci venenatis auctor. Phasellus porttitor, metus non tincidunt dapibus, orci pede pretium neque, sit amet adipiscing ipsum lectus et libero. Aenean bibendum. Curabitur mattis quam id urna. Vivamus dui. Donec nonummy lacinia lorem. Cras risus arcu, sodales ac, ultrices ac, mollis quis, justo. Sed a libero. Quisque risus erat, posuere at, tristique non, lacinia quis, eros.

![image-center](/assets/images/unleashing-transformers-fids.png){: .align-center}

