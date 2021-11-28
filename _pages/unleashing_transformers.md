---
title: "Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes"
title_short: "Unleashing Transformers"
layout: project
header:
    overlay_image: /assets/images/unleashing_transformers/unleashing-transformers-header.jpg
    overlay_filter: rgba(18,17,20,0.7)
    excerpt: Unleashing Transformers
# authors:
#     - Sam Bond-Taylor
#     - Peter Hessey
#     - Hiroshi Sasaki
#     - Toby P. Breckon
#     - Chris G. Willcocks
# contributed_equally:
#     - Sam Bond-Taylor
#     - Peter Hessey
excerpt: An approach that predicts Vector-Quantized image tokens in parallel to significantly speed up the sampling process.
links:
    - label: "arXiv"
      icon: "fa fa-book"
      url: ""
    - label: "Code"
      icon: "fa fa-github"
      url: "https://github.com/samb-t/unleashing-transformers"
    - label: "Contact"
      icon: "fa fa-envelope"
      url: "mailto:samuel.e.bond-taylor@durham.ac.uk,peter.hessey@durham.ac.uk?cc=hiroshi.sasaki@durham.ac.uk,toby.breckon@durham.ac.uk,christopher.g.willcocks@durham.ac.uk"
    - label: "Tweet"
      icon: "fa fa-retweet"
      url: "https://twitter.com/share?url=https%3A%2F%2Fsamb-t.github.io%2Funleashing-transformers"
permalink: /unleashing-transformers
author_profile: false
google_site_verification: XGMPMp23r5k2e3ICQN4HJx7-mrggkn3COJEAofH1VUU
gallery1:
  - url: /assets/images/unleashing_transformers/churches_5x5.jpg
    image_path: /assets/images/unleashing_transformers/churches_5x5.jpg
    alt: "Random samples from a model trained on LSUN Churches"
  - url: /assets/images/unleashing_transformers/bedroom_samples_5x5.jpg
    image_path: /assets/images/unleashing_transformers/bedroom_samples_5x5.jpg
    alt: "Random samples from a model trained on LSUN Bedroom"
  - url: /assets/images/unleashing_transformers/ffhq_5x5_t=0.85.jpg
    image_path: /assets/images/unleashing_transformers/ffhq_5x5_t=0.85.jpg
    alt: "Random samples from a model trained on FFHQ"
gallery2:
  - url: /assets/images/unleashing_transformers/churches nearest neighbours.jpg
    image_path: /assets/images/unleashing_transformers/churches nearest neighbours.jpg
    alt: "Nearest Neighbours in the training data of samples for LSUN Churches"
  - url: /assets/images/unleashing_transformers/bedroom_nearest_neighbours.jpg
    image_path: /assets/images/unleashing_transformers/bedroom_nearest_neighbours.jpg
    alt: "Nearest Neighbours in the training data of samples for LSUN Bedroom"
  - url: /assets/images/unleashing_transformers/ffhq nearest neighbours.jpg
    image_path: /assets/images/unleashing_transformers/ffhq nearest neighbours.jpg
    alt: "Nearest Neighbours in the training data of samples for LSUN FFHQ"
gallery3:
  - url: /assets/images/unleashing_transformers/inpainting_figure.jpg
    image_path: /assets/images/unleashing_transformers/inpainting_figure.jpg
    alt: "Image inpainting"
  - url: /assets/images/unleashing_transformers/big_bedrooms.jpg
    image_path: /assets/images/unleashing_transformers/big_bedrooms.jpg
    alt: "Big Bedrooms"
  - url: /assets/images/unleashing_transformers/temperature_figure.jpg
    image_path: /assets/images/unleashing_transformers/temperature_figure.jpg
    alt: "Temperature figure"
---

## Summary
We propose a novel parallel token prediction approach for generating Vector-Quantized image representations that allows for significantly faster sampling than autoregressive approaches. During training, tokens are randomly masked in an order-agnostic manner and an unconstrained Transformer learns to predict the original tokens. Our approach is able to generate globally consistent images at resolutions exceeding that of the original training data by applying the network to various locations at once and aggregating outputs, allowing for much larger context regions. Our approach achieves state-of-the-art results in terms of Density and Coverage, and performs competitively on FID whilst offering advantages in terms of both computation and reduced training set requirements.

{% include figure image_path="/assets/images/unleashing_transformers/unleashing-transformers-samples.jpg" alt="Samples" caption="Examples of unconditional samples from our models. Our approach uses a discrete diffusion to generate high quality images optionally larger than the training data (right)." %} {: .align-center}


{% include figure image_path="/assets/images/unleashing_transformers/all_datasets_samples.jpg" alt="Samples" caption="Samples from our models trained on LSUN Churches, FFHQ, and LSUN Bedroom." %} {: .align-left .align-left-50}


## Approach

First, a Vector-Quantized image model compresses images to a compact discrete latent space:

$$
\boldsymbol{z}_q = q(E(\boldsymbol{x})), \quad \hat{\boldsymbol{x}} = G(\boldsymbol{z}_q),
$$

$$
\text{where } q(\boldsymbol{z}_i) = \underset{\boldsymbol{c}_{j} \in \mathcal{C}}{\operatorname{min}}||\boldsymbol{z}_i - \boldsymbol{c}_j||
$$

Subsequently, an absorbing diffusion model learns to model the latent distribution by gradually unmasking latents

$$
p_\theta(\boldsymbol{x}_{0:T}) = p_\theta(\boldsymbol{x}_T) \prod_{t=1}^T p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)
$$

Efficient training is possible by optimising the ELBO,

$$
\mathbb{E}_{q(\boldsymbol{z}_0)} \Bigg[ \sum_{t=1}^T \frac{1}{t} \mathbb{E}_{q(\boldsymbol{z}_t|\boldsymbol{z}_0)} \Big[ \sum_{[\boldsymbol{z}_t]_i=m}\log p_\theta([\boldsymbol{z}_0]_i|\boldsymbol{z}_t)  \Big] \Bigg]
$$

By skipping time steps, sampling can be significantly faster than autoregressive approaches.


{% include figure image_path="/assets/images/unleashing_transformers/unleashing-transformers-diagram.jpg" alt="Diagram" caption="Our approach uses a discrete absorbing diffusion model to represent Vector-Quantized images allowing fast high-resolution image generation. Specifically, after compressing images to an information-rich discrete space, elements are randomly masked and an unconstrained Transformer is trained to predict the original data, using global context to ensure samples are consistent and high quality." %} {: .align-center}

## Evaluation
We evaluate our approach on three high resolution 256x256 datasets, LSUN Churches, LSUN Bedroom, and FFHQ. Below are the quantitative results compared to other approaches. Samples obtained with a temperature value of 1.0 on the LSUN datasets achieves the highest Precision, Density, and Coverage; indicating that the data and sample manifolds have the most overlap. On FFHQ our approach achieves the highest Precision and Recall. Despite using a fraction of the number of parameters compared to other Vector-Quantized image models, our approach achieves substantially lower FID scores. 

By predicting tokens in parallel, faster sampling is possible. Specifically, we use a simple step skipping scheme: evenly skipping a constant number of steps to meet some fixed computational budget. As expected, FID increases with fewer sampling steps. However, the increase in FID is minor relative to the improvement in sampling speed.


{% include figure image_path="/assets/images/unleashing_transformers/unleashing-transformers-tables.png" alt="Table" caption="Quantitative evaluation of our approach on FFHQ, LSUN Bedroom and LSUN Churches. Despite having significantly fewer parameters, our approach achieves lower FID than other Vector-Quantized image modelling approaches." %}


{% include figure image_path="/assets/images/unleashing_transformers/unleashing-transformers-big-churches.jpg" alt="Big Samples" caption="Examples of samples generated by our model that are larger than those observed in the training data. Global consistency is encouraged by applying the denoising network to all parts of the image and aggregating probabilities." %}


{% include gallery id="gallery1" caption="Non-cherry picked samples from our models trained on LSUN Churches, LSUN Bedroom, and FFHQ." %}

{% include gallery id="gallery2" caption="Nearest neighbours for a model trained on various datasets based on LPIPS distance. The left columns contain samples from our model and the right columns contain the nearest neighbours in the training set (increasing in distance from left to right)." %}

{% include gallery id="gallery3" caption="Additional figures from our paper. Left: unlike autoregressive priors, our approach allows internal image regions to edited directly. Middle: unconditional samples from a model trained on LSUN Bedroom larger than images in the training dataset. Right: impact of sampling temperature on diversity, for small temperature changes it is less obvious how bias has changed." %}

## Citation

```bibtex
@article{bond2021unleashing,
  title     = {Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes},
  author    = {Sam Bond-Taylor and Peter Hessey and Hiroshi Sasaki and Toby P. Breckon and Chris G. Willcocks},
  journal   = {arXiv preprint arXiv:2111.12701},
  year      = {2021}
}
```