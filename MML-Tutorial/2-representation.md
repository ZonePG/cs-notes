# Challenge 1: Representation

## Sub-Challenge 1a: Representation Fusion

**Definition**: Learn a joint representation that models cross-modal interactions between individual elements of different modalities

![2-01](/MML-Tutorial/images/2-01.png)

### Fusion with Unimodal Encoders

Unimodal encoders can be jointly learned with fusion network, or pre-trained

![2-02](/MML-Tutorial/images/2-02.png)

- Additive Fusion
![2-06](/MML-Tutorial/images/2-06.png)
- Multiplicative Fusion
  - Multiplicative fusion
  - Bilinear Fusion
- Tensor Fusion
  - Zadeh et al., Tensor Fusion Network for Multimodal Sentiment Analysis, EMNLP 2017
![2-03](/MML-Tutorial/images/2-03.png)
- Low-rank Fusion
  - Liu et al., Efficient Low-rank Multimodal Fusion with Modality-Specific Factors, ACL 2018
![2-04](/MML-Tutorial/images/2-04.png)
- High-Order Polynomial Fusion
  - Hou et al., Deep Multimodal Multilinear Fusion with High-order Polynomial Pooling, Neurips 2019
- Gated Fusion
  - Arevalo et al., Gated Multimodal Units for information fusion, ICLR-workshop 2017
  - Tsai et al., Transformer Dissection: A Unified Understanding of Transformer’s Attention via the Lens of Kernel, EMNLP 2019
![2-05](/MML-Tutorial/images/2-05.png)
- Nonlinear Fusion
![2-07](/MML-Tutorial/images/2-07.png)
- Measuring Non-Additive Interactions
  - Projection from nonlinear to additive (using EMAP)
  - Hessel and Lee, Does my multimodal model learn cross-modal interactions? It’s harder to tell than you might think!, EMNLP 2020 → introduced the EMAP method
- Complex Fusion
  - Barnum, et al. “On the Benefits of Early Fusion in Multimodal Representation Learning." arxiv 2022

## Sub-Challenge 1b: Representation Coordination

**Definition**: Learn multimodally-contextualized representations that are coordinated through their cross-modal interactions
- Strong Coordination
- Partial Coordination

### Coordination Function

- Cosine similarity
- Kernel similarity functions
- Canonical Correlation Analysis (CCA)

### Deep Canonically Correlated Autoencoders (DCCAE)

Wang et al., On deep multi-view representation learning, PMLR 2015

![2-08](/MML-Tutorial/images/2-08.png)

### Multi-view Latent “Intact” Space

Xu et al., Multi-View Intact Space Learning, TPAMI 2015

Given multiple views $z_i$ from the same “object”:

![2-09](/MML-Tutorial/images/2-09.png)

- There is an “intact” representation which is complete and not damaged
- The views $z_i$ are partial (and possibly degenerated) representations of the intact representation

### Auto-Encoder in Auto-Encoder Network

Zhang et al., AE2-Nets: Autoencoder in Autoencoder Networks, CVPR 2019

![2-10](/MML-Tutorial/images/2-10.png)

### Gated Coordination

![2-11](/MML-Tutorial/images/2-11.png)

### Coordination with Contrastive Learning

![2-12](/MML-Tutorial/images/2-12.png)

#### Example – Visual-Semantic Embeddings

Kiros et al., Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, NIPS 2014

![2-13](/MML-Tutorial/images/2-13.png)

#### Example – CLIP (Contrastive Language–Image Pre-training)

Radford et al., Learning Transferable Visual Models From Natural Language Supervision, arxiv 2021

![2-14](/MML-Tutorial/images/2-14.png)

## Sub-Challenge 1c: Representation Fission

**Definition**: learning a new set of representations that reflects multimodal internal structure such as data factorization or clustering

### Modality-Level Fission

![2-15](/MML-Tutorial/images/2-15.png)

- Tsai et al., Learning Factoriazed Multimodal Representations, ICLR 2019
- Tsai et al., Self-Supervised Learning from a Multi-View Perspective, ICLR 2021

### Fine-Grained Fission

![2-16](/MML-Tutorial/images/2-16.png)

- Hu et al., Deep Multimodal Clustering for Unsupervised Audiovisual Learning, CVPR 2019

