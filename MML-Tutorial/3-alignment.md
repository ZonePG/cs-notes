# Challenge 2: Alignment

## Sub-Challenge 2a: Connections

**Definition**: Identifying connections between elements of multiple modalities

![3-01](/MML-Tutorial/images/3-01.png)

- Supervised Approach
- Unsupervised Approach

## Sub-Challenge 2b: Aligned Representations

**Definition**: Model all cross-modal connections and interactions to learn better representations

### Aligned Representations – A Popular Approach

![3-02](/MML-Tutorial/images/3-02.png)

### Aligned Representation – Early Fusion

Li et al., VisualBERT: A Simple and Performant Baseline for Vision and Language, arxiv 2019

### Aligned Representations – Two-Way Directional Alignment

![3-03](/MML-Tutorial/images/3-03.png)

### Multimodal Transformer – Pairwise Cross-Modal

![3-04](/MML-Tutorial/images/3-04.png)

### Cross-Modal Transformer Module (V -> L)

### Example of Two-Way Directional Alignment

- Lu, Jiasen, et al. "Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks." arXiv (August 6, 2019).
- Tan, Hao, and Mohit Bansal. "Lxmert: Learning cross-modality encoder representations from transformers." arXiv (August 20, 2019).

### Aligned Representations with Graph Networks

- First advantage: Does not require all elements to be connected
- Second advantage: Allows different edge functions for modality and temporal connections

#### Modal-Temporal Attention Graph 

Yang et al., MTAG: Modal-Temporal Attention Graph for Unaligned Human Multimo23dal Language Sequences, NAACL 2021

## Sub-Challenge 2c: Segmentation

**Definition**: Handle ambiguity in segmentation and element’s granularity during alignment

### Alignment and Segmentation – A Simple Approach

![3-05](/MML-Tutorial/images/3-05.png)

### Alignment and Segmentation – A Classification Approach

Grave et al., Connectionist Temporal Classification: Labelling Unsegmented Seque26nce Data with Recurrent Neural Networks, ICML 2006

![3-06](/MML-Tutorial/images/3-06.png)

### Representation and Segmentation – Cluster-based Approaches

Hsu et al., HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units, arxiv 2021
