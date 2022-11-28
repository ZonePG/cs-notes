# Challenge 5: Transference

**Definition: Transfer knowledge between modalities, usually to help the primary modality which may be noisy or with limited resources**

![6-01](/MML-Tutorial/images/6-01.png)

## Sub-Challenge 5a: Transfer via Pretrained Models

**Definition**: Transferring knowledge from large-scale pretrained models to downstream tasks involving the primary modality.

![6-01](/MML-Tutorial/images/6-01.png)

### Transfer via prefix tuning

Tsimpoukelli et al., Multimodal Few-Shot Learning with Frozen Language Models. NeurIPS 2021

### Transfer via representation tuning

Ziegler et al., Encoder-Agnostic Adaptation for Conditional Language Generation. arXiv 2019

Rahman et al., Integrating Multimodal Information in Large Pretrained Transformers. ACL 2020

### Transfer across partially observable modalities

Liang et al., HighMMT: Towards Modality and Task Generalization for High-Modality Representation Learning. arXiv 2022

### Transfer across partially observable modalities

Reed et al., A Generalist Agent. arXiv 2022

## Sub-Challenge 5b: Co-learning via Representation

**Definition**: Transferring information from secondary to primary modality by sharing representation spaces between both modalities.

![6-03](/MML-Tutorial/images/6-03.png)

### Representation coordination: word embedding space for zero-shot visual classification

Socher et al., Zero-Shot Learning Through Cross-Modal Transfer. NeurIPS 2013

### Representation fusion

Zadeh et al., Foundations of Multimodal Co-learning. Information Fusion 2020

## Sub-Challenge 5c: Co-learning via Generation

**Definition**: Transferring information from secondary to primary modality by using the secondary modality as a generation target.

![6-04](/MML-Tutorial/images/6-04.png)

### Bimodal translations

Pham et al., Found in Translation: Learning Robust Joint Representations via Cyclic Translations Between Modalities. AAAI 2019

### Predicting images from corresponding language

Tan and Bansal, Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision. EMNLP 2020

## Many more dimensions of transfer

![6-05](/MML-Tutorial/images/6-05.png)

Open challenges:
- Low-resource: little downstream data, lack of paired data, robustness (next section)
- Beyond language and vision
- Settings where SOTA unimodal encoders are not deep learning e.g., tabular data
- Complexity in data, modeling, and training
- Interpretability (next section)
