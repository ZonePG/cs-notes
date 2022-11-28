# Challenge 6: Quantification

**Definition**: Empirical and theoretical study to better understand heterogeneity, cross-modal interactions, and the multimodal learning process.

![7-01](/MML-Tutorial/images/7-01.png)

## Sub-Challenge 6a: Heterogeneity

**Definition**: Quantifying the dimensions of heterogeneity in multimodal datasets and how they subsequently influence modeling and learning.
- Structure: static, temporal, spatial, hierarchical, invariances
- Representation space: discrete, continuous, interpretable
- Information: entropy, density, information overlap, range
- Precision: sampling rate, resolution, granularity
- Noise: uncertainty, signal-to-noise ratio, missing data
- Relevance: task relevance, context dependence

### Modality Biases

#### Unimodal biases and modality collapse

- Wu et al., Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks. ICML 2022
- Javaloy et al., Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization. ICML 2022
- Goyal et al., Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. CVPR 2017

#### Fairness and social biases – unimodal social biases

Hendricks et al., Women also Snowboard: Overcoming Bias in Captioning Models. ECCV 2018

#### Fairness and social biases – cross-modal interactions worsen social biases

Srinivasan and Bisk, Worst of Both Worlds: Biases Compound in Pre-trained Vision-and-Language Models. NAACL 2022

### Noise Topologies and Robustness

- Liang et al., MultiBench: Multiscale Benchmarks for Multimodal Representation Learning. NeurIPS 2021
- Ngiam et al., Multimodal Deep Learning. ICML 2011
- Srivastava and Salakhutdinov, Multimodal Learning with Deep Boltzmann Machines. JMLR 2014
- Tran et al., Missing Modalities Imputation via Cascaded Residual Autoencoder. CVPR 2017
- Pham et al., Found in Translation: Learning Robust Joint Representations via Cyclic Translations Between Modalities. AAAI 2019

## Sub-Challenge 6b: Cross-modal Interactions

**Definition**: Quantifying the presence and type of cross-modal connections and interactions in multimodal datasets and trained models.

![7-02](/MML-Tutorial/images/7-02.png)

### Quantifying Cross-modal Connections

- Hessel and Lee, Does my multimodal model learn cross-modal interactions? It’s harder to tell than you might think!, EMNLP 2020
- Liang et al., MultiViz: An Analysis Benchmark for Visualizing and Understanding Multimodal Models. arXiv 2022
- Wang et al., M2Lens: Visualizing and Explaining Multimodal Models for Sentiment Analysis. IEEE Trans Visualization and Computer Graphics 2021
  - https://andy-xingbowang.com/m2lens/
- Aflalo et al., VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers. CVPR 2022
  - https://github.com/IntelLabs/VL-InterpreT

### Evaluating Interpretability

Liang et al., MultiViz: A Framework for Visualizing and Understanding Multimodal Models. arXiv 2022

### Challenges

Open challenges:
- Faithfulness: do explanations accurately reflect model’s internal mechanics?
- Usefulness: unclear if explanations help humans
- Disagreement: different interpretation methods may generate different explanations
- Evaluate: how to best evaluate interpretation methods

Chandrasekaran et al., Do explanations make VQA models more predictable to a human? EMNLP 2018

Krishna et al., The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective. arXiv 2022

## Sub-Challenge 6c: Multimodal Learning Process

**Definition**: Characterizing the learning and optimization challenges involved when learning from heterogeneous data.

![7-03](/MML-Tutorial/images/7-03.png)

### Optimization challenges

Wang et al., What Makes Training Multi-modal Classification Networks Hard? CVPR 2020
