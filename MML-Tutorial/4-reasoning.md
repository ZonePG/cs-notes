# Challenge 3: Reasoning

**Definition**: Combining knowledge, usually through multiple inferential steps, exploiting multimodal alignment and problem structure.

![4-01](/MML-Tutorial/images/4-01.png)

![4-02](/MML-Tutorial/images/4-02.png)

## Sub-Challenge 3a: Structure Modeling

**Definition**: Defining or learning the relationships over which reasoning occurs.

![4-03](/MML-Tutorial/images/4-03.png)

### Temporal Structure

**Temporal structure in multi-view sequences**

Key ideas: memory to capture cross-modal interactions across time

![4-04](/MML-Tutorial/images/4-04.png)

- Structuring multimodal memory: ideas from representation fusion, coordination, and fission
  - Rajagopalan et al., Extending Long Short-Term Memory for Multi-View Structured Learning. ECCV 2016
- **Writing**: Coordination function measuring similarity between feature and memory to weight feature:
  - Wang et al., Multimodal Memory Modelling for Video Captioning. CVPR 2018
- **Compose**: Weighted function to compose previous memory and new addition
  - Xiong et al., Dynamic Memory Networks for Visual and Textual Question Answering. arXiv 2016
- **Reading**: Summary function to summarize multimodal information
  - Hazarika et al., ICON: Interactive Conversational Memory Network for Multimodal Emotion Detection. EMNLP 2018

### Hierarchical Structure

Hong et al., Learning to Compose and Reason with Language Tree Structures for Visual Grounding. IEEE TPAMI 2019

![4-05](/MML-Tutorial/images/4-05.png)

### Interactive Structure

**Structure defined through interactive environment**

Main difference from temporal - actions taken at previous time steps affect future states

Integrates multimodality into the reinforcement learning framework

Luketina et al., A Survey of Reinforcement Learning Informed by Natural Language. IJCAI 2019

### Structure Discovery

Xu et al., MUFASA: Multimodal Fusion Architecture Search for Electronic Health Records. AAAI 2021

![4-06](/MML-Tutorial/images/4-06.png)

## Sub-Challenge 3b: Intermediate Concepts

**Definition**: The parameterization of individual multimodal concepts in the reasoning process.

### Neuro-symbolic Concepts

**Hand-crafted concepts based on domain knowledge**

Andreas et al., Neural Module Networks. CVPR 2016

## Sub-Challenge 3c: Inference Paradigm

**Definition**: How increasingly abstract concepts are inferred from individual multimodal evidences.

**Towards explicit inference paradigms**:
- Logical inference: given premises inferred from multimodal evidence, how can one derive logical conclusions?
- Causal inference: how can one determine the actual causal effect of a variable in a larger system?

### Logical Inference

Gokhale et al., VQA-LOL: Visual Question Answering Under the Lens of Logic. ECCV 2020

### Causal Inference

**Causal VQA: does my multimodal model capture causation or correlation?**

Agarwal et al., Towards Causal VQA: Revealing & Reducing Spurious Correlations by Invariant & Covariant Semantic Editing. CVPR 2020

## Sub-Challenge 3d: External Knowledge

**Definition**: The derivation of knowledge in the study of inference, structure, and reasoning.

### External Knowledge: Multimodal Knowledge Graphs

**Knowledge can also be gained from external sources**

Marino et al., OK-VQA: A visual question answering benchmark requiring external knowledge. CVPR 2019

Gui et al., KAT: A Knowledge Augmented Transformer for Vision-and-Language. NAACL 2022

Zhu et al., Building a Large-scale Multimodal Knowledge Base System for Answering Visual Queries. arXiv 2015
