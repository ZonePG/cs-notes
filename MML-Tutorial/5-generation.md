# Challenge 4: Generation

**Definition**: Learning a generative process to produce raw modalities that reflects cross-modal interactions, structure, and coherence.

![5-01](/MML-Tutorial/images/5-01.png)

## Sub-challenge 4a: Translation

**Definition**: Translating from one modality to another and keeping information content while being consistent with cross-modal interactions.

**DALL·E: Text-to-image translation at scale**: Ramesh et al., Zero-Shot Text-to-Image Generation. ICML 2021

**DALL·E 2: Combining with CLIP, diffusion models**: Ramesh et al., Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv 2022

## Sub-challenge 4b: Summarization

**Definition**: Summarizing multimodal data to reduce information content while highlighting the most salient parts of the input.

**Video summarization**

Palaskar et al., Multimodal Abstractive Summarization for How2 Videos. ACL 2019

## Sub-challenge 4c: Creation

**Definition**: Simultaneously generating multiple modalities to increase information content while maintaining coherence within and across modalities.

Tsai et al., Learning Factorized Multimodal Representations. ICLR 2019

## Model Evaluation & Ethical Concerns

**Open challenges**:
- Modalities beyond text + images or video
- Translation beyond descriptive text and images (beyond corresponding cross-modal interactions)
- Creation: fully multimodal generation, with cross-modal coherence + within modality consistency
- Model evaluation: human and automatic
- Ethical concerns of generative models

Menon et al., PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models. CVPR 2020

Carlini et al., Extracting Training Data from Large Language Models. USENIX 2021

Sheng et al., The Woman Worked as a Babysitter: On Biases in Language Generation. EMNLP 2019
