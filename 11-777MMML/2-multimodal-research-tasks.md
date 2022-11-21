# Multimodal Research Tasks

![2-09](/11-777MMML/images/2-09.png)

## Affective Computing

Core Challenges: Fusion, Representation, Alignment, Co-Learning

### Common Topics in Affective Computing

- **Affective states**: emotions, moods, and feelings.
  - Anger, Disgust, Fear, Happiness, Sadness, Positivity, Activation, Pride, Desire
  - Frustration, Anxiety, Contempt, Shame, Guilt, Wonder, Relaxation, Pain, Envy
- **Cognitive states**: thinking and information processing
  - Engagement, Interest, Attention, Concentration, Effort, Surprise, Confusion, Agreement, Doubt, Knowledge
  - Pessimistic, Anxious, Moody, Curious, Artistic, Creative, Sincere, Modest, Fair
- **Personality**: patterns of acting, feeling, and thinking
  - Outgoing, Assertive, Energetic, Sympathetic, Respectful, Trusting, Organized, Productive, Responsible
- **Pathology**: health, functioning, and disorders
  - Depression, Anxiety, Trauma, Addiction, Schizophrenia, Antagonism, Detachment, Disinhibition, Negative, Affectivity, Psychoticism
- **Social processes**: groups, cultures, and perception
  - Rapport, Cohesion, Cooperation, Competition, Status, Conflict, Attraction, Persuasion, Genuineness, Culture, Skillfulness

### Audio-visual Emotion Challenge
- 2011/2012 [AVEC 2011 – The First International Audio/Visual Emotion Challenge, B. Schuller et al., 2011]
- 2013/2014 [AVEC 2013 – The Continuous Audio/Visual Emotion and Depression Recognition Challenge, Valstar et al. 2013]
- 2015/2016 [Introducing the RECOLA Multimodal Corpus of Remote Collaborative and Affective Interactions, F. Ringeval et al., 2013]

### Multimodal Sentiment Analysis

- Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos ([MOSI](https://docs.google.com/forms/d/e/1FAIpQLSd8LfYr1AZuxeNBNlRUwl8coSoB52qj53Wd9WTwoWEplC4djQ/viewform?c=0&w=1))
- [CMU-MOSEI](https://github.com/A2Zadeh/CMU-MultimodalSDK): 23,453 annotated video segments from 1,000 distinct speakers and 250 topics.

### Multi-Party Emotion Recognition

- [MELD](https://affective-meld.github.io/): Multi-party dataset for emotion recognition in conversations

### Project Example

#### Select-Additive Learning

[Haohan Wang, Aaksha Meghawat, Louis-Philippe Morency and Eric P. Xing, Select-additive Learning: Improving Generalization In Multimodal Sentiment Analysis, ICME 2017](https://arxiv.org/abs/1609.05244)
- **Research task**: Multimodal sentiment analysis
- **Datasets**: MOSI, YouTube, MOUD
- **Main idea**: Reducing the effect of confounding factors when limited dataset size
![2-01](/11-777MMML/images/2-01.png)

#### Word-Level Gated Fusion

[Minghai Chen, Sen Wang, Paul Pu Liang, Tadas Baltrušaitis, Amir Zadeh, Louis-Philippe Morency, Multimodal Sentiment Analysis with Word-Level Fusion and Reinforcement Learning, ICMI 2017](https://arxiv.org/abs/1802.00924)
- **Research task**: Multimodal sentiment analysis
- **Datasets**: MOSI, YouTube, MOUD
- **Main idea**: Estimating importance of each modality at the word-level in a video. build an interpretable model that estimates modality and temporal importance, and learns to attend to important information
![2-02](/11-777MMML/images/2-02.png)

## Media Description

Core Challenges: Translation, Representation, Alignment, Fusion

Given a media (image, video, audio-visual clips) provide a free form text description.

### Large-Scale Image Captioning Dataset

Microsoft Common Objects in COntext ([MS COCO](https://cocodataset.org/#home))
- 120000 images
- Each image is accompanied with five free form sentences describing it (at least 8 words)
- Sentences collected using crowdsourcing (Mechanical Turk)
- Also contains object detections, boundaries and keypoints

### Video Description and Alignment

**Charade Dataset**: http://allenai.org/plato/charades/

### Large-Scale Description and Grounding Dataset

**Visual Genome Dataset**: https://visualgenome.org/

## Multimodal QA

Task - Given an image and a question, answer the question (http://www.visualqa.org/)

- Multimodal QA dataset 1 – VQA Challenge 2016 and 2017 (C1)
- VQA 2.0
- TVQA (C7)
- [VCR](https://visualcommonsense.com/): Visual Commonsense Reasoning (C8)
- [Social-IQ](https://www.thesocialiq.com/) (A10)

### Project Example

#### Adversarial Attacks on VQA models

[Vasu Sharma, Ankita Kalra, Vaibhav, Simral Chaudhary, Labhesh Patel, Louis-Philippe Morency, Attend and Attack: Attention Guided Adversarial Attacks on Visual Question Answering Models. NeurIPS ViGIL workshop 2018.](https://nips2018vigil.github.io/static/papers/accepted/33.pdf)
- **Research task**: Adversarial Attacks on VQA models
- **Datasets**: VQA
- **Main idea**: Test the robustness of VQA models to adversarial attacks on the image.
- How can we design a targeted attack on images in VQA models, which will help in assessing robustness of existing models? 
- Use fusion over original image and question to generate an **adversarial perturbation map** over the image
![2-03](/11-777MMML/images/2-03.png)

## Multimodal Navigation

Core Challenges: Representation, Fusion, Alignment, Translation

### Navigating in a Virtual House

[Room-2-Room](https://bringmeaspoon.org/): 21,567 open vocabulary, crowd-sourced navigation instructions

### Multiple Step Instructions

[Refer360](Multiple Step Instructions)

### Language meets Games

[Heinrich Kuttler and Nantas Nardelli and Alexander H. Miller and Roberta Raileanu and Marco Selvatici and Edward Grefenstette and Tim Rocktaschel, The Nethack Learning Environment.](https://arxiv.org/abs/2006.13760)

### Project Example

####  Instruction Following

[Devendra Singh Chaplot, Kanthashree Mysore Sathyendra, Rama Kumar Pasumarthi, Dheeraj Rajagopal, Ruslan Salakhutdinov, Gated-Attention Architectures for Task-Oriented Language Grounding. AAAI 2018](https://arxiv.org/abs/1706.07230)
- **Research task**: Task-Oriented Language Grounding in an Environment
- **Datasets**: ViZDoom, based on the Doom video game
- **Main idea**: Build a model that comprehends natural language instructions, grounds the entities and relations to the environment, and execute the instruction.
- **Solution**: Gated attention architecture to attend to instruction and states
![2-04](/11-777MMML/images/2-04.png)

#### Multiagent Trajectory Forecasting

[Seong Hyeon Park, Gyubok Lee, Manoj Bhat, Jimin Seo, Minseok Kang, Jonathan Francis, Ashwin R. Jadhav, Paul Pu Liang, Louis-Philippe Morency, Diverse and Admissible Trajectory Forecasting through Multimodal Context Understanding.ECCV 2020](https://arxiv.org/abs/1706.07230)
- **Research task**: Multiagent trajectory forecasting for autonomous driving
- **Datasets**: Argoverse and Nuscenes autonomous driving datasets
- **Main idea**: Build a model that understands the environment and multiagent trajectories and predicts a set of multimodal future trajectories for each agent.
![2-05](/11-777MMML/images/2-05.png)

## Project Examples, Advice and Support

### Latest List of Multimodal Datasets

![2-06](/11-777MMML/images/2-06.png)
![2-07](/11-777MMML/images/2-07.png)
![2-08](/11-777MMML/images/2-08.png)

### Some Advice About Multimodal Research

- Think more about the research problems, and less about the datasets themselves
  - Aim for generalizable models across several datasets
  - Aim for models inspired by existing research e.g. psychology
- Some areas to consider beyond performance:
  - Robustness to missing/noisy modalities, adversarial attacks
  - Studying social biases and creating fairer models
  - Interpretable models
  - Faster models for training/storage/inference
- Theoretical projects are welcome too – make sure there are also experiments to validate theory
- If you are used to deal with text or speech
  - Space will become an issue working with image/video data
  - Some datasets are in 100s of GB (compressed)
- Memory for processing it will become an issue as well
  - Won’t be able to store it all in memory
- Time to extract features and train algorithms will also become an issue
- Plan accordingly!
  - Sometimes tricky to experiment on a laptop (might need to do it on a subset of data)


