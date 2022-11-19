# Introduction: Research and technical challenges

## What is MultiModal?
### Sensory Modalities

![1-what-is-multiModal](/11-777MMML/images/1-01.png)

### Multimodal Communicative Behaviors

![1-02](/11-777MMML/images/1-02.png)

### Modalitiy and Medium

**Modality**: The way in which something happens or is experienced.
- Modality refers to a certain type of information and/or the representation format in which information is stored.
- Sensory modality: one of the primary forms of sensation, as vision or touch; channel of communication.

**Medium**: A means or instrumentality for storing or communicating
information; system of communication/transmission.
- Medium is the means whereby this information is delivered to the senses of the interpreter.

### Examples of Modalities

- Natural language (both spoken or written)
- Visual (from images or videos)
- Auditory (including voice, sounds and music) 
- Haptics / touch
- Smell, taste and self-motion
- Physiological signals
  - Electrocardiogram (ECG), skin conductance
- Other modalities
  - Infrared images, depth images, fMRI

## A Historical View

Four eras of multimodal research:
- The “behavioral” era (1970s until late 1980s)
  - language and Gestures
- The “computational” era (late 1980s until 2000)
  - Audio-Visual Speech Recognition (AVSR)
  - Multimodal/multisensory interfaces
  - Multimedia Computing
- The “interaction” era (2000 - 2010)
  - Modeling Human Multimodal Interaction: AMI Project, CHIL Project, CALO Project(Siri), SSP Project.
- **The “deep learning” era (2010s until ...)**
  - Representation learning (a.k.a. deep learning)
    - [Multimodal deep learning [ICML 2011]](https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf)
    - Multimodal Learning with Deep Boltzmann Machines [NIPS 2012]
    - Visual attention: [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention [ICML 2015]](https://arxiv.org/abs/1502.03044)
  - Key enablers for multimodal research:
    - New large-scale multimodal datasets
    - Faster computer and GPUS
    - High-level visual features
    - “Dimensional” linguistic features

## Core Challenges in “Deep” Multimodal ML

**MultiModal Machine Learning: A Survey and Taxonomy**, 2017
- https://arxiv.org/abs/1705.09406
- 5 core challenges
- 37 taxonomy classes
- 253 referenced citations

### Five Multimodal Core Challenges

![1-03](/11-777MMML/images/1-03.png)

**Representation**: Learning how to represent and summarize multimodal data in away that exploits the complementarity and redundancy.
- Audio-visual speech recognition [Ngiam et al., ICML 2011]: Bimodal Deep Belief Network
- Image captioning [Srivastava and Salahutdinov, NIPS 2012]: Multimodal Deep Boltzmann Machine
- Audio-visual emotion recognition [Kim et al., ICASSP 2013]: Deep Boltzmann Machine 
- [Kiros et al., Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, 2014]: Multimodal Vector Space Arithmetic

![1-04](/11-777MMML/images/1-04.png)

**Alignment**: Identify the direct relations between (sub)elements from two or more different modalities.
- Karpathy et al., Deep Fragment Embeddings for Bidirectional Image Sentence Mapping, https://arxiv.org/pdf/1406.5679.pdf

![1-05](/11-777MMML/images/1-05.png)

**Translation**: Process of changing data from one modality to another, where the translation relationship can often be open-ended or subjective.
- Marsella et al., Virtual character performance from speech, SIGGRAPH/Eurographics Symposium on Computer Animation, 2013
- [Ahuja, C., & Morency, L. P. (2019). Language2Pose: Natural Language Grounded Pose Forecasting. Proceedings of 3DV Conference](https://arxiv.org/abs/1907.01108)

![1-06](/11-777MMML/images/1-06.png)

**Fusion**: To join information from two or more modalities to perform a prediction task.

![1-07](/11-777MMML/images/1-07.png)
![1-08](/11-777MMML/images/1-08.png)

**Co-Learning**: Transfer knowledge between modalities, including their representations and predictive models.
- Pham et al., Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities, https://arxiv.org/abs/1812.07809

![1-09](/11-777MMML/images/1-09.png)

### Taxonomy of Multimodal Research

![1-10](/11-777MMML/images/1-10.png)

## Course Project Timeline

- Pre-proposal (one week)
  - Define dataset, research task
- First project assignment (one month)
  - Experiment with unimodal representations
  - Study prior work on selected research topic
- Midterm project assignment (one month)
  - Implement and evaluate state-of-the-art model(s)
  - Discuss new multimodal model(s)
- Final project assignment (one month)
  - Implement and evaluate new multimodal model(s)
  - Discuss results and possible future directions