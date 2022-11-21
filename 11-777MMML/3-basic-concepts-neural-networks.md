# Basic Concepts: Neural Network

## Unimodal Basic Representations

The main modalities include visual, luaguage and acoustic.

### Visual Modality

Each pixel is represented in $R^d$, d is the number of colors
(d=3 for RGB)

### Language Modality

**"one-hot" vector** (word-level classification): xi = number of words in dictionary

**“bag-of-word” vector** (document-level classification)

### Acoustic Modality

![3-01](/11-777MMML/images/3-01.png)

### Sensors

Sundaram et al., Learning the signatures of the human grasp using a scalable tactile glove. Nature 2019

Lee et al., Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks. ICRA 2019

### Tables

Bao et al., Table-to-Text: Describing Table Region with Natural Language. AAAI 2018

![3-02](/11-777MMML/images/3-02.png)

### Graphs

Hamilton and Tang, Tutorial on Graph Representation Learning. AAAI 2019

![3-03](/11-777MMML/images/3-03.png)

### Sets

Zaheer et al., DeepSets. NeurIPS 2017, Li et al., Point Cloud GAN. arxiv 2018

![3-04](/11-777MMML/images/3-04.png)

## Machine Learning – Basic Concepts

### Training, Testing and Dataset

- **Dataset**: Collection of labeled samples $D: {𝑥_i, y_i}$
- **Training**: Learn classifier on training set
- **Validation**: Select optimal hyper-parameters
- **Testing**: Evaluate classifier on hold-out test set

### Simple Classifier: Nearest Neighbor

![3-05](/11-777MMML/images/3-05.png)

### Definition of K-Nearest Neighbor

![3-06](/11-777MMML/images/3-06.png)

### Evaluation methods (for validation and testing)

- **Holdout set**: The available data set $D$ is divided into two disjoint subsets,
  - the training set $D_{train}$ (for learning a model)
  - the test set $D_{test}$ (for testing the model)
- **n-fold cross-validation**: The available data is partitioned into $n$ equal-size disjoint subsets.
- **Leave-one-out cross-validation**: This method is used when the data set is very small.

## Linear Classification: Scores and Loss

- Define a (linear) score function
- Define the loss function (possibly nonlinear)
- Optimization

### Score Function

![3-07](/11-777MMML/images/3-07.png)

### Loss Function

![3-08](/11-777MMML/images/3-08.png)

![3-09](/11-777MMML/images/3-09.png)

#### Cross-Entropy Loss (logistic loss)

![3-10](/11-777MMML/images/3-10.png)

#### Hinge Loss (or max-margin loss or Multi-class SVM loss)

![3-11](/11-777MMML/images/3-11.png)

![3-12](/11-777MMML/images/3-12.png)

Loss function is often made up of three parts: $$L=L_{\text {data }}+\lambda_1 L_{\text {regularization }}+\lambda_2 L_{\text {constraints }}$$
- Data term
  - How well our model is explaining/predicting training data (e.g. cross-entropy loss, Euclidean loss) $$\sum_i L_i=-\sum_i \log \left(\frac{e^{f_{y_i}\left(x_i ; W\right)}}{\sum_j e^{f_j\left(x_i ; W\right)}}\right)$$ $$\sum_i L_i=\sum_i\left(y_i-f\left(x_i, W\right)\right)^2$$
- Regularization/Smoothnessterm
  - Prevent the model from becoming too complex. $\|\mathrm{W}\|_2$ for parameters smoothness, $\|\mathrm{W}\|_1$ for parameter sparsity.
  - $\lambda_1$ is a hyper-parameter. Optional, but almost never omitted
- Additionalconstraints
  - Optional and not always used. Help with certain models. Example during lecture 3.2 about coordinated multimodal representation
  - Example of loss functions using constraints: Triplet loss, hinge ranking loss, reconstruction loss

## Neural Networks inference and learning

- Inference (Testing)
  - Use the score function ( $y = f(x; W)$ )
  - Have a trained model (parameters $W$)
- Learning model parameters (Training)
  - Loss function ( $L$ )
  - Gradient
  - Optimization
