# Basic Concepts: Network Optimization

## Learning model parameters

![4-01](/11-777MMML/images/4-01.png)

## Gradient descent

### Parameter Update Strategies

![4-02](/11-777MMML/images/4-02.png)

### Interpreting learning rates

![4-03](/11-777MMML/images/4-03.png)

## Optimization – Practical Guidelines

### Adaptive Learning Rate

General Idea: Let neurons who just started learning have huge learning rate.

Adaptive Learning Rate is an active area of research:
- Adadelta
- RMSProp
  - cache = decay_rate * cache + (1 - decay_rate) * dx**2
  - x += - learning_rate * dx / (np.sqrt(cache) + eps)
- Adam
  - m = beta1 * m + (1-beta1) * dx
  - v = beta2 * v + (1-beta2) * (dx**2)
  - x += - learning_rate * m / (np.sqrt(v) + eps)

### Regularization

#### Parameter Regularization:

- $L_1$ (Lasso) and $L_2$ (Ridge) are the most famous norms used. Sometimes combined (Elastic)
- Other norms are computationally ineffective.

#### Structural Regularization

- Lots of models can learn everything.
- Go for simpler ones. (Occam’s razor)
- Use task specific models: 
  - CNNs
  - RecNNs
  - LSTMs
  - GRUs

### Co-adaptation

- dropout
- Gaussian dropout: instead of multiplying with a Bernoulli random variable, multiply with a Gaussian with mean 1.
- Swapout: Allow skip-connections to happen

### Multimodal Optimization

Biggest Challenge:
- Data from different sources
- Different networks

Example:
- Question Answering: LSTM(s) connected to a CNN
- Multimodal Sentiment: LSTM(s) fused with MLPs and 3D- CNNs

CNNs work well with high decaying learning rate

LSTMs work well with adaptive methods and normal SGD

MLPs are very good with adaptive methods

How to work with all of them ?
- Pre-training is the most straight forward way:
  - Train each individual component of the model separately
  - Put together and fine tune
- Example: Multimodal Sentiment Analysis

#### Pre-training

![4-04](/11-777MMML/images/4-04.png)

![4-05](/11-777MMML/images/4-05.png)

- In the final stage (5), it is better to not use adaptive methods such as Adam.
  - Adam starts with huge momentum on all the networks parameters and can destroy the effects of pretraining.
  - Simple SGD mostly helpful.
- Initialization from other pre-trained models:
  - VGG for CNNs
  - Language models for RNNs
  - Layer by layer training for MLPs

