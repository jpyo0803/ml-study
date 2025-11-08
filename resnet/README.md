# Deep Residual Learning for Image Recognition

**Paper:** [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)

---

## Problem

Deeper neural networks tend to show better performance in image-related tasks.
However, deep neural networks suffer from the **degradation problem**; adding more layers leads to **higher training error**. Such high training error does not stem from overfitting, but rather an optimization difficulty (somehow learning to become identity mapping is difficult) that prevents very deep networks from converging properly

---

## Proposed Idea
ResNet introduces the concept of **residual learning** using **shortcut (skip) connections**.
Instead of learning a direct mapping \( H(x) \), the network learns a **residual function** \( F(x) = H(x) - x\). In other words, the network learns how much the output needs to diverge from the input, instead of the output itself. As the network weights begins with the identity mapping, it easily learns to become the identity mapping if it needs to be the one.  

---

## Results
- **Datasets:** ImageNet, CIFAR-10, and others  
- **Performance:**  
  - ResNet achieved **state-of-the-art accuracy** on ImageNet in 2015.  
  - On CIFAR-10, deeper residual networks (up to 110 layers) outperformed plain counterparts and converged successfully.  
- **Impact:** The residual architecture became the foundation of many modern deep learning models (e.g., ResNeXt, DenseNet, Transformers).

---