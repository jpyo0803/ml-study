# ResNet: Deep Residual Learning for Image Recognition

**논문:** [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)

## Problem

뉴럴 네트워크는 깊으면 깊을수록 이미지 관련 테스크에 더 좋은 성능을 보임.
그러나 깊은 뉴럴 네트워크는 레이어의 개수를 추가할수록 높은 학습 에러율을 보이는 **degradation problem** 현상을 겪음. 여기서의 높은 학습 에러율의 원인은 과적합(overfitting)이 아니라 네트워크의 깊이로 인한 최적화 어려움으로 인해 발생함.

## Proposed Idea
ResNet에서는 shortcut (skip) connection에 기반한 **residual learning** 을 제안함. 일반적인 네트워크에서처럼 $ H(X) $을 직접 학습하는 대신 $ H(x)=F(x)+x $의 형태로 학습하여, 네트워크가 **잔차(residual)** 인 $F(x)$를 학습하도록 함. 이는 깊은 뉴럴 네트워크의 특정 레이어가 identity mapping ($H(x) = x$)인 경우가 optimal일때 이를 수월하게 찾아냄으로서 매우 깊은 뉴럴 네트워크의 학습이 가능하게함.

이러한 접근을 통해 ResNet은 152층까지 안정적으로 학습되었고, ILSVRC 2015에서 우수한 성능을 기록함