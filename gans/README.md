# Generative Adversarial Networks

**Paper:** [Generative Adversarial Networks (Ian J. Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)

## Problem

이전의 생성 모델들(generative models)은 학습 데이터의 확률 분포, $p_{data}$,의 직접 근사를 시도하였음. 하지만 이러한 확률 분포의 직접 근사 기반 모델의 학습 및 추론 과정은 매우 큰 연산량을 필요로해 실제로는 사용이 어려웠음. 예를 들어, maximum likelihood estimation 기반 생성형 모델들 (e.g. Boltzmann Machines, Variational Autoencoders)는 정규화 상수를 계산(복잡한 적분 연산 필요)하거나 근사 추론을 수행하여야함.

## Proposed Idea

논문에서는 기존 생성 모델들의 문제점들을 해결하기 위해 **Adversarial Training** 프레임워크를 제안함. Adversarial Training 프레임워크는 데이터의 확률 분포를 직접적으로 모델링하는 대신 두 네트워크 **Generator (G)** 와 **Discriminator (D)** 가 Minimax 게임 형태로 경쟁적 학습을 하도록함.
- **Generator (G)** 는 latent vector $z \sim \mathcal{N}(0, I)$을 입력으로 받아 가짜 데이터($ x_{fake} $)를 생성함.
- **Discriminator (D)** 는 진짜 or 가짜 데이터를 입력으로 받아 진짜 데이터일 확률을 출력함. 출력값은 [0, 1] 확률 범위를 갖으며 진짜 데이터라고 판단할수록 1에 가까운 값을 출력

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] $$

학습 과정 중에 **Discriminator** 는 진짜 데이터에 대해 높은 값을, 가짜 데이터에 대해 낮은 값을 출력하도록 학습됨. 반면 **Generator** 는 생성한 샘플이 **D** 로부터 진짜로 판별되도록 학습됨 (**G** 는 **D** 가 항상 0.5를 출력하는 것을 원함). 최종적으로 **G** 는 가짜 데이터의 확률 분포 $p_g(x)$가 진짜 데이터의 확률 분포 $p_{data}(x)$와 동일하게 학습하는 것이 목표.


## Example Result (MNIST)

실험적으로, 간단한 GAN 구조만으로도 MNIST 숫자 이미지를 실제와 유사하게 생성할 수 있음을 보였다.

![Generated MNIST Samples](mnist_result.png)