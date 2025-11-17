# Attention Is All You Need

**Paper:** [Attention Is All You Need (Ashish Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

## Problem
기존 sequence-to-sequence 모델(RNN/LSTM 기반 인코더-디코더)들은 다음과 같은 근본적 한계를 가지고 있었음.

1. RNN의 순차적 계산 (sequential computation)
- 입력을 순차적으로만 처리해야했으므로 example 내부 병렬화가 불가능 (GPU underutilization)
- 시퀀스 길이가 길어질수록 연산량 급증

2. 장기 의존성(long-range dependency) 문제
- LSTM/GRU는 장기 의존성 문제를 완전히 해결하지 못함
- 멀리 떨어진 두 토큰의 관계를 모델링하기 위해서는 긴 time-step을 거쳐 hidden vector를 전달해야함 (gradient vanishing/explosion 문제)

3. CNN기반 Seq2Seq(ConvS2S, ByteNet)의 거리 의존성 문제에 대한 한계
- 병렬화는 잘 되지만
- 먼 위치간 정보를 연결하려면 convolution layer 깊이가 늘어남
- ConvS2S는 거리간 연산량이 $O(n)$ 에 비례, ByteNet은 $O(log n)$

4. Attention이 존재했지만 RNN의 보조 도구 수준
- RNN의 순차 구조로 오는 문제점을 해결하진 못함
- 여전히 RNN이 병목 

즉, RNN/CNN 기반 Seq2Seq 모델은 장기 의존성, 병렬화 측면에서 근본적 한계를 지니고 있었으며, 이를 근본적으로 해결하는 새로운 구조가 필요.

## Proposed Idea
본 논문에서는 RNN/CNN을 완전히 제거하고 Attention-mechanism으로만 구성된 Transformer 아키텍처를 제안

1. RNN 제거를 통한 완전한 병렬화: 모든 토큰간 의존관계를 한번에 병렬 계산 (example내에서 병렬화 가능)
2. Self-attention으로 모든 토큰 위치 사이에 의존성을 직접 연결: 토큰간 거리에 상관없이 의존성 계산량 고정 $O(1)$
3. Multi-Head Attention으로 다양한 관계 학습: 여러개의 Attention을 병렬적으로 두어 다양한 관점에서 학습
4. RNN의 순차적 정보 기능을 Positional Encoding으로 대체

정리하자면 RNN 방식은 시간적 정보를 모델 구조 자체가 자연스럽게 처리할 수 있다는 장점을 가진다. 그러나 이러한 순차성은 **병렬화 불가능**, **장기 의존성 처리의 비효율성**이라는 근본적 한계를 초래한다. 본 논문에서는 Seq2Seq 문제를 Transformer 구조를 통해 RNN보다 성능적 연산적 효율성 측면에서 더 효과적으로 해결할 수 있음을 보인다. 
