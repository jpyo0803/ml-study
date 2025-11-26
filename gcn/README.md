# Semi-Supervised Classification with Graph Convolutional Networks

**Paper:** [Semi-Supervised Classification with Graph Convolutional Networks (Thomas N. Kipf et al., 2017)](https://arxiv.org/abs/1609.02907)

## Problem
그래프는 정점(Node)와 간선(Edge)로 이루어진 자료구조로 두 정점을 연결하는 간선을 통해 두 정점의 관계 정보를 표현하거나 전체 그래프 형상(Topology)를 통해 특정한 정보를 표현할 수 있다. 이러한 그래프 정보를 기반으로 학습된 모델은 특정 노드가 주어졌을 때 어떤 label로 분류할 것인지 분류 문제를 풀거나, 그래프 형상이 주어졌을 때 어떠한 특징을 갖게 될 것인지 예측 문제 등을 푸는 데 활용할 수 있다.

특히 그래프의 일부 노드의 label 정보만 주어졌을 때 분류 문제를 효과적으로 해결할 수 있는 방식에 대한 연구가 기존에 많이 이루어졌다. 기존 방식은 손실 함수에 직접 graph Laplacian regularization term을 두는 방식으로 학습이 이루어졌다.

$$
L = L_{0} + \lambda \cdot L_{\mathrm{reg}}, 
\qquad 
L_{\mathrm{reg}} = \sum_{i,j} A_{ij} \, \| f(X_i) - f(X_j) \|^2 
= f(X)^{T} \Delta f(X)
$$

여기서  
- $A_{ij}$는 인접 행렬,  
- $f(\cdot)$은 미분 가능한 뉴럴 네트워크,  
- $\lambda$는 weighting factor,  
- $X$는 node feature vector의 행렬이다.  

이런 기존 방식은 그래프의 각 간선이 연결하는 두 노드가 유사함을 기본 전제로 하기에, 간선이 유사성이 아닌 다른 의미를 갖는 경우에는 사용이 어렵다는 한계가 있다.

---

## Proposed Idea
본 논문에서는 그래프 구조 정보를 뉴럴 네트워크 모델에 직접 포함하여 표현하는 방식 $f(X, A)$, 즉 Graph Convolutional Networks (GCN)를 제안한다. 라벨이 있는 노드에서 supervision이 GCN layer를 통해 인접 노드 방향으로 전파되어 라벨이 없는 노드도 주변 정보를 통해 표현을 학습하게 된다(semi-supervised).  

즉 **각 노드는 자신의 인접한 노드들과 상호작용하여 출력이 label과 가까워지도록 학습된다.**

GCN의 한 layer는 다음과 같이 표현된다.

여기서  
- $\tilde{A} = A + I_N$ (self-loop 추가),  
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ 는 각 노드의 차수를 담은 대각행렬,  
- $W^{(l)}$ 은 layer $l$의 학습 파라미터,  
- $H^{(l)}$ 은 layer $l$의 입력 ($H^{(0)} = X$).

$\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 부분은 **이웃 정보를 평균내는 smoothing operator**라고 볼 수 있다.

$$
H^{(l+1)} = \sigma \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)
$$

위 수식은 spectral graph convolution 식  
$$
g_{\theta} \star x = U g_{\theta} U^{T} x
$$
로부터 도출된다.

이는 입력 $x$ 를 spatial domain에서 **frequency domain으로 변환하여 convolution을 단순한 곱셈으로 치환**하는 아이디어에서 시작한다.

하지만 이 방식은  
- Laplacian $L$ 의 eigen-decomposition 필요  
- $U$와의 곱이 비용 큼  

이라는 단점이 있어, Chebyshev polynomial을 이용해 근사한 식을 사용한다.

근사된 spectral graph convolution은 다음과 같다:

$$
g_{\theta} \star x \approx \sum_{k=0}^{K} \theta'_k \, T_k(\tilde{L}) x
$$

여기서  
- $K$는 몇 step까지 이웃 노드를 고려할지 결정  
- sparse graph이면 계산은 $O(K|\mathcal{E}|)$  
- eigenvector $U$를 구하는 비용은 $O(N^3)$ 로 매우 큼  

논문에서는 **$K=1$** 일 때의 근사식을 사용한다.  
- $K=1$이면 바로 이웃만 고려  
- 계산량 적음  
- 과도한 global 정보 섞임 방지 → 일반화 성능 향상  

그리고 더 먼 영역을 고려해야 한다면 **layer를 깊게 쌓아** 표현력을 높이면 된다.

