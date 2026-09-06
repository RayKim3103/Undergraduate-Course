# Instance-Based Learning: Nearest Neighbor와 Kernel Regression

tags: #artificial-intelligence #machine-learning #knn #kernel-regression #nonparametric #local-model

관련 노트: [[13 Evaluating Classifiers - Precision Recall]], [[15 Clustering - K Means 군집화]]

## 핵심 요약

이 강의는 전역적인 parametric function을 학습하는 대신, query point 근처의 training instance를 사용해 local prediction을 수행하는 instance-based learning을 다룬다. 1-NN, k-NN, weighted k-NN, kernel regression, local linear regression이 핵심이다.

## Global Fit과 Local Fit

Linear regression이나 polynomial regression은 전체 데이터에 하나의 global function을 맞춘다. 반면 instance-based learning은 query가 들어온 뒤 그 주변 data point를 찾아 예측한다.

```text
query x_q -> 가까운 training examples 검색 -> local average 또는 local model
```

부동산 중개인이 비슷한 집의 최근 거래가를 찾아 집값을 추정하는 방식이 좋은 예이다.

## Distance Metric

Nearest neighbor를 찾으려면 similarity 또는 distance를 정의해야 한다. 가장 기본은 Euclidean distance이다.

```text
distance(x, x') = sqrt(sum_j (x[j] - x'[j])^2)
```

Feature scale이 다르면 큰 scale feature가 distance를 지배하므로 scaled Euclidean distance나 feature normalization을 사용한다.

## 1-Nearest Neighbor

1-NN은 query에 가장 가까운 training point 하나의 label 또는 output을 그대로 사용한다.

절차:

1. query와 모든 training point의 거리를 계산한다.
2. 가장 작은 거리의 point를 찾는다.
3. regression은 그 point의 `y`, classification은 그 point의 class를 예측한다.

1-NN은 training data에는 잘 맞지만 noise에 민감하고 decision boundary가 매우 불규칙할 수 있다.

## Voronoi Tessellation

1-NN은 feature space를 각 training point에 가장 가까운 영역으로 나눈다. 이 partition을 Voronoi tessellation이라고 한다. 각 영역 안의 모든 query는 같은 nearest neighbor를 가진다.

## k-Nearest Neighbors

k-NN은 가장 가까운 `k`개 이웃을 사용한다.

Regression:

```text
y_hat = average of y values among k nearest neighbors
```

Classification:

```text
y_hat = majority vote among k nearest neighbors
```

`k`가 작으면 variance가 크고, `k`가 크면 bias가 커진다.

## Weighted k-NN

가까운 이웃일수록 더 큰 weight를 주는 방법이다.

```text
y_hat = sum_j weight_j * y_j / sum_j weight_j
```

단순 평균보다 query와 매우 가까운 point의 영향을 더 크게 반영한다.

## Kernel Regression

Kernel regression은 k개의 neighbor만이 아니라 모든 training point에 kernel weight를 부여한다.

```text
weight_i = Kernel_lambda(distance(x_i, x_q))
y_hat = weighted average
```

Kernel bandwidth `lambda`는 local neighborhood의 폭을 정한다.

| `lambda` | 효과 |
|---:|---|
| 작음 | 가까운 점만 영향, variance 증가 |
| 큼 | 많은 점이 영향, bias 증가 |

Kernel 종류보다 bandwidth 선택이 더 중요할 때가 많다.

## Local Linear Regression

Kernel regression이 query 주변에서 상수 함수를 맞추는 방식이라면, local linear regression은 query 주변 데이터에 locally weighted linear model을 맞춘다. 경계 부근이나 trend가 있는 구간에서 단순 local average보다 나을 수 있다.

## k-NN Classification

Classification에서도 같은 원리를 쓴다. 문서나 이메일을 embedding space에 놓고, query와 가까운 labeled example들의 다수결로 class를 정한다.

이미지에서는 pixel distance를 그대로 쓰는 k-NN이 직관적이지만, raw pixel distance는 semantic similarity를 잘 반영하지 못할 수 있다.

## Curse of Dimensionality

Feature dimension이 커지면 가까운 이웃을 찾기가 어려워진다. 고차원에서는 대부분의 점들이 서로 멀어지고, 충분한 data가 없으면 local neighborhood가 비어 있거나 의미가 약해진다.

## 시험ㆍ복습 체크포인트

- 1-NN과 k-NN의 예측 절차를 설명할 수 있어야 한다.
- `k`와 kernel bandwidth가 bias/variance에 미치는 영향을 이해해야 한다.
- Feature scaling이 distance 기반 방법에서 중요한 이유를 말할 수 있어야 한다.
- Kernel regression과 local linear regression의 차이를 설명할 수 있어야 한다.

