# Neural Networks Part 3: 실전 학습과 정규화

tags: #artificial-intelligence #machine-learning #neural-network #optimization #mini-batch #dropout #regularization

관련 노트: [[18 Neural Networks Part 2 - Backpropagation]], [[20 Convolutional Neural Networks - 합성곱신경망]]

## 핵심 요약

이 강의는 neural network를 실제로 학습할 때 만나는 optimization 문제와 regularization 기법을 다룬다. Learning rate, momentum, adaptive optimizer, mini-batch SGD, activation function 선택, data augmentation, L2 regularization, dropout, early stopping이 핵심이다.

## Neural Network Optimization의 어려움

Perceptron이나 linear model의 일부 목적함수는 convex라 global optimum으로 수렴하기 쉽다. 반면 다층 neural network의 loss surface는 non-convex이며 local minima, saddle point, plateau가 존재한다.

```text
training neural networks = large nonconvex optimization
```

## Learning Rate

Learning rate는 gradient update의 step size이다.

| Learning rate | 결과 |
|---:|---|
| 너무 작음 | 수렴이 느리고 plateau에 오래 머묾 |
| 너무 큼 | optimum을 지나쳐 불안정하거나 발산 |
| 적절함 | loss가 안정적으로 감소 |

실무에서는 여러 learning rate를 시도하고 validation 성능으로 선택한다.

## Adaptive Learning Rate와 Momentum

### Momentum

Momentum은 이전 update 방향을 누적해 현재 gradient와 함께 사용한다.

```text
v <- rho*v - eta*gradient
w <- w + v
```

일관된 방향으로는 빠르게 진행하고, 지그재그 진동은 줄이는 효과가 있다.

### Adaptive Optimizer

Adagrad, RMSProp, Adam은 parameter별 gradient 통계를 사용해 effective learning rate를 조정한다. 이 알고리즘들도 기본 learning rate hyperparameter를 가지므로 tuning이 필요하다.

## Learning Rate Decay

초기에는 큰 learning rate로 빠르게 이동하고, 후반에는 learning rate를 줄여 optimum 근처의 진동을 줄인다.

```text
eta_t decreases over epochs
```

## Mini-Batch SGD

Gradient descent는 전체 dataset을 사용하고, SGD는 sample 하나를 사용한다. Mini-batch SGD는 그 중간이다.

| 방식 | Gradient 계산 |
|---|---|
| Full batch | 전체 데이터 |
| SGD | sample 1개 |
| Mini-batch SGD | 작은 batch |

Mini-batch는 gradient noise와 계산 효율의 균형이 좋아 neural network 학습의 표준이다.

## Activation Functions

### Sigmoid

```text
sigma(x) = 1 / (1 + exp(-x))
```

출력이 `[0,1]`로 제한된다. 입력 절댓값이 크면 gradient가 거의 0이 되어 saturation 문제가 생긴다. 출력이 zero-centered가 아니라 optimization이 느려질 수 있다.

### tanh

출력이 `[-1,1]`이고 zero-centered라 sigmoid보다 나은 경우가 많다. 하지만 큰 입력에서 saturation되는 문제는 남아 있다.

### ReLU

```text
ReLU(x) = max(0,x)
```

양수 영역에서 gradient가 사라지지 않고 계산이 단순하다. 현대 neural network에서 널리 사용된다. 다만 음수 영역에서는 gradient가 0이라 neuron이 죽는 문제가 생길 수 있다.

## Overfitting과 Regularization

Neural network는 매우 복잡한 함수를 표현할 수 있으므로 overfitting 가능성이 크다. Regularization은 학습 문제에 제약을 추가해 일반화 성능을 높이는 기법이다.

## Data Augmentation

원본 데이터를 변형해 학습 sample을 늘린다.

예:

- translation
- rotation
- crop
- color jitter
- random combination

이미지 분류에서는 같은 label을 유지하는 변형을 사용한다.

## L2 Regularization

Loss에 weight magnitude penalty를 더한다.

```text
total loss = data loss + lambda * ||W||_2^2
```

Weight가 지나치게 커지는 것을 막아 모델의 variance를 줄인다.

## Dropout

Dropout은 training 중 각 forward pass에서 일부 neuron 출력을 확률적으로 0으로 만든다.

```text
drop probability = p
```

효과:

- 특정 neuron 조합에 과도하게 의존하는 것을 막는다.
- 여러 subnetworks를 평균내는 ensemble과 비슷한 효과가 있다.
- network가 redundant representation을 배우도록 유도한다.

Test time에는 모든 neuron을 사용하므로 activation scale을 맞춰야 한다.

## Early Stopping

Validation accuracy가 더 이상 좋아지지 않거나 감소하기 시작하면 학습을 멈춘다. 이는 추가 학습으로 training loss는 줄어도 generalization이 나빠지는 시점을 피하기 위한 regularization 방법이다.

## 시험ㆍ복습 체크포인트

- Non-convex loss에서 saddle point와 local minima가 학습을 어렵게 하는 이유를 설명할 수 있어야 한다.
- Learning rate가 너무 크거나 작을 때의 현상을 구분해야 한다.
- Momentum과 adaptive optimizer의 직관을 이해해야 한다.
- Sigmoid, tanh, ReLU의 장단점을 비교할 수 있어야 한다.
- Dropout과 early stopping이 overfitting을 줄이는 원리를 말할 수 있어야 한다.

