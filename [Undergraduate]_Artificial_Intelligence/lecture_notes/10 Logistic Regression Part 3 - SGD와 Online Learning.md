# SGD와 Online Learning

tags: #artificial-intelligence #machine-learning #logistic-regression #sgd #online-learning #optimization

관련 노트: [[09 Logistic Regression Part 2 - 분류 Overfitting과 L2 정규화]], [[11 Decision Tree - 결정트리]]

## 핵심 요약

이 강의는 logistic regression 학습을 대규모 데이터로 확장하기 위한 stochastic gradient descent를 다룬다. 전체 데이터 gradient를 매번 계산하는 batch gradient ascent는 느리므로, 한 data point 또는 mini-batch의 gradient로 자주 update하는 SGD를 사용한다. 이어서 streaming data에 적합한 online learning을 설명한다.

## Batch Gradient Ascent의 비용

Logistic regression의 log likelihood gradient는 모든 training data point의 contribution을 합한 것이다.

```text
gradient ll(w) = sum_i gradient_i
```

데이터가 매우 크면 매 update마다 전체 데이터를 훑어야 하므로 계산 비용이 크다. 한 번의 update는 정확하지만 update 횟수가 적어 학습이 느릴 수 있다.

## Stochastic Gradient Ascent

Stochastic gradient ascent는 하나의 data point 또는 작은 mini-batch에서 계산한 gradient로 weight를 갱신한다.

```text
w <- w + eta * gradient_i
```

개별 gradient는 noisy하지만 평균적으로는 전체 gradient 방향을 따른다. 그래서 path는 흔들리지만 자주 update할 수 있어 큰 데이터에서는 빠르게 좋은 해에 접근한다.

## Logistic Regression의 SGD

각 data point `(x_i, y_i)`에 대해 다음을 반복한다.

1. 현재 weight로 `P(y_i | x_i, w)`를 계산한다.
2. 해당 sample의 log likelihood gradient를 계산한다.
3. weight를 update한다.
4. 다음 sample로 넘어간다.

L2 regularization이 있으면 update에 weight decay 항이 포함된다.

```text
w <- w + eta * (sample_gradient - 2*lambda*w)
```

## Batch Gradient와 SGD 비교

| 항목 | Batch gradient | Stochastic gradient |
|---|---|---|
| 한 update 비용 | 큼, 전체 데이터 사용 | 작음, 일부 데이터 사용 |
| 방향 정확도 | 높음 | noisy |
| 대규모 데이터 | 느림 | 효율적 |
| 수렴 경로 | 매끄러움 | 흔들림 |

## 왜 SGD가 동작하는가

전체 gradient는 각 data point가 주는 작은 방향의 합이다. 임의로 선택한 sample의 gradient는 완벽한 방향은 아니지만, 여러 update를 거치면 평균적으로 steepest direction을 따라간다.

학습 후반에는 learning rate를 줄이면 gradient noise로 인한 진동을 줄일 수 있다.

## Online Learning

Online learning은 데이터가 한 번에 모두 주어지지 않고 시간에 따라 도착하는 상황에서 모델을 계속 갱신하는 방식이다.

예:

- 광고 클릭 예측
- 추천 시스템
- 검색 ranking
- 실시간 사용자 행동 모델링

새로운 sample이 들어올 때마다 모델을 update할 수 있으므로, 데이터 분포가 시간에 따라 변하는 문제에 적합하다.

## Learning Rate Scheduling

SGD는 learning rate에 민감하다. 너무 크면 발산하거나 optimum 근처에서 계속 흔들리고, 너무 작으면 학습이 느리다. 따라서 시간이 지날수록 learning rate를 줄이는 decay schedule을 자주 사용한다.

## 시험ㆍ복습 체크포인트

- Batch gradient와 stochastic gradient의 계산 비용 차이를 설명할 수 있어야 한다.
- SGD가 noisy하지만 평균적으로 좋은 방향을 따르는 이유를 이해해야 한다.
- L2-regularized SGD update에 regularization term이 어떻게 들어가는지 말할 수 있어야 한다.
- Online learning이 batch learning과 다른 점을 예시와 함께 설명할 수 있어야 한다.

