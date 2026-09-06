# 분류 Overfitting과 L2 정규화

tags: #artificial-intelligence #machine-learning #classification #logistic-regression #regularization #l2

관련 노트: [[08 Logistic Regression Part 1 - 선형분류와 로지스틱회귀]], [[10 Logistic Regression Part 3 - SGD와 Online Learning]]

## 핵심 요약

이 강의는 logistic regression에서 feature complexity가 커질 때 decision boundary가 과하게 복잡해지는 overfitting 문제를 다룬다. 해결책으로 L2 regularization을 log likelihood objective에 추가하고, gradient ascent update가 어떻게 바뀌는지 설명한다.

## Classification Overfitting

2차원 feature에서 linear boundary는 단순한 직선이다. 그러나 quadratic, degree 6, degree 20 feature처럼 고차 feature를 추가하면 decision boundary가 매우 유연해진다.

```text
feature complexity 증가 -> training data에는 잘 맞음 -> test data에서는 성능 악화 가능
```

분류에서도 regression과 마찬가지로 모델 복잡도가 너무 높으면 noise나 우연한 패턴까지 학습한다.

## Logistic Regression Model 복습

```text
Score(x_i) = w^T h(x_i)
P(y_i=+1 | x_i,w) = sigmoid(w^T h(x_i))
```

학습은 log likelihood를 최대화하는 weight를 찾는 문제이다.

```text
max_w ll(w)
```

## Regularized Objective

큰 coefficient를 억제하기 위해 L2 penalty를 추가한다.

```text
max_w ll(w) - lambda * ||w||_2^2
```

또는 minimization 관점에서는 cross entropy에 L2 penalty를 더한다.

```text
min_w cross_entropy(w) + lambda * ||w||_2^2
```

## Lambda의 영향

| `lambda` | 결정경계 | 해석 |
|---:|---|---|
| 작음 | 복잡하고 training data에 민감 | overfitting 가능 |
| 큼 | 부드럽고 단순 | underfitting 가능 |

Degree 20 feature를 쓰더라도 regularization이 강하면 coefficient가 작아져 boundary가 과도하게 흔들리지 않는다.

## Gradient Ascent Update

Regularization이 없는 logistic regression은 log likelihood gradient 방향으로 weight를 증가시킨다. L2 penalty를 넣으면 weight를 0 쪽으로 당기는 항이 추가된다.

```text
w <- w + eta * (gradient ll(w) - 2*lambda*w)
```

intercept에는 penalty를 적용하지 않는 설정도 자주 사용한다.

## L2 Regularization의 직관

각 coefficient가 커질수록 score가 입력 변화에 과민하게 반응한다. L2 penalty는 모든 coefficient를 부드럽게 줄여 복잡한 boundary를 완화한다.

분류에서 regularization은 다음 균형을 만든다.

```text
data likelihood를 높이는 것
coefficient magnitude를 작게 유지하는 것
```

## 시험ㆍ복습 체크포인트

- Logistic regression에서 고차 feature가 overfitting을 일으키는 이유를 설명할 수 있어야 한다.
- L2-regularized logistic regression objective를 쓸 수 있어야 한다.
- `lambda`가 decision boundary와 coefficient 크기에 미치는 영향을 이해해야 한다.
- Gradient update에서 regularization term이 어떤 방향으로 작용하는지 말할 수 있어야 한다.

