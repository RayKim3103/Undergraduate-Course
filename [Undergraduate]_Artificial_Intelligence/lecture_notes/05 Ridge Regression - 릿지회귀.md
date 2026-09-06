# Ridge Regression

tags: #artificial-intelligence #machine-learning #ridge-regression #l2-regularization #overfitting #cross-validation

관련 노트: [[04 Linear Regression Part 2 - 성능평가와 Bias Variance]], [[06 Lasso Regression - 라쏘회귀와 Feature Selection]]

## 핵심 요약

이 강의는 많은 feature나 높은 차수 polynomial을 사용할 때 발생하는 overfitting을 줄이기 위해 ridge regression을 도입한다. Ridge regression은 RSS에 L2 penalty를 더해 큰 coefficient를 억제하고, bias-variance tradeoff를 조절하는 regularization 방법이다.

## Overfitting의 증상

높은 차수 polynomial이나 많은 feature를 가진 모델은 training data를 지나치게 잘 맞출 수 있다. 이때 학습된 coefficient가 매우 커지는 경우가 많다.

```text
큰 coefficient -> 입력의 작은 변화에도 예측이 크게 흔들림 -> 높은 variance
```

관측 수 `N`이 작고 feature 수 `D`가 크면 가능한 입력 조합을 충분히 관찰하기 어렵기 때문에 overfitting 위험이 커진다.

## Ridge Regression의 목적함수

Ridge regression은 데이터 적합도와 coefficient 크기를 함께 고려한다.

```text
cost(w) = RSS(w) + lambda * ||w||_2^2
```

| 항 | 의미 |
|---|---|
| `RSS(w)` | training data에 맞추는 정도 |
| `lambda * ||w||_2^2` | coefficient가 커지는 것을 벌점 |
| `lambda` | regularization strength |

`lambda`가 크면 coefficient가 더 작아지고 모델이 단순해진다.

## Bias-Variance 관점

| `lambda` | 모델 특성 | Bias | Variance |
|---:|---|---:|---:|
| 작음 | training data에 많이 맞춤 | 낮음 | 높음 |
| 큼 | coefficient를 강하게 축소 | 높음 | 낮음 |

Ridge는 variance를 줄이는 대신 bias를 약간 증가시키는 방법이다.

## Matrix Form

Linear regression의 RSS는 다음과 같다.

```text
RSS(w) = (y - H w)^T (y - H w)
```

Ridge objective는 다음과 같다.

```text
cost(w) = (y - H w)^T (y - H w) + lambda * w^T w
```

gradient:

```text
gradient cost(w) = -2 H^T (y - H w) + 2 lambda I w
```

closed-form solution:

```text
w_hat = (H^T H + lambda I)^(-1) H^T y
```

`lambda I`가 더해지면 `H^T H`가 singular하거나 ill-conditioned일 때도 안정성이 좋아진다.

## Gradient Descent

Ridge regression은 gradient descent로도 풀 수 있다.

```text
w_j <- w_j - eta * gradient_j
```

regularization term 때문에 각 weight는 데이터 gradient에 의해 갱신되면서도 0 방향으로 shrink된다.

## Gaussian Noise 관점

Linear regression에서 Gaussian noise를 가정하면 least squares는 maximum likelihood와 연결된다. Ridge penalty는 parameter에 Gaussian prior를 둔 MAP estimation으로 해석할 수도 있다. 즉, coefficient가 너무 큰 모델보다 작은 coefficient를 가진 모델을 사전적으로 선호한다.

## Lambda 선택

`lambda`는 training error만 보고 고르면 안 된다. regularization strength는 모델 성능 평가에 직접 영향을 주는 hyperparameter이므로 validation set이나 cross validation으로 선택해야 한다.

전형적인 데이터 분할:

| 데이터 | 용도 |
|---|---|
| Training set | 여러 lambda에서 weight 학습 |
| Validation set | lambda 선택 |
| Test set | 최종 일반화 성능 보고 |

## Intercept Penalty

표준 ridge cost는 모든 coefficient에 penalty를 줄 수 있지만, intercept `w0`에는 penalty를 주지 않는 경우가 많다. intercept는 전체 평균 위치를 조정하는 항이므로 크다고 해서 모델 복잡도가 높다고 해석하기 어렵기 때문이다.

## 시험ㆍ복습 체크포인트

- Ridge objective를 RSS와 L2 penalty로 쓸 수 있어야 한다.
- `lambda`가 커질 때 coefficient, bias, variance가 어떻게 변하는지 설명할 수 있어야 한다.
- Ridge closed-form solution을 linear regression closed-form과 비교할 수 있어야 한다.
- Hyperparameter 선택에 validation set이 필요한 이유를 말할 수 있어야 한다.
- Intercept를 penalize하지 않는 이유를 이해해야 한다.

