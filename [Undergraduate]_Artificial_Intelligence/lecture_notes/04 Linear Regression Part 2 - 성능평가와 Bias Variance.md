# 성능평가와 Bias-Variance Tradeoff

tags: #artificial-intelligence #machine-learning #linear-regression #generalization #bias-variance #test-error

관련 노트: [[03 Linear Regression Part 1 - 선형회귀 모델과 학습]], [[05 Ridge Regression - 릿지회귀]]

## 핵심 요약

이 강의는 학습된 regression model을 어떻게 평가할 것인가를 다룬다. Training error는 학습 데이터에 맞춘 정도를 측정하지만 일반화 성능을 과대평가하기 쉽다. Test error는 generalization error의 근사이며, 모델 복잡도에 따른 overfitting과 bias-variance tradeoff를 이해하는 것이 핵심이다.

## Loss Function

Loss function은 실제값 `y`와 예측값 `f_w(x)` 사이의 손실을 수치화한다.

예:

```text
squared error = (y - f_w(x))^2
absolute error = |y - f_w(x)|
```

완벽한 예측이면 loss는 0이다. 실제 문제에서는 loss가 의사결정 비용과 연결되므로 어떤 loss를 선택하는지가 중요하다.

## Training Error

Training error는 학습에 사용한 데이터에서 평균 loss를 계산한 값이다.

```text
Training error = (1/N_train) * sum_i L(y_i, f_w_hat(x_i))
```

모델 복잡도가 커질수록 training error는 보통 감소한다. 높은 차수의 polynomial은 학습 데이터 점을 매우 잘 통과할 수 있기 때문이다.

## Training Error의 한계

학습된 parameter `w_hat`은 training data에 맞춰 선택된다. 따라서 같은 데이터에서 평가한 training error는 실제 새 데이터 성능보다 낙관적이다.

```text
training error는 모델 선택과 평가가 같은 데이터에서 일어나기 때문에 과대평가된다.
```

## Generalization Error

Generalization error는 아직 보지 못한 모든 가능한 데이터에 대한 기대 손실이다.

```text
E_{x,y}[ L(y, f_w_hat(x)) ]
```

현실에서는 전체 데이터 분포를 모르므로 정확히 계산할 수 없다. 따라서 별도 test set으로 근사한다.

## Test Error

Test set은 모델 fitting에 사용하지 않은 데이터이다.

```text
Test error = (1/N_test) * sum_{test} L(y_i, f_w_hat(x_i))
```

좋은 test set은 실제 배포 환경에서 만날 데이터 분포를 대표해야 한다.

## Model Complexity와 Error

모델 복잡도가 낮으면 데이터의 구조를 충분히 표현하지 못해 bias가 크다. 모델 복잡도가 너무 높으면 학습 데이터의 noise까지 맞춰 variance가 커진다.

| 복잡도 | Training error | Test/generalization error |
|---|---|---|
| 너무 낮음 | 높음 | 높음, underfitting |
| 적절함 | 중간 또는 낮음 | 낮음 |
| 너무 높음 | 매우 낮음 | 높음, overfitting |

## Training/Test Split

전체 데이터를 training set과 test set으로 나눈다. Training set은 parameter fitting에 사용하고, test set은 최종 성능 추정에 사용한다.

test point가 너무 적으면 generalization error estimate의 variance가 커진다. 반대로 test set이 너무 크면 training data가 줄어 모델 학습이 약해질 수 있다.

## 세 가지 Error Source

예측 오차는 세 요소로 분해해서 볼 수 있다.

| 요소 | 의미 | 줄이는 방법 |
|---|---|---|
| Noise | 데이터 자체의 불확실성, irreducible error | 모델로 제거 불가 |
| Bias | 평균 모델이 true function에서 벗어난 정도 | 모델 표현력 증가 |
| Variance | training set 변화에 따라 모델이 흔들리는 정도 | regularization, 데이터 증가 |

## Bias-Variance Tradeoff

모델 복잡도가 증가하면 bias는 줄어드는 경향이 있지만 variance는 증가한다. 최적의 모델은 bias와 variance 사이의 균형점에 있다.

```text
Expected prediction error = noise + bias^2 + variance
```

이 분해는 왜 training error만 최소화하는 것이 위험한지 설명한다.

## 시험ㆍ복습 체크포인트

- Loss, training error, test error, generalization error를 구분할 수 있어야 한다.
- Training error가 optimistic한 이유를 설명할 수 있어야 한다.
- Underfitting과 overfitting을 모델 복잡도 관점에서 해석할 수 있어야 한다.
- bias, variance, noise의 의미를 각각 말할 수 있어야 한다.
- bias-variance tradeoff 그래프를 설명할 수 있어야 한다.

