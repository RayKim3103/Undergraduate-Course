# 정규화회귀의 기하학과 Cross Validation

tags: #artificial-intelligence #machine-learning #regularization #ridge #lasso #cross-validation

관련 노트: [[06 Lasso Regression - 라쏘회귀와 Feature Selection]], [[08 Logistic Regression Part 1 - 선형분류와 로지스틱회귀]]

## 핵심 요약

이 강의는 ridge와 lasso 해가 왜 서로 다른 형태를 갖는지 기하학적으로 설명하고, regularization 강도 `lambda`를 선택하기 위한 validation set과 K-fold cross validation을 다룬다.

## Ridge의 기하학

Ridge regression의 목적함수는 다음과 같다.

```text
RSS(w) + lambda * ||w||_2^2
```

이를 constraint 형태로 보면 coefficient vector가 원형 또는 구형 영역 안에 있도록 제한하는 것과 같다.

```text
min RSS(w)
subject to ||w||_2^2 <= c
```

2차원에서 L2 constraint는 원에 가깝다. RSS contour가 이 원과 접하는 지점이 ridge 해가 된다. 원은 모서리가 없으므로 해가 좌표축에 정확히 놓일 가능성이 낮다. 따라서 ridge는 coefficient를 줄이지만 보통 0으로 만들지는 않는다.

## Lasso의 기하학

Lasso의 목적함수는 다음과 같다.

```text
RSS(w) + lambda * ||w||_1
```

Constraint 형태:

```text
min RSS(w)
subject to ||w||_1 <= c
```

2차원에서 L1 constraint는 마름모꼴이다. 마름모의 꼭짓점은 좌표축 위에 있으므로 RSS contour가 꼭짓점에 닿을 가능성이 크다. 이때 어떤 weight는 정확히 0이 된다.

## Ridge와 Lasso 해의 차이

| 관점 | Ridge | Lasso |
|---|---|---|
| 제약 영역 | 원형 | 마름모 |
| coefficient | 작아지지만 대부분 남음 | 일부가 0 |
| 모델 해석 | 모든 feature 활용 | feature selection 가능 |
| correlated feature | weight를 나눠 가질 수 있음 | 하나를 선택하는 경향 |

## Lambda의 의미

`lambda`는 모델이 training data를 맞추는 정도와 단순한 coefficient를 선호하는 정도 사이의 균형을 조절한다.

| `lambda` | 효과 |
|---:|---|
| 작음 | training data에 더 잘 맞춤, overfitting 위험 |
| 큼 | coefficient 축소 강함, underfitting 위험 |

## Validation Set

Training set만으로 `lambda`를 고르면 training error가 가장 낮은 값을 선택하기 쉽다. 따라서 데이터 일부를 validation set으로 분리해 hyperparameter를 선택한다.

```text
training set -> weight 학습
validation set -> lambda 선택
test set -> 최종 성능 평가
```

Test set은 마지막 한 번의 평가를 위해 남겨 두어야 한다.

## K-Fold Cross Validation

데이터가 충분하지 않으면 validation split 하나만으로 성능 추정이 불안정할 수 있다. K-fold cross validation은 데이터를 K개 fold로 나누고, 각 fold를 한 번씩 validation set으로 사용한다.

절차:

1. 데이터를 K개 fold로 나눈다.
2. 하나의 fold를 validation set으로 둔다.
3. 나머지 K-1개 fold로 학습한다.
4. validation error를 기록한다.
5. 모든 fold에 대해 반복한다.
6. 평균 validation error가 가장 낮은 `lambda`를 선택한다.

## Cross Validation의 해석

K-fold cross validation은 데이터 사용 효율을 높인다. 각 sample이 한 번은 validation에, 여러 번은 training에 사용된다. 다만 계산량은 K배 가까이 늘어난다.

## 시험ㆍ복습 체크포인트

- Ridge와 lasso constraint의 기하학적 모양을 그릴 수 있어야 한다.
- Lasso에서 sparse solution이 나오는 이유를 설명할 수 있어야 한다.
- `lambda`가 bias와 variance에 주는 영향을 말할 수 있어야 한다.
- Validation set과 test set의 역할을 구분해야 한다.
- K-fold cross validation 절차를 순서대로 설명할 수 있어야 한다.

