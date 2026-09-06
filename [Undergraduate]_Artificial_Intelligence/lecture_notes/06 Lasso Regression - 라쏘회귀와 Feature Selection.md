# Lasso Regression과 Feature Selection

tags: #artificial-intelligence #machine-learning #lasso #l1-regularization #feature-selection #coordinate-descent

관련 노트: [[05 Ridge Regression - 릿지회귀]], [[07 Regularized Regression Geometry - 정규화회귀의 기하학과 Cross Validation]]

## 핵심 요약

이 강의는 feature selection을 위한 lasso regression을 다룬다. Ridge가 coefficient를 작게 만들지만 보통 정확히 0으로 만들지는 않는 반면, lasso는 L1 penalty를 사용해 일부 coefficient를 0으로 만들어 sparse model을 만든다.

## Feature Selection

Feature selection은 많은 입력 feature 중 예측에 중요한 feature만 선택하는 작업이다.

### 필요한 이유

- 모델을 해석하기 쉬워진다.
- 불필요한 feature가 만든 noise를 줄일 수 있다.
- 측정 비용이 높은 feature를 줄일 수 있다.
- overfitting을 완화할 수 있다.

집값 예측에서 dishwasher, bathroom 수, lot size, year 등 수많은 feature가 있을 때 모든 feature가 의미 있는 것은 아니다.

## All Subsets와 Greedy 방법

### All subsets

모든 feature subset을 시도해 가장 좋은 조합을 찾는다. Feature가 `D`개이면 가능한 subset은 `2^D`개라서 feature 수가 커지면 계산이 불가능해진다.

### Greedy algorithms

- Forward stepwise: 빈 모델에서 시작해 하나씩 feature를 추가한다.
- Backward stepwise: 전체 feature에서 시작해 하나씩 제거한다.

Greedy 방법은 빠르지만 전체 최적 subset을 보장하지 않는다.

## Ridge Thresholding의 한계

작은 ridge coefficient를 나중에 0으로 잘라내는 방법은 feature 간 상관관계가 있을 때 불안정하다. Ridge는 correlated feature의 coefficient를 함께 나눠 가지는 경향이 있으므로, 단순 thresholding은 어떤 feature를 남길지 명확한 기준을 주지 못한다.

## Lasso Objective

Lasso는 RSS에 L1 norm penalty를 더한다.

```text
cost(w) = RSS(w) + lambda * ||w||_1
||w||_1 = sum_j |w_j|
```

L1 penalty는 해가 좌표축에 닿기 쉬운 구조를 만들며, 그 결과 일부 coefficient가 정확히 0이 된다.

## Ridge와 Lasso 비교

| 항목 | Ridge | Lasso |
|---|---|---|
| Penalty | L2, `sum w_j^2` | L1, `sum |w_j|` |
| 효과 | coefficient shrinkage | shrinkage + feature selection |
| 해의 형태 | 대부분 nonzero | sparse 가능 |
| 최적화 | 미분 가능 | 0에서 미분 불가능 |

## Coordinate Descent

Lasso objective는 절댓값 때문에 모든 좌표에 대해 단순 gradient를 적용하기 어렵다. Coordinate descent는 한 번에 하나의 weight만 최적화한다.

```text
repeat until convergence:
    for each coordinate j:
        w_j만 바꾸고 나머지 w는 고정
```

좌표 선택은 순차적으로 할 수도 있고, 가장 개선이 큰 좌표를 선택할 수도 있다.

## Feature Normalization

Lasso에서는 feature scale이 매우 중요하다. Scale이 큰 feature는 같은 coefficient라도 예측에 미치는 영향이 다르기 때문이다. 따라서 보통 각 feature를 평균 0, 분산 1 또는 norm 1로 정규화한 뒤 lasso를 적용한다.

## Soft Thresholding

Lasso coordinate update의 핵심은 soft thresholding이다.

```text
rho_j < -lambda/2  -> w_j = (rho_j + lambda/2) / z_j
|rho_j| <= lambda/2 -> w_j = 0
rho_j > lambda/2   -> w_j = (rho_j - lambda/2) / z_j
```

즉, `rho_j`가 충분히 크지 않으면 해당 feature의 weight가 0이 된다. 이것이 lasso가 feature selection을 수행하는 직접적인 메커니즘이다.

## Normalized와 Unnormalized Update

Feature가 normalized되어 있으면 `z_j`가 일정하거나 단순해져 update가 간결해진다. Unnormalized feature에서는 각 feature의 scale을 반영하는 `z_j`가 필요하다.

## Lasso의 영향

Lasso는 통계학, 머신러닝, 전기전자 분야에서 sparse model을 만들기 위한 중요한 도구이다. 특히 많은 feature 중 일부만 실제로 중요한 high-dimensional setting에서 강력하다.

## 시험ㆍ복습 체크포인트

- Feature selection이 왜 필요한지 설명할 수 있어야 한다.
- L1 penalty와 L2 penalty의 차이를 말할 수 있어야 한다.
- Lasso가 coefficient를 정확히 0으로 만들 수 있는 이유를 기하학적으로 이해해야 한다.
- Coordinate descent와 soft thresholding update를 설명할 수 있어야 한다.
- Lasso 전에 feature normalization이 필요한 이유를 알아야 한다.

