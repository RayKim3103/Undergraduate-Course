# Decision Tree

tags: #artificial-intelligence #machine-learning #decision-tree #nonparametric #pruning #classification

관련 노트: [[10 Logistic Regression Part 3 - SGD와 Online Learning]], [[12 Ensemble Learning - Bagging Random Forest Boosting]]

## 핵심 요약

이 강의는 nonparametric approach의 대표 예인 decision tree를 다룬다. Decision tree는 feature에 대한 질문을 순서대로 던져 데이터를 분할하고, leaf node에서 class label이나 class probability를 예측한다. Greedy split selection, real-valued threshold, overfitting, pruning이 핵심이다.

## Parametric과 Nonparametric

| 구분 | 설명 |
|---|---|
| Parametric model | 고정된 수의 parameter로 데이터를 요약 |
| Nonparametric approach | 데이터가 많아질수록 모델 복잡도가 함께 커질 수 있음 |

Linear regression과 logistic regression은 parametric model이다. Decision tree, k-NN, kernel method는 nonparametric 성격을 가진다.

## Loan Default 예제

대출 신청자를 safe 또는 risky로 분류하는 문제를 사용한다. Feature는 credit, term, income, age 같은 속성이며, tree는 각 node에서 하나의 질문을 던진다.

예:

```text
Credit이 good인가?
Term이 short인가?
Income이 특정 threshold보다 큰가?
```

## Decision Stump

Decision stump는 한 번만 split하는 깊이 1 tree이다. 각 feature로 나눴을 때 classification error가 얼마나 줄어드는지 비교해 가장 좋은 split을 선택한다.

```text
best feature = argmin_feature classification error after split
```

## Classification Error

Node에서 다수 class로 예측했을 때 틀린 sample 비율이다.

```text
error = # mistakes / # data in node
```

Split의 품질은 child node들의 weighted average error로 평가한다.

## Greedy Decision Tree Learning

Decision tree는 보통 recursive greedy algorithm으로 학습한다.

1. 현재 node에 도달한 데이터를 본다.
2. 가능한 feature split을 평가한다.
3. error를 가장 많이 줄이는 split을 선택한다.
4. child node에 대해 반복한다.
5. stopping condition을 만족하면 leaf로 만든다.

Greedy 방식은 각 단계에서 최선의 split을 고르지만 전체 tree의 전역 최적을 보장하지는 않는다.

## Stopping Conditions

대표적인 종료 조건은 다음과 같다.

- node의 모든 sample이 같은 class이다.
- 더 이상 split할 feature가 없다.
- split해도 error가 충분히 줄지 않는다.
- 최대 depth에 도달한다.
- node sample 수가 너무 작다.

## Real-Valued Feature

Income이나 age처럼 연속값 feature는 threshold split을 사용한다.

```text
h_j(x) <= threshold
h_j(x) > threshold
```

Threshold 후보는 정렬된 feature 값 사이의 중간점으로 만들 수 있다. 각 threshold에서 split error를 계산해 최적 threshold를 선택한다.

## Decision Boundary

Decision tree는 feature space를 축에 평행한 영역들로 나눈다. Logistic regression은 하나의 linear boundary를 만들지만, decision tree는 여러 threshold를 조합해 계단형 decision boundary를 만든다.

## Probability Prediction

Leaf node에서 class probability를 예측할 수 있다.

```text
P(y=c | leaf) = leaf 안의 class c sample 비율
```

예측 class는 probability가 가장 큰 class로 정한다.

## Overfitting

Decision tree를 깊게 만들면 training data를 거의 완벽히 맞출 수 있다. 하지만 leaf가 너무 작아지면 noise까지 외워 test 성능이 나빠진다.

## Pruning

Pruning은 큰 tree를 학습한 뒤 불필요한 split을 제거하는 방법이다.

목적함수:

```text
C(T) = Error(T) + lambda * L(T)
```

| 항 | 의미 |
|---|---|
| `Error(T)` | tree의 예측 오류 |
| `L(T)` | leaf 수 또는 tree complexity |
| `lambda` | 단순함에 대한 penalty |

Split을 유지한 tree보다 제거한 작은 tree의 total cost가 낮으면 prune한다.

## 시험ㆍ복습 체크포인트

- Decision tree가 nonparametric approach인 이유를 설명할 수 있어야 한다.
- Greedy split selection 알고리즘을 순서대로 말할 수 있어야 한다.
- Real-valued feature에서 threshold 후보를 만드는 방법을 이해해야 한다.
- Decision tree overfitting과 pruning objective를 설명할 수 있어야 한다.

