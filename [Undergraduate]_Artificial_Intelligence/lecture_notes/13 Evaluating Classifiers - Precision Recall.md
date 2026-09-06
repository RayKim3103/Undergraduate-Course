# Evaluating Classifiers: Precision과 Recall

tags: #artificial-intelligence #machine-learning #classification #precision #recall #confusion-matrix #pr-curve

관련 노트: [[12 Ensemble Learning - Bagging Random Forest Boosting]], [[14 Instance Based Learning - 최근접이웃과 커널회귀]]

## 핵심 요약

이 강의는 classifier 평가에서 accuracy만으로 충분하지 않은 이유를 설명하고, precision, recall, precision-recall curve를 다룬다. Positive 문장을 찾아 레스토랑 리뷰 홍보에 쓰는 예제를 통해 false positive와 false negative의 비용이 다를 수 있음을 보여준다.

## Accuracy의 한계

Accuracy는 전체 sample 중 맞춘 비율이다.

```text
accuracy = (TP + TN) / total
```

하지만 positive class가 희귀하거나, false positive와 false negative의 비용이 다르면 accuracy는 좋은 평가 지표가 아니다.

예를 들어 긍정 리뷰 문장을 홍보용으로 고르는 상황에서는 부정 문장을 긍정으로 잘못 고르는 false positive가 특히 문제가 될 수 있다.

## Confusion Matrix

Binary classification 결과는 네 가지로 나뉜다.

| 실제 / 예측 | Positive 예측 | Negative 예측 |
|---|---:|---:|
| 실제 Positive | TP | FN |
| 실제 Negative | FP | TN |

| 용어 | 의미 |
|---|---|
| TP | positive를 positive로 맞춤 |
| FP | negative를 positive로 잘못 예측 |
| FN | positive를 negative로 놓침 |
| TN | negative를 negative로 맞춤 |

## Precision

Precision은 positive라고 예측한 것 중 실제 positive의 비율이다.

```text
precision = TP / (TP + FP)
```

Precision이 높다는 것은 모델이 positive라고 말할 때 신뢰할 수 있다는 뜻이다.

## Recall

Recall은 실제 positive 중 모델이 찾아낸 비율이다.

```text
recall = TP / (TP + FN)
```

Recall이 높다는 것은 positive sample을 많이 놓치지 않는다는 뜻이다.

## Precision과 Recall의 Tradeoff

Classifier가 class probability를 출력하면 threshold를 조정할 수 있다.

```text
P(y=positive | x) >= t -> positive
```

Threshold를 높이면 positive 예측이 줄어 precision은 올라갈 수 있지만 recall은 떨어진다. Threshold를 낮추면 많은 sample을 positive로 잡아 recall은 올라가지만 false positive가 늘어 precision이 낮아질 수 있다.

## Precision-Recall Curve

Precision-recall curve는 threshold를 바꾸며 precision과 recall의 변화를 그린 것이다. Curve가 오른쪽 위에 가까울수록 좋은 classifier이다.

완벽한 모델은 높은 recall에서도 precision을 유지한다.

## 모델 비교

여러 classifier를 비교할 때 precision-recall curve 전체를 보거나, 하나의 숫자로 요약해야 할 수 있다. 예를 들어 top-k 결과를 보는 문제에서는 `k=5`에서 precision을 비교할 수 있다.

모델 선택에서 중요한 점은 실제 응용에서 무엇이 더 비싼 오류인지 정하는 것이다.

## Positive Sentiment 예제

레스토랑 리뷰에서 홍보 문구를 찾는 경우:

- Precision 중요: 홍보용으로 뽑은 문장이 실제로 긍정이어야 한다.
- Recall 중요: 좋은 홍보 문구를 많이 놓치지 않아야 한다.

둘 중 무엇을 더 중시할지는 서비스 목적에 따라 달라진다.

## 지금까지의 분류 평가 흐름

1. Baseline으로 random classifier나 majority classifier를 생각한다.
2. Accuracy를 계산한다.
3. Confusion matrix로 오류 유형을 본다.
4. Precision과 recall을 계산한다.
5. Threshold를 바꿔 tradeoff를 분석한다.

## 시험ㆍ복습 체크포인트

- TP, FP, FN, TN을 confusion matrix에서 찾을 수 있어야 한다.
- Precision과 recall 공식을 외우고 의미를 설명할 수 있어야 한다.
- Threshold를 올리거나 내릴 때 precision/recall이 어떻게 변하는지 이해해야 한다.
- Accuracy가 불충분한 예시를 들 수 있어야 한다.

