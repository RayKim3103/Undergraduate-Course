# Ensemble Learning

tags: #artificial-intelligence #machine-learning #ensemble #bagging #random-forest #boosting #adaboost

관련 노트: [[11 Decision Tree - 결정트리]], [[13 Evaluating Classifiers - Precision Recall]]

## 핵심 요약

이 강의는 여러 약한 classifier를 결합해 강한 classifier를 만드는 ensemble learning을 다룬다. Bagging은 bootstrap sample로 모델을 여러 개 만들어 variance를 줄이고, random forest는 decision tree bagging에 feature randomness를 더한다. Boosting은 이전 모델이 틀린 sample에 더 집중해 순차적으로 classifier를 학습한다.

## Ensemble의 아이디어

약한 learner 하나는 성능이 제한적일 수 있다. 하지만 여러 learner가 서로 다른 오류를 만든다면, 예측을 평균하거나 투표함으로써 더 안정적인 모델을 만들 수 있다.

```text
ensemble prediction = 여러 모델 예측의 결합
```

## Bagging

Bagging은 Bootstrap Aggregation의 줄임말이다.

절차:

1. Training set에서 복원추출로 bootstrap dataset을 여러 개 만든다.
2. 각 dataset에 model을 학습한다.
3. Regression은 평균, classification은 majority vote로 예측한다.

Bagging은 model variance를 줄이는 데 특히 효과적이다. Decision tree처럼 data 변화에 민감한 모델과 잘 맞는다.

## Bootstrap Sample

Bootstrap은 원래 training data에서 같은 크기의 dataset을 복원추출로 만드는 방법이다. 어떤 sample은 여러 번 뽑히고, 어떤 sample은 빠질 수 있다. 각 learner가 조금씩 다른 데이터를 보므로 ensemble diversity가 생긴다.

## Random Forest

Random forest는 decision tree ensemble이다. Bagging에 더해 각 split에서 전체 feature가 아니라 무작위로 선택한 feature subset만 고려한다.

### 효과

- Tree 사이의 correlation을 줄인다.
- 평균 또는 투표의 variance 감소 효과를 키운다.
- 많은 feature가 있는 문제에서 강력하다.

## Boosting

Boosting은 learner를 순차적으로 학습한다. 각 단계에서 이전 learner들이 틀린 sample의 weight를 키워 다음 learner가 어려운 sample에 집중하게 한다.

```text
잘 맞춘 sample -> weight 감소
틀린 sample -> weight 증가
```

Boosting은 bias와 variance를 모두 줄일 수 있지만, 너무 오래 반복하면 결국 overfitting될 수 있다.

## AdaBoost

AdaBoost는 대표적인 boosting algorithm이다.

### 학습 절차

1. 모든 training sample에 같은 weight를 준다.
2. 현재 weight를 고려해 weak classifier를 학습한다.
3. Weighted error를 계산한다.
4. classifier의 coefficient를 계산한다.
5. 틀린 sample의 weight를 증가시키고 맞춘 sample의 weight를 감소시킨다.
6. weight를 normalize한다.
7. 정해진 반복 수 `T`까지 반복한다.

최종 classifier는 weak classifier들의 weighted sum이다.

```text
score(x) = sum_t alpha_t * f_t(x)
y_hat = sign(score(x))
```

## AdaBoost 계수

각 classifier의 weight `alpha_t`는 weighted error가 낮을수록 커진다. 즉, 더 잘 맞춘 weak learner가 최종 decision에 더 큰 영향을 준다.

## Boosting의 Training Error

AdaBoost 이론은 각 weak classifier가 random보다 조금만 나아도 반복이 진행되며 training error가 빠르게 줄어들 수 있음을 보여준다. 실제로 boosting은 training error를 0까지 낮추는 경우가 많다.

## 언제 멈출 것인가

Boosting도 무한정 반복하면 overfitting이 생길 수 있으므로 최대 component 수 `T`를 선택해야 한다. 이 값은 validation set이나 cross validation으로 고른다.

## Bagging과 Boosting 비교

| 항목 | Bagging | Boosting |
|---|---|---|
| 학습 방식 | 병렬적, 독립적 | 순차적 |
| 데이터 weight | bootstrap sample | 틀린 sample에 더 큰 weight |
| 주효과 | variance 감소 | bias 감소와 어려운 sample 집중 |
| 대표 모델 | Random forest | AdaBoost |

## 시험ㆍ복습 체크포인트

- Weak learner를 ensemble로 결합하는 이유를 설명할 수 있어야 한다.
- Bagging과 bootstrap sample의 관계를 이해해야 한다.
- Random forest가 split마다 feature subset을 쓰는 이유를 말할 수 있어야 한다.
- AdaBoost의 sample weight update 직관을 설명할 수 있어야 한다.
- Bagging과 boosting의 차이를 비교할 수 있어야 한다.

