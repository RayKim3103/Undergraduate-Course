# 선형분류와 Logistic Regression

tags: #artificial-intelligence #machine-learning #classification #logistic-regression #cross-entropy #softmax

관련 노트: [[07 Regularized Regression Geometry - 정규화회귀의 기하학과 Cross Validation]], [[09 Logistic Regression Part 2 - 분류 Overfitting과 L2 정규화]]

## 핵심 요약

이 강의는 classification 문제와 linear classifier, logistic regression을 설명한다. 핵심은 score를 threshold로 자르는 단순 분류에서 나아가 class probability를 모델링하고, likelihood 또는 cross entropy를 최적화해 classifier를 학습하는 것이다.

## Classification 문제

Classification은 출력이 범주형 label인 문제이다.

예:

- spam filtering
- image classification
- sentiment analysis
- personalized medical diagnosis
- brain activity decoding

Binary classification에서는 `y`가 `+1` 또는 `-1`처럼 두 class 중 하나이다. Multiclass classification에서는 class가 세 개 이상이다.

## Linear Classifier

입력 `x`를 feature vector `h(x)`로 바꾸고, weight vector `w`와 내적해 score를 계산한다.

```text
Score(x) = w^T h(x)
y_hat = sign(Score(x))
```

결정경계는 `Score(x)=0`인 점들의 집합이다. Feature 공간에서 이 결정경계는 hyperplane이다.

## Threshold Classifier의 문제

단어 기반 sentiment classifier에서 positive word와 negative word 목록을 직접 만들면 다음 문제가 생긴다.

- 단어 중요도를 사람이 일일이 정해야 한다.
- 문맥과 조합 효과를 반영하기 어렵다.
- 예측 confidence를 얻기 어렵다.

따라서 weight를 데이터에서 학습하고, score를 probability로 해석하는 방법이 필요하다.

## Odds와 Logit

확률 `p`에 대한 odds는 다음과 같다.

```text
odds = p / (1 - p)
```

logit은 odds의 log이다.

```text
logit(p) = log(p / (1 - p))
```

Logistic regression은 log-odds가 feature의 linear combination이라고 가정한다.

```text
logit(P(y=+1 | x)) = w^T h(x)
```

## Sigmoid Function

Sigmoid는 real-valued score를 0과 1 사이 확률로 바꾼다.

```text
sigmoid(z) = 1 / (1 + exp(-z))
P(y=+1 | x, w) = sigmoid(w^T h(x))
```

score가 0이면 확률은 0.5이고, score가 양수로 커질수록 positive class 확률이 커진다.

## Logistic Regression 학습

학습 데이터의 label이 관측될 likelihood를 최대화한다.

```text
w_hat = argmax_w log P(D | w)
```

Linear regression과 달리 logistic regression은 closed-form solution이 없으므로 gradient ascent 또는 gradient descent로 최적화한다.

## Cross Entropy Loss

Log likelihood를 최대화하는 것은 negative log likelihood를 최소화하는 것과 같다. Binary logistic regression의 cross entropy loss는 다음과 같다.

```text
J(w) =
- 1[y=+1] log P(y=+1 | x,w)
- 1[y=-1] log P(y=-1 | x,w)
```

정답 class에 높은 확률을 주면 loss가 작고, 정답 class에 낮은 확률을 주면 loss가 매우 커진다.

## Entropy, Cross Entropy, KL Divergence

| 개념 | 의미 |
|---|---|
| Entropy | 분포 자체의 불확실성 |
| Cross entropy | 실제 분포를 다른 분포로 encoding할 때의 평균 비용 |
| KL divergence | 두 분포의 차이, extra cost |

분류 학습에서는 model distribution이 true label distribution에 가까워지도록 cross entropy를 줄인다.

## Multiclass Logistic Regression

Class가 여러 개이면 softmax를 사용한다.

```text
P(y=c | x,w) = exp(score_c) / sum_k exp(score_k)
```

각 class마다 score를 만들고, softmax가 모든 class 확률의 합을 1로 정규화한다.

## Accuracy와 Confusion Matrix

Accuracy는 전체 sample 중 맞춘 비율이다.

```text
accuracy = # correct / # total
error = # mistakes / # total
```

하지만 false positive와 false negative의 비용이 다르면 accuracy만으로 모델을 평가하기 어렵다. Confusion matrix는 예측 label과 실제 label의 조합을 보여준다.

## 시험ㆍ복습 체크포인트

- Linear classifier의 score와 decision boundary를 설명할 수 있어야 한다.
- Logistic regression이 class probability를 어떻게 모델링하는지 알아야 한다.
- Odds, logit, sigmoid의 관계를 이해해야 한다.
- Cross entropy loss가 정답 class 확률과 어떻게 연결되는지 설명할 수 있어야 한다.
- Binary와 multiclass logistic regression의 차이를 말할 수 있어야 한다.

