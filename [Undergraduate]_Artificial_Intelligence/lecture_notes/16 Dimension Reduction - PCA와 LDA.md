# Dimension Reduction: PCA와 LDA

tags: #artificial-intelligence #machine-learning #dimension-reduction #pca #lda #curse-of-dimensionality

관련 노트: [[15 Clustering - K Means 군집화]], [[17 Neural Networks Part 1 - 다층신경망 기초]]

## 핵심 요약

이 강의는 curse of dimensionality를 완화하기 위한 dimension reduction을 다룬다. PCA는 label 없이 variance를 최대한 보존하는 projection을 찾고, LDA는 class label을 사용해 class separability를 보존하는 projection을 찾는다.

## Curse of Dimensionality

Feature dimension이 증가하면 공간의 부피가 급격히 커진다. 같은 granularity로 각 축을 나누면 필요한 sample 수가 지수적으로 증가한다.

문제점:

- 고차원 공간에서 데이터가 희박해진다.
- 거리 기반 방법이 불안정해진다.
- 모델이 쉽게 overfitting된다.
- 계산량과 저장량이 증가한다.

## Dimension Reduction

Dimension reduction은 원래 feature vector `x`를 더 낮은 차원의 vector `y`로 변환한다.

```text
x in R^N -> y in R^M,  M < N
```

목표는 정보 손실을 최소화하면서 더 간단한 표현을 얻는 것이다.

## Feature Selection과 Feature Extraction

| 방법 | 설명 |
|---|---|
| Feature selection | 기존 feature 중 일부 선택 |
| Feature extraction | 기존 feature를 조합해 새로운 feature 생성 |

PCA와 LDA는 feature extraction 방법이다.

## PCA

PCA(Principal Components Analysis)는 데이터의 variance를 가장 많이 보존하는 orthogonal projection 축을 찾는다.

### 목적

```text
projection된 sample들의 variance를 최대화하는 방향 w를 찾는다.
```

첫 번째 principal component는 분산이 가장 큰 방향이고, 두 번째 component는 첫 번째와 직교하면서 남은 분산을 가장 많이 설명하는 방향이다.

## PCA 계산

1. 데이터 평균을 빼서 center한다.
2. covariance matrix를 계산한다.
3. covariance matrix의 eigenvector와 eigenvalue를 구한다.
4. 큰 eigenvalue에 대응하는 eigenvector를 선택한다.
5. 데이터를 선택한 eigenvector span으로 projection한다.

```text
Sigma w = lambda w
```

Eigenvalue가 큰 방향은 데이터 variance를 많이 설명한다. 작은 eigenvalue 방향은 noise나 중복 정보일 수 있다.

## PCA의 한계

PCA는 class label을 사용하지 않는다. 따라서 variance가 큰 방향이 classification에 좋은 방향이라는 보장은 없다. Class를 구분하는 정보가 variance가 작은 방향에 있을 수도 있다.

## LDA

LDA(Linear Discriminant Analysis)는 class separability를 보존하면서 차원을 줄이는 supervised dimension reduction이다.

두 class 문제에서 LDA는 class mean 사이의 거리는 크게, class 내부 scatter는 작게 만드는 projection 방향을 찾는다.

## Fisher Criterion

Fisher linear discriminant는 다음 비율을 최대화한다.

```text
J(w) = between-class scatter / within-class scatter
```

행렬 형태로는 between-class scatter matrix `S_B`와 within-class scatter matrix `S_W`를 사용한다.

```text
maximize (w^T S_B w) / (w^T S_W w)
```

최적화는 generalized eigenvalue problem으로 이어진다.

## LDA의 한계

- Class별 분포가 Gaussian이고 unimodal하다는 가정에 의존한다.
- 구분 정보가 평균 차이가 아니라 분산 차이에 있을 때 실패할 수 있다.
- 복잡한 nonlinear class boundary에는 적합하지 않을 수 있다.

## PCA와 LDA 비교

| 항목 | PCA | LDA |
|---|---|---|
| 학습 종류 | Unsupervised | Supervised |
| label 사용 | 사용 안 함 | 사용 |
| 보존 목표 | 전체 variance | class separability |
| 한계 | 분류에 필요한 방향을 놓칠 수 있음 | 분포 가정과 class 구조에 민감 |

## 시험ㆍ복습 체크포인트

- Curse of dimensionality가 왜 문제인지 설명할 수 있어야 한다.
- PCA의 eigenvalue/eigenvector 해석을 이해해야 한다.
- PCA가 label을 사용하지 않는다는 한계를 말할 수 있어야 한다.
- Fisher criterion에서 between-class와 within-class scatter의 의미를 설명할 수 있어야 한다.
- PCA와 LDA를 목적과 데이터 요구사항 관점에서 비교할 수 있어야 한다.

