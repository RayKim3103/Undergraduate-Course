# 머신러닝 개요

tags: #artificial-intelligence #machine-learning #overview #regression #classification #clustering

관련 노트: [[02 Point Estimation - 점추정과 MLE]]

## 핵심 요약

이 강의는 인공지능과 머신러닝의 역사, 머신러닝의 정의, 대표 문제 유형을 소개한다. 핵심 메시지는 컴퓨터가 명시적으로 모든 규칙을 프로그래밍받는 대신, 데이터에서 패턴을 학습해 예측이나 의사결정을 수행한다는 것이다.

## 역사적 흐름

| 시기 | 사건 | 의미 |
|---|---|---|
| 1936 | Turing machine | 계산 가능성과 알고리즘의 이론적 기반 |
| 1950 | Turing test | 기계 지능을 판별하는 사고 실험 |
| 1956 | Dartmouth conference | Artificial Intelligence라는 분야의 출발점 |
| 1957 | Perceptron | 초기 신경망 모델 |
| 1969-1985 | AI winter | perceptron 한계와 기대 하락 |
| 1986 이후 | MLP와 backpropagation | 다층 신경망 학습의 재부상 |
| 2006 이후 | Deep learning | 깊은 신경망과 대규모 데이터의 결합 |
| 2012 이후 | AlexNet, ImageNet | 현대 deep learning 폭발의 계기 |

## 머신러닝의 정의

머신러닝은 경험, 즉 데이터를 통해 성능이 개선되는 알고리즘을 연구하는 분야이다. 더 실용적으로는 이미지, 비디오, 텍스트, 음성, 센서 데이터 같은 입력에서 예측값이나 결정을 만들어내는 방법이다.

```text
Data -> Learning algorithm -> Intelligence
```

여기서 intelligence는 규칙 기반 프로그램이라기보다 학습된 함수, 분류기, 검색 모델, 군집 구조 등을 의미한다.

## 머신러닝 파이프라인

1. 데이터를 수집한다.
2. 입력 `x`와 출력 `y`를 정의한다.
3. 모델 구조를 정한다.
4. 학습 알고리즘으로 모델 파라미터를 추정한다.
5. 새로운 데이터에 대해 예측이나 결정을 수행한다.
6. 성능을 평가하고 개선한다.

## 대표 문제 유형

### Regression

Regression은 출력 `y`가 연속값인 문제이다.

예:

- 집 면적, 위치, 방 개수로 집값 예측
- 주식 가격 예측
- 트윗이 얼마나 많이 retweet될지 예측
- 뇌 영역 intensity로 연속적인 인지 상태 예측

핵심 질문은 `x`가 주어졌을 때 실수값 `y`를 얼마나 정확히 예측할 수 있는가이다.

### Classification

Classification은 출력 `y`가 범주형 label인 문제이다.

예:

- 이메일을 spam/not spam으로 분류
- 이미지를 고양이, 자동차, 사람 등으로 분류
- 의료 진단에서 질병 종류 예측
- 감성 분석에서 문장을 positive/negative로 분류

출력 범주가 두 개이면 binary classification, 세 개 이상이면 multiclass classification이다.

### Retrieval

Retrieval은 query와 관련 있는 item을 찾는 문제이다. 문서 검색에서는 문서를 similarity space에 배치하고 query와 가까운 문서를 반환한다.

예:

- 검색 엔진
- 유사 이미지 검색
- 뉴스 기사 추천
- 문서 집합에서 관련 문서 찾기

### Clustering

Clustering은 label 없이 데이터 내부의 유사한 그룹을 찾는 비지도학습 문제이다.

예:

- 이미지를 바다, 산, 도시처럼 시각적으로 유사한 그룹으로 묶기
- 웹사이트 사용자를 행동 패턴별로 묶기
- 문서를 주제별로 자동 구조화하기

### Embedding과 표현 학습

Embedding은 복잡한 데이터를 similarity가 보존되는 vector space로 옮기는 표현이다. 이미지나 문서가 embedding 공간에서 가까우면 의미적으로도 비슷하다고 해석할 수 있다.

## Deep Learning의 확산

2012년 이후 deep learning은 image classification, object detection, segmentation, video classification, activity recognition, pose estimation, reinforcement learning, text-to-image/video generation 등으로 확장되었다. 대규모 데이터, GPU 연산 성능, 알고리즘 개선이 함께 작용했다.

## 시험ㆍ복습 체크포인트

- AI와 ML의 관계를 설명할 수 있어야 한다.
- Regression, classification, retrieval, clustering의 차이를 예시와 함께 구분할 수 있어야 한다.
- `Data -> Intelligence` 관점에서 학습 문제를 정의할 수 있어야 한다.
- ImageNet과 AlexNet이 deep learning 확산에 중요한 이유를 말할 수 있어야 한다.

