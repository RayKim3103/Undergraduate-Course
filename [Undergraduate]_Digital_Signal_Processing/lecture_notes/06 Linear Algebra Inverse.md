---
title: "06. Linear Algebra Inverse"
pages: 5
tags: [DSP, lecture-note, inverse-problem, least-squares]
---

# 06. Linear Algebra Inverse

> 이전: [[05 LTI System to Linear Algebra]]
> 다음: [[07 Z-Transform Introduction]]

## 학습 목표

이 자료는 DSP 문제를 행렬 방정식으로 보았을 때 inverse가 언제 가능한지, 불가능할 때 least squares inverse를 어떻게 쓰는지 정리한다.

## 기본 모델

많은 DSP 문제는 다음 형태로 쓸 수 있다.

$$
\mathbf{y}=H\mathbf{x}
$$

- $\mathbf{x}$: 입력 또는 원 신호
- $H$: 시스템, 필터, 관측 행렬
- $\mathbf{y}$: 출력 또는 관측 신호

잡음이 있으면:

$$
\mathbf{y}=H\mathbf{x}+\mathbf{n}
$$

## Direct Inverse

$H$가 정방행렬이고 nonsingular이면

$$
\mathbf{x}=H^{-1}\mathbf{y}
$$

로 복원할 수 있다.

하지만 실제 DSP에서는 direct inverse가 잘 되지 않는 경우가 많다.

1. $H$가 정방행렬이 아닐 수 있다.
2. $H$가 singular일 수 있다.
3. $H$가 거의 singular라 noise가 크게 증폭될 수 있다.
4. 관측 $\mathbf{y}$가 noise를 포함한다.

## Overdetermined Case

방정식 수가 미지수 수보다 많은 경우:

$$
H \in \mathbb{R}^{m\times n}, \quad m>n
$$

일반적으로 모든 방정식을 정확히 만족하는 $\mathbf{x}$가 없을 수 있다. 이때 residual을 최소화한다.

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}}\|\mathbf{y}-H\mathbf{x}\|_2^2
$$

정규방정식:

$$
H^TH\hat{\mathbf{x}}=H^T\mathbf{y}
$$

해:

$$
\hat{\mathbf{x}}=(H^TH)^{-1}H^T\mathbf{y}
$$

복소수 행렬에서는 transpose 대신 Hermitian transpose를 쓴다.

$$
\hat{\mathbf{x}}=(H^HH)^{-1}H^H\mathbf{y}
$$

## Underdetermined Case

미지수 수가 방정식 수보다 많은 경우:

$$
m<n
$$

해가 무수히 많거나 추가 정보 없이는 결정되지 않는다. 이때 최소 norm 해, regularization, prior information이 필요하다.

대표적 정식화:

$$
\hat{\mathbf{x}}=
\arg\min_{\mathbf{x}}\|H\mathbf{x}-\mathbf{y}\|_2^2
+\lambda R(\mathbf{x})
$$

여기서 $R(\mathbf{x})$는 smoothness, sparsity 같은 prior를 반영한다.

## Ill-Posed/Singular Case

$H$가 singular 또는 ill-conditioned이면 작은 noise가 복원 결과에서 크게 증폭된다.

예:

$$
Y(e^{j\omega})=H(e^{j\omega})X(e^{j\omega})+N(e^{j\omega})
$$

단순 inverse:

$$
\hat{X}(e^{j\omega})=\frac{Y(e^{j\omega})}{H(e^{j\omega})}
$$

만약 $|H(e^{j\omega})|$가 매우 작으면 noise term도 크게 증폭된다.

## DSP 문제 유형

| 주어진 것 | 구할 것 | 문제 이름 |
|---|---|---|
| $x$, $T\{\cdot\}$ | $y$ | filtering |
| $x$, $y$ | $T\{\cdot\}$ | filter design/system identification |
| $y$, $T\{\cdot\}$ | $x$ | inverse filtering/deconvolution |
| $y$만 주어짐 | $x$, $T\{\cdot\}$ | blind deconvolution |

## Norm

### Vector Norm

대표적인 $p$-norm:

$$
\|\mathbf{x}\|_p =
\left(\sum_i |x_i|^p\right)^{1/p}
$$

특히 least squares에서는 $L_2$ norm이 중요하다.

$$
\|\mathbf{x}\|_2^2 = \sum_i |x_i|^2
$$

### Matrix Norm

행렬 norm은 시스템이 입력을 얼마나 증폭하는지 나타낸다.

$$
\|H\| = \max_{\mathbf{x}\neq0}\frac{\|H\mathbf{x}\|}{\|\mathbf{x}\|}
$$

inverse problem에서는 condition number가 중요하다.

$$
\kappa(H)=\|H\|\|H^{-1}\|
$$

condition number가 크면 작은 오차가 해에서 크게 증폭된다.

## 체크포인트

- direct inverse는 이상적인 경우에만 안전하다.
- least squares는 정확히 맞추기보다 residual energy를 최소화한다.
- overdetermined 문제는 보통 LS로 안정적으로 풀 수 있다.
- underdetermined 문제는 prior나 regularization 없이는 해가 결정되지 않는다.
- inverse filtering은 수학적으로 가능해도 noise 때문에 실용적으로 불안정할 수 있다.
