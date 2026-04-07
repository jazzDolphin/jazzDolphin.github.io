---
title: Linear Regression (선형 회귀)
date: 2026-04-07 19:32:00 +0900
categories: [Machine Learning, Linear Models]
order: 1
math: true
---

## Linear Regression (선형 회귀)

$N$개의 feature과 target이 있는 데이터셋 $\mathcal{D} = \\{(\boldsymbol{x}_n, y_n)\\}\_{n=1}^N$이 주어졌다고 하자. 

여기서 $\boldsymbol{x}_n \in \mathbb{R}^{D} = [x\_{n, 1}, x\_{n, 2}, \dots x\_{n, D}]^\top$은 $n$번째 데이터의 $D$개의 feature를 담은 벡터이고, $y_n \in \mathbb{R}$은 $n$번째 데이터의 정답값(target)이다.

집값을 예측하는 문제를 예로 들면 $\boldsymbol{x}_n$은 집의 특징(평수, 방 개수, 역과의 거리 등)이고, $y_n$은 $n$번째 집의 실제 집값이다.

$y_n$이 $\boldsymbol{x}_n$와 선형적 관계를 갖고 있다고 가정하자. 

그렇다면 우리는 $n$번째 데이터의 feature $\boldsymbol{x}_n$이 주어지면 parameter $\boldsymbol{w} \in \mathbb{R}^{D+1}$에 대해 아래와 같이 예측값을 계산할 수 있다.

$$
\sum_{i=0}^D w_i \boldsymbol{x}_{n,i} = \boldsymbol{w}^\top \boldsymbol{x}_n
$$

(편의상 bias term $w_0$는 $x_{n, 0} = 1$을 가정해 $\boldsymbol{w}$에 포함시킨다.)

데이터셋 전체에 대해 이를 행렬 형태로 표현하기 위해, 각 데이터 포인트 $\boldsymbol{x}_n^\top$을 행으로 갖는 design matrix $\boldsymbol{X} \in \mathbb{R}^{N \times (D+1)}$와 정답값들을 모은 벡터 $\boldsymbol{y} \in \mathbb{R}^N$를 다음과 같이 정의하자.

$$
\boldsymbol{X} = \begin{bmatrix} 
  \boldsymbol{x}_1^\top \\ 
  \vdots \\ 
  \boldsymbol{x}_N^\top 
\end{bmatrix}, 
\quad \boldsymbol{y} = \begin{bmatrix}
  y_1 \\ 
  \vdots 
  \\ y_N 
\end{bmatrix}
$$

$n$번째 데이터의 예측 $\boldsymbol{w}^\top \boldsymbol{x}_n$이 $y_n$과 차이가 적게하는 parameter $\boldsymbol{w}$를 찾고 싶다. 

그러기 위해 실제 예측과 내 예상치가 얼마나 차이나는지 정량화하는 함수가 필요하다. 

$$
\mathrm{RSS}(\boldsymbol{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \boldsymbol{w}^\top \boldsymbol{x}_n)^2 = \frac{1}{2}(\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})^\top (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})
$$

$y_n - \boldsymbol{w}^\top \boldsymbol{x}_n$ 을 그냥 사용하지 않고 제곱하는 이유는 실제값이 예측값보다 더 큰 경우와 더 작은 경우를 더했을 때 그 차이가 상쇄되어버리는 것을 막기 위함이다.

$$
\arg \min_\boldsymbol{w} \mathrm{RSS}(\boldsymbol{w})
$$

를 풀면 선형 관계를 가정했을 때 $\mathcal{D}$를 가장 잘 설명하는 $\boldsymbol{w}$를 찾을 수 있다.

그렇다면 왜 하필 절댓값도, 4제곱도 아닌 잔차를 제곱한 것을 objective fuction으로 사용하는 것일까?

## Probabilistic Interpretation

선형 회귀 문제에서 input feature $\boldsymbol{x}_n$와 output $y_n$이 선형 관계가 있다고 가정한다. 하지만 실제 데이터는 완전한 선형 관계를 갖지 않으므로 노이즈 $\epsilon$을 더한 관계를 가정한다.

$$
y_n = \boldsymbol{w}^\top \boldsymbol{x}_n + \epsilon
$$

이때 노이즈 $\epsilon$이 평균이 $0$이고 분산이 $\sigma^2$인 가우시안 분포를 따른다고 가정하자. ($\epsilon \sim \mathcal{N}(0, \sigma^2)$)

$\boldsymbol{x}_n$이 주어졌을 때 $\boldsymbol{w}^\top \boldsymbol{x}_n$은 상수가 되므로, 가우시안 분포를 따르는 확률변수 $\epsilon$에 상수를 더한 $y_n$ 역시 가우시안 분포를 따른다.

즉, 기존 $\epsilon$의 분포에서 mean만  $\boldsymbol{w}^\top \boldsymbol{x}_n$만큼 이동하게 되며, 선형 회귀 모델은 결과적으로 다음과 같이 확률 분포로 정의된다.

$$
p(y_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}) = \mathcal{N}(y_n \mid \boldsymbol{w}^\top \boldsymbol{x}_n, \sigma^2)
$$

여기서 $\boldsymbol{\theta} = (\boldsymbol{w}, \sigma^2)$는 모델의 파라미터를 의미한다. 

### Maximum Likelihood Estimation (MLE)

이제 이 모델에 MLE를 적용하여 데이터셋 $\mathcal{D}$를 가장 잘 설명하는 최적의 파라미터를 찾아보자. 

$\mathcal{D}$가 i.i.d라고 가정할 때, objective funcion $\mathrm{NLL}(\boldsymbol{\theta})$은 다음과 같이 전개된다.

$$
\begin{align*}
  \mathrm{NLL}(\boldsymbol{w},\sigma^2) 
  &= -\sum_{n=1}^N \log \left[ \left(\frac{1}{2\pi \sigma^2}\right)^{\frac{1}{2}} \exp\left(-\frac{1}{2\sigma^2}(y_n - \boldsymbol{w}^\top \boldsymbol{x}_n)^2\right) \right] \\
  &= \frac{1}{2\sigma^2}\sum_{n=1}^N (y_n - \boldsymbol{w}^\top \boldsymbol{x}_n)^2 + \frac{N}{2}\log(2\pi\sigma^2)
\end{align*}
$$

전개한 $\mathrm{NLL}$ 함수를 바탕으로 $\mathrm{NLL}$을 최소화하는 $\boldsymbol{w}$를 구하는 최적화 문제를 아래와 같이 나타낼 수 있다.

$$
\begin{align*}
  \arg \min_\boldsymbol{w} \mathrm{NLL}(\boldsymbol{w},\sigma^2) 
  &= \arg \min_\boldsymbol{w} \left[ 
    \frac{1}{2\sigma^2}\sum_{n=1}^N (y_n - \boldsymbol{w}^\top \boldsymbol{x}_n)^2 + \frac{N}{2}\log(2\pi\sigma^2)
  \right] \\
  &= \arg \min_\boldsymbol{w} \underbrace{\frac{1}{2}\sum_{n=1}^N (y_n - \boldsymbol{w}^\top \boldsymbol{x}_n)^2}_{\mathrm{RSS}(\boldsymbol{w})} 
\end{align*}
$$

$\mathrm{NLL}$에서 $\boldsymbol{w}$와 무관한 상수항을 제외하면 $\boldsymbol{w}$에 대해 NLL을 최소화하는 문제는 결국  $\mathrm{RSS}(w)$를 최소화하는 문제와 완벽하게 일치하게 된다.

결론적으로, $y \mid \boldsymbol{x}, \boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{w}^\top \boldsymbol{x}, \sigma^2)$ 이고, 데이터가 i.i.d.라는 가정 하에 MLE를 수행하는 것은 RSS를 최소화하는 것과 동일한 결과를 갖는다.

이것이 object function으로 RSS를 사용하는 확률론적 근거이다.

## RSS의 Convexity

$\mathrm{RSS}(\boldsymbol{w})$를 $\boldsymbol{w}$에 대해 두 번 미분하여 Hessian 행렬을 구하면 다음과 같다.

$$
\nabla_\boldsymbol{w}^2 \mathrm{RSS}(\boldsymbol{w}) = \boldsymbol{X}^\top \boldsymbol{X}
$$

임의의 벡터 $\boldsymbol{v}$에 대하여 

$$
\boldsymbol{v}^\top (\boldsymbol{X}^\top \boldsymbol{X}) \boldsymbol{v} = (\boldsymbol{X}\boldsymbol{v})^\top (\boldsymbol{X}\boldsymbol{v}) = \|\boldsymbol{X}\boldsymbol{v}\|_2^2 \ge 0
$$

이 성립하므로, $\boldsymbol{X}^\top \boldsymbol{X}$는 positive semi-definite (PSD)이다.

Hessian 행렬이 항상 PSD이므로 $\mathrm{RSS}(\boldsymbol{w})$는 $\boldsymbol{w}$에 대한 convex 함수이며, 이는 local minimum이 곧 global minimum이 됨을 보장한다.

따라서 Gradient Descent와 같은 최적화 알고리즘을 사용하면 global minimum에 도달할 수 있다.

## Normal Equation

$\arg \min_\boldsymbol{w} \mathrm{RSS}(\boldsymbol{w})$를 만족하는 최적의 가중치 $\boldsymbol{w}$를 iterative한 방법 말고 closed form으로 구해보자.

$\mathrm{RSS}(\boldsymbol{w})$를 $\boldsymbol{w}$에 대해 gradient를 구하면 다음과 같다.

$$
\nabla_\boldsymbol{w} \mathrm{RSS}(\boldsymbol{w}) = \boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{w} - \boldsymbol{X}^\top \boldsymbol{y}
$$

최솟값에서는 gradient가 0이 되어야 하므로, $\nabla_\boldsymbol{w} \mathrm{RSS}(\boldsymbol{w}) = 0$으로 두고 이항하면 아래의 방정식을 얻는다.

$$
\boldsymbol{X}^\top \boldsymbol{X}\boldsymbol{w} = \boldsymbol{X}^\top \boldsymbol{y}
$$

이를 Normal equation이라고 부른다.

이 방정식을 $\boldsymbol{w}$에 대해 정리하면 OLS(Ordinary Least Squares) solution을 얻을 수 있다.

$$
\hat{\boldsymbol{w}} = (\boldsymbol{X}^\top \boldsymbol{X})^{-1}\boldsymbol{X}^\top \boldsymbol{y}
$$

## Reference
* Stanford CS229 Autumn 2018 Lecture 2, 3
* Murphy, K. P. (2022). [Probabilistic Machine Learning: An Introduction.](https://probml.github.io/pml-book/book1.html) MIT press.
