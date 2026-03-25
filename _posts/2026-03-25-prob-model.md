---
title: Probabilistic Model
date: 2026-03-25 23:48:00 +0900
categories: [Statistics]
order: 1
math: true
---

## Model이란
집 평수($x$)를 input으로 받으면 집값($y$)을 output으로 하는 관계식을 만들어보자.

집은 클수록 보통 가격이 비싸니 $y=\theta x$와 같은 선형적인 관계를 가진다고 가정할 수도 있다.

반면에 집이 커질수록 가격 상승률이 감소한다고 생각해 $y=\theta \log x$와 같은 로그 함수의 관계를 가진다고 가정할 수도 있다.

만약 집의 평수와 집값이 선형 관계를 가진다고 가정하면 집이 $k$평일 때 집값이 $\theta k$ 라고 예측할 수 있다.

이처럼 데이터를 잘 설명하는 input과 output의 관계식을 알 수 있다면 새로운 input에 대한 output을 predict할 수 있다.

하지만 현실에서는 관측되지 않은 변수나 노이즈 등으로 인해 완벽한 관계식을 세우는 것은 불가능하다.

예를 들어 집의 평수 뿐만 아니라 집의 층수, 준공 연도등의 변수 역시 가격에 영향을 끼칠 수 있고, 집의 평수를 측정하는데 오류가 생길 수도 있는 것이다.

따라서 불확실성 속에서 결정을 해야하므로 집의 평수가 $x$로 주어졌을 때 집값 $y$의 확률 분포를 세우는 것이 새로운 집의 평수가 주어졌을 때 그 집의 집값을 예측하는데 더 효과적이다.

Model은 $x$가 주어졌을 때 $y$의 확률분포의 군(family of probability distribution)을 의미한다. 즉 $p(y \| x ; \theta)$을 model이라고 한다.

위의 집값 예시에서 평수와 집값 사이에 선형적인 관계 위에 $\epsilon \sim \mathcal{N}(0, \sigma^2)$가 더해진다고 가정하자. 

즉 $y = \theta x + \epsilon$이 되며 x가 주어졌을 때 y의 확률 분포는 $p(y \| x ; \theta, \sigma^2) = \mathcal{N}(y \| \theta x , \sigma^2)$가 된다.



## Model Fitting(training)
Model을 fitting하거나 training한다는 것은, 선택한 모델 즉 family of probability distribution 중에 가장 training data의 분포를 잘 설명하는 분포의 parameter를 찾는 것이다.

이 말은 아래의 수식으로 표현할 수 있다.

$$
\hat{\theta} = \arg\min_\theta \mathcal{L}(\theta)
$$ 

여기서 $\mathcal{L}(\theta)$ 는 loss function이거나 objective function이라고 부르는데, 보통 model이 $\theta$ 라는 parameter를 가질 때 얼마나 training data의 분포를 설명하지 못 하는지의 측도가 되는 함수이다.

이렇게 구한 $\hat{\theta}$처럼 하나의 값으로 파라미터를 추정하는 방식을 point estimate(점 추정)이라고 한다. 

하지만 $\hat{\theta}$은 모든 데이터를 보고 결정한 것이 아니라 일부 만을 보고 결정한 것이므로 이 점 추정치가 얼마나 정확한 결과인지 불확실함을 정량화 할 필요가 있다. 
통계학에서는 이를 inference라고 한다. 이러한 inference에는 frequentist와 Bayesian 관점이 있다.

## Reference
* Murphy, K. P. (2022). [Probabilistic Machine Learning: An Introduction.](https://probml.github.io/pml-book/book1.html) MIT press.
* Prince, S. J. (2023). [Understanding Deep Learning.](https://udlbook.github.io/udlbook/) MIT press.
