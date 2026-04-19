---
title: Exponential Family (지수족)
date: 2026-04-18 20:24:00 +0900
categories: [Statistics]
order: 3
math: true
---

## Exponential family의 정의

어떤 확률 분포가 $\boldsymbol{\eta} \in \mathbb{R}^K$에 대해 parameterized 되고,

$$
p(\boldsymbol{y} \mid \boldsymbol{\eta})
\triangleq \frac{1}{Z(\boldsymbol{\eta})} h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y})\right)
= h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})\right)
$$

위의 형태로 표현할 수 있는 확률 분포를 exponential family에 속해있다고 한다.

- $$h(\boldsymbol{y})$$ : scaling constant, base measure
- $$\boldsymbol{\eta}$$ : natural parameter (canonical parameter)
- $$\mathcal{T}(\boldsymbol{y}) \in \mathbb{R}^K$$ : sufficient statistics
- $$Z(\boldsymbol{\eta})$$ : partition function
- $$A(\boldsymbol{\eta}) = \log Z(\boldsymbol{\eta})$$ : log partition function

확률 분포는 다양한 parameterization이 가능하다. 가우시안 분포를 예로 들면 $$\mu, \sigma^2$$로 parameterized할 수도 있고, $$\boldsymbol{\eta}$$로 parameterized할 수도 있다.

가능한 $$\boldsymbol{\eta}$$의 집합을 natural parameter space $$\mathcal{N}$$이라 하고, $$\mathcal{N}$$과 다른 parameter들의 집합을 $$\Theta$$라고 하자. 

일대일 mapping $$f:\mathcal{\Theta} \to \mathcal{N}$$가 존재하여 $$\boldsymbol{\eta} = f(\boldsymbol{\theta})$$라고 하자.

그렇다면 $$\boldsymbol{\theta}$$에 대해 parameterized된 확률 분포의 family를 아래와 같이 표현할 수 있다.

$$
p(\boldsymbol{y} \mid \boldsymbol{\theta}) = h(\boldsymbol{y}) \exp \left( f(\boldsymbol{\theta})^{\top} \mathcal{T}(\boldsymbol{y}) - A(f(\boldsymbol{\theta}))\right)
$$

Exponential family의 special case로는 다음과 같은 것들이 있다.

- Canonical Form: $$\eta = \boldsymbol{\theta}$$인 경우.
- Natural Exponential Family(NEF): Canonical form인데, $$\mathcal{T}(\boldsymbol{y}) = \boldsymbol{y}$$인 경우.

NEF는 아래와 같이 표현할 수 있다.

$$
p(\boldsymbol{y} \mid \boldsymbol{\eta}) = h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \boldsymbol{y} - A(\boldsymbol{\eta})\right)
$$

## Exponential family의 예시

Exponential family에 속하는 분포의 예시로는 Bernoulli, Binomial, Poisson, Gaussian, Exponential, Gamma 등이 있다.

각각의 분포는 서로 다른 $$h(\boldsymbol{y})$$, $$\boldsymbol{\eta}$$, $$\mathcal{T}(\boldsymbol{y})$$, $$A(\boldsymbol{\eta})$$를 가진다.

이러한 분포들을 잘 조작하면 exponential family의 형태로 표현할 수 있다.

### Example 1. Bernoulli distribution

$$
\begin{align*}
    p(y;\phi) &= \phi^y (1-\phi)^{1-y} \\
    &= \exp(y \log \phi + (1-y) \log(1-\phi)) \\
    &= \exp\left(\log \left( \frac{\phi}{1-\phi}\right) y + \log(1-\phi) \right)
\end{align*}\\
$$

베르누이 분포를

$$
\begin{align*}
    \eta &= \log \left( \frac{\phi}{1-\phi} \right) \\
    \mathcal{T}(y) &= y \\
    A(\eta) &= -\log(1-\phi) = \log(1+e^\eta) \\
    h(y) &= 1
\end{align*}
$$

인 exponential family form으로 표현 가능하기에, 베르누이 분포는 지수족에 속한다.

여기서 $$f(\phi) = \log \left( \frac{\phi}{1-\phi} \right)$$이다.

그리고 $$\phi$$를 $$\eta$$로 표현하면 

$$
\phi = f^{-1}(\eta) = \frac{1}{1 + e^{-\eta}} = \sigma(\eta)
$$

시그모이드 함수가 나온다.

### Example 2. Gaussian distribution

$$
\begin{align*}
    p(y;\mu, \sigma^2)
    &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} (y-\mu)^2 \right) \\
    &= \frac{1}{\sqrt{2\pi}} \exp\left( 
        -\frac{1}{2\sigma^2} y^2 + \frac{\mu}{\sigma^2} y - \frac{\mu^2}{2\sigma^2} - \log \sigma
    \right)
\end{align*}
$$

가우시안 분포를

$$
\begin{align*}
    \boldsymbol{\eta} &= \begin{bmatrix}\frac{\mu}{\sigma^2} \\ -\frac{1}{2\sigma^2}\end{bmatrix} \\
    \mathcal{T}(y) &= \begin{bmatrix}y \\ y^2\end{bmatrix} \\
    A(\boldsymbol{\eta}) &= \frac{\mu^2}{2\sigma^2} + \log \sigma = \frac{-\eta_1^2}{4\eta_2} - \frac{1}{2} \log(-2\eta_2) \\
    h(y) &= \frac{1}{\sqrt{2\pi}}
\end{align*}
$$

인 exponential family form으로 표현 가능하기에 가우시안 분포는 지수족에 속한다.

### Example 3. Gaussian distribution with fixed variance

$$\sigma^2$$가 주어져 변수가 아닌 상수라고 하자.

$$
\begin{align*}
    p(y;\mu, \sigma^2)
    &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} (y-\mu)^2 \right) \\
    &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} y^2 \right) \exp\left( \frac{\mu}{\sigma^2} y - \frac{\mu^2}{2\sigma^2} \right)
\end{align*}
$$

가우시안 분포를

$$
\begin{align*}
    \eta &= \frac{\mu}{\sigma^2} \implies \mu = \eta \sigma^2 \\
    \mathcal{T}(y) &= y \\
    A(\eta) &= \frac{\mu^2}{2\sigma^2} = \frac{\sigma^2}{2}\eta^2 \\
    h(y) &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} y^2 \right) = \mathcal{N}(y \mid 0, \sigma^2)
\end{align*}
$$

인 exponential family form으로 표현 가능하기에 가우시안 분포는 지수족에 속한다.

## Log Partition Funtion $$A(\boldsymbol{\eta})$$의 성질

$$
\begin{align*}
    \int p(\boldsymbol{y} \mid \boldsymbol{\eta})  &= 1 \\
    \int h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})\right) d\boldsymbol{y} &= 1 \\
    \exp(A(\boldsymbol{\eta})) &= \int h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y})\right) d\boldsymbol{y} \\
    \end{align*}
$$

양변을 $\boldsymbol{\eta}$에 대해 미분하면

$$
\begin{align*}
    \nabla_{\boldsymbol{\eta}} \exp(A(\boldsymbol{\eta})) &= \nabla_{\boldsymbol{\eta}} \int h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y})\right) d\boldsymbol{y} \\
    \exp(A(\boldsymbol{\eta})) \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) &= \int h(\boldsymbol{y}) \mathcal{T}(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y})\right) d\boldsymbol{y} \\
    \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) &= \int \mathcal{T}(\boldsymbol{y}) \underbrace{h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})\right) }_{p(\boldsymbol{y}\mid \boldsymbol{\eta})}d\boldsymbol{y} \\
    &= \mathbb{E}[\mathcal{T}(\boldsymbol{y})]
\end{align*}
$$

즉 $$A(\boldsymbol{\eta})$$의 gradient는 $$\mathcal{T}(\boldsymbol{y})$$의 기대값과 같다. 

또한 $$A(\boldsymbol{\eta})$$의 hessian을 구하면

$$
\begin{align*}
    \nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta})
    &= \nabla_{\boldsymbol{\eta}} \mathbb{E}[\mathcal{T}(\boldsymbol{y})] ^\top \\
    &= \nabla_{\boldsymbol{\eta}} \int  h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})\right) \mathcal{T}(\boldsymbol{y})^\top d\boldsymbol{y} \\
    &= \int  h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})\right) \left( \mathcal{T}(\boldsymbol{y}) - \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta})\right) \mathcal{T}(\boldsymbol{y})^\top d\boldsymbol{y} \\
    &= \int \underbrace{h(\boldsymbol{y}) \exp \left( \boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y}) - A(\boldsymbol{\eta})\right)}_{p(\boldsymbol{y} \mid \boldsymbol{\eta})} \mathcal{T}(\boldsymbol{y}) \mathcal{T}(\boldsymbol{y})^\top d\boldsymbol{y} - \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta})^\top \\
    &= \mathbb{E}[\mathcal{T}(\boldsymbol{y}) \mathcal{T}(\boldsymbol{y})^\top] - \mathbb{E}[\mathcal{T}(\boldsymbol{y})] \mathbb{E}[\mathcal{T}(\boldsymbol{y})]^\top \\
    &= \mathrm{Cov}[\mathcal{T}(\boldsymbol{y})]
\end{align*}
$$

즉 $$A(\boldsymbol{\eta})$$의 hessian은 $$\mathcal{T}(\boldsymbol{y})$$의 공분산 행렬과 같다.

공분산 행렬은 항상 positive semi-definite이므로, $$A(\boldsymbol{\eta})$$는 convex function이다.

계산할 때 $$\int$$와 $$\nabla$$의 순서를 바꿨는데, 지수족 분포는 $$\mathcal{N}$$ 내부에서는 Liebniz integral rule이 적용되어 바꿀 수 있다고 한다.

## References
- Murphy, K. P. (2022). [Probabilistic Machine Learning: An Introduction.](https://probml.github.io/pml-book/book1.html) MIT press.
- Murphy, K. P. (2023). [Probabilistic Machine Learning: Advanced Topics.](https://probml.github.io/pml-book/book2.html) MIT press.
- Stanford CS229 Autumn 2018 Lecture 4
