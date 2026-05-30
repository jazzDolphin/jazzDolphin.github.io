---
title: "Exponential Family 1: Exponential Family(지수족)의 정의와 예시"
date: 2026-05-30 17:48:00 +0900
categories: [Statistics]
order: 9
math: true
---

## Exponential family의 정의

어떤 확률 분포가 $\boldsymbol{\eta} \in \mathbb{R}^K$에 대해 parameterized 되고,

$$
p(\boldsymbol{x} \mid \boldsymbol{\eta})
\triangleq \frac{1}{Z(\boldsymbol{\eta})} h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x})\right)
= h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}) - A(\boldsymbol{\eta})\right)
$$

위의 형태로 표현할 수 있는 확률 분포를 exponential family에 속해있다고 한다.

Exponential family 분포는 [Fisher-Neyman factorization theorem](https://jazzdolphin.github.io/posts/data-reduction3/)의 조건을 그 자체로 만족한다. 즉, $$T(\boldsymbol{x})$$는 sufficient statistic이다.

Exponential family의 정의에서 각 구성 요소는 다음과 같은 의미를 가진다.

- $$h(\boldsymbol{x})\ge 0$$ : base measure
- $$\boldsymbol{\eta}$$ : natural parameter (canonical parameter)
- $$T(\boldsymbol{x}) \in \mathbb{R}^K$$ : sufficient statistic
- $$Z(\boldsymbol{\eta})=\int h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x})\right) d\boldsymbol{x}$$ : partition function
- $$A(\boldsymbol{\eta}) = \log Z(\boldsymbol{\eta})$$ : log partition function

확률 분포는 다양한 parameterization이 가능하다. 가우시안 분포를 예로 들면 $$\mu, \sigma^2$$로 parameterized할 수도 있고, $$\boldsymbol{\eta}$$로 parameterized할 수도 있다.

가능한 $$\boldsymbol{\eta}$$의 집합을 natural parameter space $$\Omega=\{\boldsymbol{\eta} \in \mathbb{R}^K: A(\boldsymbol{\eta}) < \infty \}$$이라 하고, $$\Omega$$과 다른 parameter들의 집합을 $$\Theta$$라고 하자. 

Bijection mapping $$f:\mathcal{\Theta} \to \Omega$$가 존재하여 $$\boldsymbol{\eta} = f(\boldsymbol{\theta})$$라고 하자.

그렇다면 exponential family를 $$\boldsymbol{\theta}$$에 대해 parameterized된 확률 분포의 family로 generalize하여 다음과 같이 표현할 수 있다.

$$
p(\boldsymbol{x} \mid \boldsymbol{\theta}) = h(\boldsymbol{x}) \exp \left( f(\boldsymbol{\theta})^{\top} T(\boldsymbol{x}) - A(f(\boldsymbol{\theta}))\right)
$$

이때 $$\eta = f(\boldsymbol{\theta}) = \boldsymbol{\theta}$$라서 처음 정의에서처럼 표현된 경우를 canonical form으로 표현되었다고 한다.

### Exponential family의 special case

Exponential family의 special case로는 다음과 같은 것들이 있다.

- Natural Exponential Family(NEF): $$T(\boldsymbol{x}) = \boldsymbol{x}$$인 경우. 

  즉 $$p(\boldsymbol{x} \mid \boldsymbol{\eta}) = h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} \boldsymbol{x} - A(\boldsymbol{\eta})\right)$$인 경우.
- Minimal Exponential Family: 0벡터가 아닌 모든 $$\boldsymbol{\eta}$$에 대해 $$\boldsymbol{\eta}^\top T(\boldsymbol{x})\neq 0$$인 경우. 

## Exponential family의 예시

Exponential family에 속하는 분포의 예시로는 Bernoulli, Binomial, Poisson, Gaussian, Exponential, Gamma 등이 있다.

각각의 분포는 서로 다른 $$h(\boldsymbol{x})$$, $$\boldsymbol{\eta}$$, $$T(\boldsymbol{x})$$, $$A(\boldsymbol{\eta})$$를 가진다.

이러한 분포들을 잘 조작하면 exponential family의 형태로 표현할 수 있다.

### Example 1. Bernoulli distribution

$$
\begin{align*}
    p(x \mid \phi) &= \phi^x (1-\phi)^{1-x} \\
    &= \exp(x \log \phi + (1-x) \log(1-\phi)) \\
    &= \exp\left(\log \left( \frac{\phi}{1-\phi}\right) x + \log(1-\phi) \right)
\end{align*}\\
$$

베르누이 분포를

$$
\begin{align*}
    \eta &= \log \left( \frac{\phi}{1-\phi} \right) \\
    T(x) &= x \\
    A(\eta) &= -\log(1-\phi) = \log(1+e^\eta) \\
    h(x) &= 1
\end{align*}
$$

인 exponential family form으로 표현 가능하기에, 베르누이 분포는 지수족에 속한다.

그리고 $$\phi$$를 $$\eta$$로 표현하면 

$$
\phi = f^{-1}(\eta) = \frac{1}{1 + e^{-\eta}} = \sigma(\eta)
$$

시그모이드 함수가 나온다.

### Example 2. Gaussian distribution

$$
\begin{align*}
    p(x \mid \mu, \sigma^2)
    &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} (x-\mu)^2 \right) \\
    &= \frac{1}{\sqrt{2\pi}} \exp\left( 
        -\frac{1}{2\sigma^2} x^2 + \frac{\mu}{\sigma^2} x - \frac{\mu^2}{2\sigma^2} - \log \sigma
    \right)
\end{align*}
$$

가우시안 분포를

$$
\begin{align*}
    \boldsymbol{\eta} &= \begin{bmatrix}\frac{\mu}{\sigma^2} \\ -\frac{1}{2\sigma^2}\end{bmatrix} \\
    T(x) &= \begin{bmatrix}x \\ x^2\end{bmatrix} \\
    A(\boldsymbol{\eta}) &= \frac{\mu^2}{2\sigma^2} + \log \sigma = \frac{-\eta_1^2}{4\eta_2} - \frac{1}{2} \log(-2\eta_2) \\
    h(x) &= \frac{1}{\sqrt{2\pi}}
\end{align*}
$$

인 exponential family form으로 표현 가능하기에 가우시안 분포는 지수족에 속한다.

### Example 3. Gaussian distribution with fixed variance

$$\sigma^2$$가 주어져 변수가 아닌 상수라고 하자.

$$
\begin{align*}
    p(x \mid \mu, \sigma^2)
    &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} (x-\mu)^2 \right) \\
    &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} x^2 \right) \exp\left( \frac{\mu}{\sigma^2} x - \frac{\mu^2}{2\sigma^2} \right)
\end{align*}
$$

가우시안 분포를

$$
\begin{align*}
    \eta &= \frac{\mu}{\sigma^2} \implies \mu = \eta \sigma^2 \\
    T(x) &= x \\
    A(\eta) &= \frac{\mu^2}{2\sigma^2} = \frac{\sigma^2}{2}\eta^2 \\
    h(x) &= \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{1}{2\sigma^2} x^2 \right) = \mathcal{N}(x \mid 0, \sigma^2)
\end{align*}
$$

인 exponential family form으로 표현 가능하기에 가우시안 분포는 지수족에 속한다.

다음 포스트에서는 exponential family의 대표적인 성질에 대해 알아보겠다.

## References
- Murphy, K. P. (2022). [Probabilistic Machine Learning: An Introduction.](https://probml.github.io/pml-book/book1.html) MIT press.
- Murphy, K. P. (2023). [Probabilistic Machine Learning: Advanced Topics.](https://probml.github.io/pml-book/book2.html) MIT press.
- Stanford CS229 Autumn 2018 Lecture 4
