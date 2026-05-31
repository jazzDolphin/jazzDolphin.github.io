---
title: "Exponential Family 2: Exponential Family의 성질"
date: 2026-05-31 21:20:00 +0900
categories: [Statistical Inference, Foundations]
order: 10
math: true
---

[이전 포스트](https://jazzdolphin.github.io/posts/exponential-family1/)에서는 exponential family의 정의와 예시를 살펴보았다. 

이번 포스트에서는 natural parameter space $$\Omega$$가 convex set임을 증명하고, log partition function $$A(\boldsymbol{\eta})$$의 성질과 cumulant function으로서의 역할에 대해서 살펴보겠다.

## 1. Natural Parameter Space $$\Omega$$는 convex set이다.

Natural Parameter space $$\Omega=\{\boldsymbol{\eta} \in \mathbb{R}^k: A(\boldsymbol{\eta}) < \infty \}$$가 convex set임을 증명하겠다.

모든 $$\boldsymbol{\eta}_1, \boldsymbol{\eta}_2 \in \Omega$$와 모든 $$\lambda \in [0,1]$$에 대해

$$\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2 \in \Omega$$

임을 보이면 된다. 이때 $$\lambda= 0 \text{ or } 1$$인 경우는 자명하게 참이므로 $$\lambda \in (0,1)$$인 경우에 대해서만 증명하겠다.

$$\Omega$$의 원소임을 증명하는 것은 곧 

$$A(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2) < \infty$$

임을 보이는 것과 같다.

$$A(\boldsymbol{\eta}) = \log Z(\boldsymbol{\eta})$$인데, $$\log$$는 단조 증가 함수이고, $$Z(\boldsymbol{\eta}) = \int h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x})\right) d\boldsymbol{x} > 0$$이므로, 

$$Z(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2) < \infty$$

를 보이는 것과 동치이다. 

모든 $$\boldsymbol{\eta}_1, \boldsymbol{\eta}_2 \in \Omega$$와 모든 $$\lambda \in (0,1)$$에 대해

$$
\begin{align*}
    Z(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2) 
    &= \int h(\boldsymbol{x}) \exp \left[
        (\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2)^{\top} T(\boldsymbol{x})
    \right] d\boldsymbol{x} \\
    &= \int h(\boldsymbol{x}) \exp \left[
        \lambda \boldsymbol{\eta}_1^{\top} T(\boldsymbol{x})
    \right]
    \exp \left[
        (1-\lambda) \boldsymbol{\eta}_2^{\top} T(\boldsymbol{x})
    \right] d\boldsymbol{x} \\
    &= \int \left[
        h(\boldsymbol{x}) \exp \left[
            \boldsymbol{\eta}_1^{\top} T(\boldsymbol{x})
        \right]
    \right]^\lambda
    \left[
        h(\boldsymbol{x}) \exp \left[
            \boldsymbol{\eta}_2^{\top} T(\boldsymbol{x})
        \right]
    \right]^{1-\lambda} d\boldsymbol{x} \\
\end{align*}
$$

$$\frac{1}{p} + \frac{1}{q} = 1$$인 모든 $$p, q > 1$$에 대해 

$$\int |fg| d\boldsymbol{x} \leq \left( \int |f|^p d\boldsymbol{x} \right)^{\frac{1}{p}} \left( \int |g|^q d\boldsymbol{x} \right)^{\frac{1}{q}}$$

Holder's inequality를 이용하기 위해 $$p=\frac{1}{\lambda}$$, $$q=\frac{1}{1-\lambda}$$로 설정하자. 이때 $$\frac{1}{p} + \frac{1}{q} = \lambda + (1-\lambda) =  1$$이고 $$\lambda \in (0,1)$$이므로 $$p, q > 1$$이다.

$$f(\boldsymbol{x}) = \left[h(\boldsymbol{x}) \exp \left[\boldsymbol{\eta}_1^{\top} T(\boldsymbol{x})\right]\right]^\lambda$$, $$g(\boldsymbol{x}) = \left[h(\boldsymbol{x}) \exp \left[\boldsymbol{\eta}_2^{\top} T(\boldsymbol{x})\right]\right]^{1-\lambda}$$라고 하자.

그렇다면 Holder's inequality에 의해

$$
\begin{align*}
    &Z(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2) \\
    &= \int f(\boldsymbol{x}) g(\boldsymbol{x}) d\boldsymbol{x} 
    \leq \left(
        \int f(\boldsymbol{x})^p d\boldsymbol{x}
    \right)^\frac{1}{p}
    \left(
        \int g(\boldsymbol{x})^q d\boldsymbol{x}
    \right)^\frac{1}{q} \\
\end{align*}
$$

가 성립한다. (절댓값은 $$f(\boldsymbol{x})$$와 $$g(\boldsymbol{x})$$가 0 이상이므로 생략했다.)

각각의 적분을 계산해보자.

$$
\begin{align*}
    \int f(\boldsymbol{x})^p d\boldsymbol{x}
    &= \int \left[h(\boldsymbol{x}) \exp \left[\boldsymbol{\eta}_1^{\top} T(\boldsymbol{x})\right]\right] d\boldsymbol{x} \\
    &= Z(\boldsymbol{\eta}_1)
\end{align*}
$$

$$\boldsymbol{\eta}_1 \in \Omega$$이므로 $$Z(\boldsymbol{\eta}_1) < \infty$$이다.

따라서 $$\left( \int f(\boldsymbol{x})^p d\boldsymbol{x} \right)^\frac{1}{p} < \infty$$이고, 같은 이유로 $$\left( \int g(\boldsymbol{x})^q d\boldsymbol{x} \right)^\frac{1}{q} < \infty$$이다.

$$Z(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2) < \infty$$이므로 $$\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2 \in \Omega$$이다.

따라서 $$\Omega$$는 convex set이다.

### $$A(\boldsymbol{\eta})$$는 convex function이다.

$$\Omega$$가 convex set이라는 사실은 $$A(\boldsymbol{\eta})$$가 $$\Omega$$를 domain으로 하는 convex function임에 필요조건이다.

Hoder's inequality에서

$$
\begin{align*}
    Z(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2) 
    &\leq \left(
        \int f(\boldsymbol{x})^p d\boldsymbol{x}
    \right)^\frac{1}{p}
    \left(
        \int g(\boldsymbol{x})^q d\boldsymbol{x}
    \right)^\frac{1}{q} \\
    &= Z(\boldsymbol{\eta}_1)^\lambda Z(\boldsymbol{\eta}_2)^{1-\lambda} \\
\end{align*}
$$

이다. 

여기에 양변에 $$\log$$를 취하면 ($$\log$$는 단조 증가 함수이므로 부등호의 방향이 바뀌지 않는다.)

$$
\begin{align*}
    A(\lambda \boldsymbol{\eta}_1 + (1-\lambda) \boldsymbol{\eta}_2)
    &\leq \log \left[ Z(\boldsymbol{\eta}_1)^\lambda Z(\boldsymbol{\eta}_2)^{1-\lambda} \right] \\
    &= \lambda \log Z(\boldsymbol{\eta}_1) + (1-\lambda) \log Z(\boldsymbol{\eta}_2) \\
    &= \lambda A(\boldsymbol{\eta}_1) + (1-\lambda) A(\boldsymbol{\eta}_2)
\end{align*}
$$

convex function의 정의에 의해 $$A(\boldsymbol{\eta})$$는 convex function이다.

## 2. Log Partition Function $$A(\boldsymbol{\eta})$$의 성질

Log Partition Function $$A(\boldsymbol{\eta})$$는 다양한 성질을 가지고 있다.

### $$A(\boldsymbol{\eta})$$의 gradient는 $$T(\boldsymbol{x})$$의 기댓값과 같다.

$$
\begin{align*}
    \int p(\boldsymbol{x} \mid \boldsymbol{\eta}) d\boldsymbol{x} &= 1 \\
    \int h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}) - A(\boldsymbol{\eta})\right) d\boldsymbol{x} &= 1 \\
    \exp(A(\boldsymbol{\eta})) &= \int h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x})\right) d\boldsymbol{x} \\
    \end{align*}
$$

양변을 $\boldsymbol{\eta}$에 대해 미분하면

$$
\begin{align*}
    \nabla_{\boldsymbol{\eta}} \exp(A(\boldsymbol{\eta})) 
    &= \nabla_{\boldsymbol{\eta}} \int h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x})\right) d\boldsymbol{x} \\
    \exp(A(\boldsymbol{\eta})) \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) &= \int h(\boldsymbol{x}) T(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x})\right) d\boldsymbol{x} \\
    \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) &= \int T(\boldsymbol{x}) \underbrace{h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}) - A(\boldsymbol{\eta})\right) }_{p(\boldsymbol{x}\mid \boldsymbol{\eta})}d\boldsymbol{x} \\
    &= \mathbb{E}[T(\boldsymbol{x})]
\end{align*}
$$

즉 $$A(\boldsymbol{\eta})$$의 gradient는 $$T(\boldsymbol{x})$$의 기댓값과 같다. 

### $$A(\boldsymbol{\eta})$$의 hessian은 $$T(\boldsymbol{x})$$의 공분산 행렬과 같다.

또한 $$A(\boldsymbol{\eta})$$의 hessian을 구하면

$$
\begin{align*}
    \nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta})
    &= \nabla_{\boldsymbol{\eta}} (\nabla_\eta A)^\top \\
    &= \nabla_{\boldsymbol{\eta}} \int  h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}) - A(\boldsymbol{\eta})\right) T(\boldsymbol{x})^\top d\boldsymbol{x} \\
    &= \int  h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}) - A(\boldsymbol{\eta})\right) \left( T(\boldsymbol{x}) - \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta})\right) T(\boldsymbol{x})^\top d\boldsymbol{x} \\
    &= \int \underbrace{h(\boldsymbol{x}) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}) - A(\boldsymbol{\eta})\right)}_{p(\boldsymbol{x} \mid \boldsymbol{\eta})} T(\boldsymbol{x}) T(\boldsymbol{x})^\top d\boldsymbol{x} - \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta})^\top \\
    &= \mathbb{E}[T(\boldsymbol{x}) T(\boldsymbol{x})^\top] - \mathbb{E}[T(\boldsymbol{x})] \mathbb{E}[T(\boldsymbol{x})]^\top \\
    &= \mathrm{Cov}[T(\boldsymbol{x})]
\end{align*}
$$

즉 $$A(\boldsymbol{\eta})$$의 hessian은 $$T(\boldsymbol{x})$$의 공분산 행렬과 같다.

공분산 행렬은 항상 positive semi-definite이므로, $$A(\boldsymbol{\eta})$$는 convex function이라는 [위에서 보인 사실](https://jazzdolphin.github.io/posts/exponential-family2//posts/exponential-family2/#aboldsymboleta%EB%8A%94-convex-function%EC%9D%B4%EB%8B%A4)과 일치한다.

계산할 때 $$\int$$와 $$\nabla$$의 순서를 바꿨는데, 지수족 분포는 $$\Omega$$ 내부에서는 Leibniz integral rule이 적용되어 교환이 보장된다.

### Log Partition Function은 Cumulant Function이다.

위에서 Log Partition Function $$A(\boldsymbol{\eta})$$의 gradient가 $$T(\boldsymbol{x})$$의 기댓값이고 hessian이 공분산 행렬임을 보였다.

이는 사실 $$A(\boldsymbol{\eta})$$가 $$T(\boldsymbol{x})$$의 cumulant function이라는 더 일반적인 사실의 special case이다.

이제 $$T(\boldsymbol{x})$$의 cumulant generating function을 구하기 위해 먼저 $$T(\boldsymbol{x})$$의 moment generating function을 구해보자.

$$
\begin{align*}
    M_{T(\boldsymbol{x})}(\mathbf{t})
    &= \mathbb{E}[\exp(\mathbf{t}^\top T(\boldsymbol{x}))] \\
    &= \int \exp(\mathbf{t}^\top T(\boldsymbol{x})) h(\boldsymbol{x}) \exp(\boldsymbol{\eta}^\top T(\boldsymbol{x}) - A(\boldsymbol{\eta})) d\boldsymbol{x} \\
    &= \int h(\boldsymbol{x}) \exp((\boldsymbol{\eta} + \mathbf{t})^\top T(\boldsymbol{x}) - A(\boldsymbol{\eta})) d\boldsymbol{x} \\
    &= \int h(\boldsymbol{x}) \exp((\boldsymbol{\eta} + \mathbf{t})^\top T(\boldsymbol{x}) - A(\boldsymbol{\eta} + \mathbf{t})) \exp(A(\boldsymbol{\eta} + \mathbf{t})- A(\boldsymbol{\eta})) d\boldsymbol{x} \\
    &= \exp(A(\boldsymbol{\eta} + \mathbf{t})- A(\boldsymbol{\eta})) \underbrace{\int h(\boldsymbol{x}) \exp((\boldsymbol{\eta} + \mathbf{t})^\top T(\boldsymbol{x}) - A(\boldsymbol{\eta} + \mathbf{t})) d\boldsymbol{x}}_{\int p(\boldsymbol{x} \mid \boldsymbol{\eta} + \mathbf{t}) d\boldsymbol{x} = 1 \ \text{if } \boldsymbol{\eta}+\mathbf{t}\in\Omega} \\
    &= \exp(A(\boldsymbol{\eta} + \mathbf{t})- A(\boldsymbol{\eta}))
\end{align*}
$$

따라서 $$T(\boldsymbol{x})$$의 cumulant generating function $$K_{T(\boldsymbol{x})}(\mathbf{t})$$는 $$\boldsymbol{\eta} + \mathbf{t} \in \Omega$$인 모든 $$\mathbf{t}$$에 대해

$$K_{T(\boldsymbol{x})}(\mathbf{t}) = \log M_{T(\boldsymbol{x})}(\mathbf{t}) = A(\boldsymbol{\eta} + \mathbf{t})- A(\boldsymbol{\eta})$$

이다.

이때 $$n$$th cumulant는 $$\boldsymbol{\eta} \in \mathrm{int}(\Omega)$$일 때 $$\nabla_{\mathbf{t}}^n \log M_{T(\boldsymbol{x})}(\mathbf{t}) \vert_{\mathbf{t} = \mathbf{0}}$$인데, 이는 $$A(\boldsymbol{\eta})$$의 $$\boldsymbol{\eta}$$에 대한 $$n$$th derivative와 같다.

따라서 $$A(\boldsymbol{\eta})$$는 $$T(\boldsymbol{x})$$의 cumulant function이다.

이는 $$A(\boldsymbol{\eta})$$의 gradient가 $$T(\boldsymbol{x})$$의 기댓값과 같고, hessian이 공분산 행렬과 같다는 사실과 일치한다.

## 3. Maximum Likelihood Estimation는 Moment Matching과 같다.

Exponential family 분포에서 iid하게 관측된 N개의 데이터 $$\mathcal{D} = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_N\}$$에 대한 joint likelihood는 다음과 같이 표현할 수 있다.

$$
\begin{align*}
    p(\mathcal{D} \mid \boldsymbol{\eta}) 
    &= \prod_{n=1}^N p(\boldsymbol{x}_n \mid \boldsymbol{\eta}) \\
    &= \prod_{n=1}^N h(\boldsymbol{x}_n) \exp \left( \boldsymbol{\eta}^{\top} T(\boldsymbol{x}_n) - A(\boldsymbol{\eta})\right) \\
    &= \left( \prod_{n=1}^N h(\boldsymbol{x}_n) \right) \exp \left( \boldsymbol{\eta}^{\top} \sum_{n=1}^N T(\boldsymbol{x}_n) - N A(\boldsymbol{\eta})\right)
\end{align*}
$$

이제 MLE를 수행하자.

$$
\begin{align*}
    \hat{\boldsymbol{\eta}}_{\text{mle}}
    &= \arg\max_{\boldsymbol{\eta} \in \Omega} p(\mathcal{D} \mid \boldsymbol{\eta}) \\
    &= \arg\max_{\boldsymbol{\eta} \in \Omega} \left( \prod_{n=1}^N h(\boldsymbol{x}_n) \right) \exp \left( \boldsymbol{\eta}^{\top} \sum_{n=1}^N T(\boldsymbol{x}_n) - N A(\boldsymbol{\eta})\right) \\
    &= \arg \max_{\boldsymbol{\eta} \in \Omega} \left( \boldsymbol{\eta}^{\top} \sum_{n=1}^N T(\boldsymbol{x}_n) - N A(\boldsymbol{\eta})\right) \\
\end{align*}
$$

$$-N A(\boldsymbol{\eta})$$는 concave function이고 $$\boldsymbol{\eta}^{\top} \sum_{n=1}^N T(\boldsymbol{x}_n)$$는 linear function이므로 $$\boldsymbol{\eta}^{\top} \sum_{n=1}^N T(\boldsymbol{x}_n) - N A(\boldsymbol{\eta})$$는 concave function이다.

따라서 $$\boldsymbol{\eta}$$에 대해 미분하여 $$\mathbf{0}$$과 같게 놓으면 global maximum을 얻을 수 있다.

$$
\begin{align*}
    \nabla_{\boldsymbol{\eta}} \left( \boldsymbol{\eta}^{\top} \sum_{n=1}^N T(\boldsymbol{x}_n) - N A(\boldsymbol{\eta})\right) 
    &= \sum_{n=1}^N T(\boldsymbol{x}_n) - N \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) \\
    &= \sum_{n=1}^N T(\boldsymbol{x}_n) - N \mathbb{E}[T(\boldsymbol{x})] \\
\end{align*}
$$

이 나온다.

이 결과를 $$\mathbf{0}$$과 같게 놓으면

$$\mathbb{E}[T(\boldsymbol{x})] = \frac{1}{N} \sum_{n=1}^N T(\boldsymbol{x}_n)$$

이 된다.

즉 exponential family 분포에서 MLE는 충분통계량의 기댓값이 관측된 데이터의 충분통계량의 empirical mean과 같도록 하는 $$\boldsymbol{\eta}$$를 찾는 것이다.

### Example 1. Bernoulli distribution

[베르누이 분포의 예시](https://jazzdolphin.github.io/posts/exponential-family1/#example-1-bernoulli-distribution)를 통해 이를 살펴보자.

베르누이 분포에서 $$T(x) = x$$이므로, MLE를 수행하는 것은 moment matching을 아래와 같이 수행하는 것과 같다.

$$
\begin{align*}
    \mathbb{E}[T(x)] &= \frac{1}{N} \sum_{n=1}^N T(x_n) \\
    \mathbb{E}[x] &= \frac{1}{N} \sum_{n=1}^N x_n \\
    \mu= \frac{1}{N} \sum_{n=1}^N x_n
\end{align*}
$$

즉 베르누이 분포에서 MLE는 $$\mu$$가 empirical mean과 같도록 하는 것이다.

### Example 2. Gaussian distribution

[이전 포스트](https://jazzdolphin.github.io/posts/exponential-family1/#example-2-gaussian-distribution)에서 가우시안 분포의 충분통계량이 $$T(x) = \begin{bmatrix}x \\ x^2\end{bmatrix}$$임을 보였다.

Moment matching을 수행하면

$$
\begin{align*}
    \mathbb{E}[T(x)] &= \frac{1}{N} \sum_{n=1}^N T(x_n) \\
    \mathbb{E}\left[\begin{bmatrix}x \\ x^2 \end{bmatrix} 
    \right] &= \frac{1}{N} \sum_{n=1}^N \begin{bmatrix} x_n \\ x_n^2 \end{bmatrix} \\
    \begin{bmatrix} \mu \\ \mu^2 + \sigma^2 \end{bmatrix} &= \frac{1}{N} \sum_{n=1}^N \begin{bmatrix} x_n \\ x_n^2 \end{bmatrix}
\end{align*}
$$

따라서 

$$
\begin{align*}
    \mu &= \frac{1}{N} \sum_{n=1}^N x_n \\
    \sigma^2 &= \frac{1}{N} \sum_{n=1}^N x_n^2 - \mu^2 \\
    &= \frac{1}{N} \sum_{n=1}^N x_n^2 - 2 \mu^2 + \mu^2 \\
    &= \frac{1}{N} \sum_{n=1}^N x_n^2 - \left( \frac{1}{N} \sum_{n=1}^N 2 \mu x_n \right) + \frac{1}{N}\sum_{n=1}^N\mu^2 \\
    &= \frac{1}{N} \sum_{n=1}^N (x_n^2 - 2 x_n \mu + \mu^2) \\
    &= \frac{1}{N} \sum_{n=1}^N (x_n - \mu)^2
\end{align*}
$$

을 얻는다.

이는 $$\mu$$가 empirical mean과 같도록 하는 것이고, $$\sigma^2$$가 empirical variance와 같도록 하는 것이다.

## References
- Murphy, K. P. (2023). [Probabilistic Machine Learning: Advanced Topics.](https://probml.github.io/pml-book/book2.html) MIT press.
