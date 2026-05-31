---
title: "Principle of Data Reduction 3: Factorization Theorem"
date: 2026-05-10 13:49:00 +0900
categories: [Statistical Inference, Foundations]
order: 6
math: true
---

[이전 포스트](https://jazzdolphin.github.io/posts/data-reduction2/)에서 Sufficiency Principle에 대해 살펴보았다. 

이번 포스트에서는 충분통계량을 쉽게 구하는 방법인 Factorization Theorem에 대해 알아보자.

## Fisher-Neyman Factorization Theorem

> **Fisher-Neyman Factorization Theorem**
> 
> $$f(\mathbf{x} \mid \theta)$$가 샘플 $$\mathbf{X}$$의 joint pdf 또는 pmf라고 하자.
> 
> $$T(\mathbf{X})$$가 parameter $$\theta$$의 충분통계량이 되기 위한 필요충분조건은 모든 sample points $$\mathbf{x}$$와 parameter points $$\theta$$에 대해 $$g(t \mid \theta)$$ 와 $$h(\mathbf{x})$$가 존재하여 다음과 같이 표현될 수 있어야 한다.
> 
> $$f(\mathbf{x} \mid \theta) = g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})$$
{: .prompt-math }

Factorization Theorem을 사용하기 위해서는 샘플의 joint pdf 또는 pmf를 $$T(\mathbf{X})$$를 통해 $$\theta$$에 대해 의존하는 함수와 $$\theta$$에 대해 상수인 함수로 분해하면 된다.

### 이산 분포에서의 factorization theorem 증명

이산분포에서의 증명을 두 방향으로 나누어 살펴보자.

1. $$T(\mathbf{X})$$가 충분통계량이다. $$\implies$$ $$f(\mathbf{x} \mid \theta)$$는 $$g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})$$로 분해될 수 있다.

    $$T(\mathbf{X})$$를 충분통계량이라고 하자.

    $$
    \begin{align*}
        f(\mathbf{x} \mid \theta) 
        &= P_\theta(\mathbf{X} = \mathbf{x}) \\
        &= P_\theta(\mathbf{X} = \mathbf{x} \text{ and } T(\mathbf{X}) = T(\mathbf{x})) \quad (\because \{\mathbf{X} = \mathbf{x}\} \subseteq \{T(\mathbf{X}) = T(\mathbf{x})\}) \\
        &= \underbrace{P_\theta(T(\mathbf{X}) = T(\mathbf{x}))}_{g(T(\mathbf{x}) \mid \theta)} \underbrace{P(\mathbf{X} = \mathbf{x} \mid T(\mathbf{X}) = T(\mathbf{x}))}_{h(\mathbf{x})} \\
    \end{align*}
    $$

    마지막 줄의 $$P(\mathbf{X} = \mathbf{x} \mid T(\mathbf{X}) = T(\mathbf{x}))$$는 $$T(\mathbf{X})$$를 충분통계량으로 가정했으므로 $$\theta$$에 대해 상수이다.

    따라서 $$T(\mathbf{X})$$가 충분통계량이라고 가정하면 $$f(\mathbf{x} \mid \theta)$$는 모든 $$\mathbf{x}$$와 $$\theta$$에 대해 $$\theta$$에 대한 함수$$(g(T(\mathbf{x}) \mid \theta))$$와 $$\theta$$에 대해 상수인 함수$$(h(\mathbf{x}))$$로 분해될 수 있다.

2. $$f(\mathbf{x} \mid \theta)$$가 $$g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})$$로 분해될 수 있다. $$\implies$$ $$T(\mathbf{X})$$는 충분통계량이다.

    $$f(\mathbf{x} \mid \theta)$$가 $$g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})$$로 분해될 수 있다고 가정하자.
    
    $$q(t \mid \theta) := P_\theta(T(\mathbf{X})=t)$$라고 정의하자. 즉 $$q(t \mid \theta)$$는 $$T(\mathbf{X})$$의 pmf이다.
    
    $$T(\mathbf{X})$$가 충분통계량임을 보이려면 $$\frac{f(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$가 $$\theta$$에 대해 상수임을 보이면 된다.

    $$A_{T(\mathbf{x})}=\{\mathbf{y} : T(\mathbf{y}) = T(\mathbf{x})\}$$라고 하자.

    $$
    \begin{align*}
        q(T(\mathbf{x})\mid \theta) 
        &= P_\theta(T(\mathbf{X}) = T(\mathbf{x})) \\
        &= \sum_{\mathbf{y} \in A_{T(\mathbf{x})}} P_\theta(\mathbf{X} = \mathbf{y}) \quad (\because \{T(\mathbf{X}) = T(\mathbf{x})\} = \bigcup_{\mathbf{y} \in A_{T(\mathbf{x})}} \{\mathbf{X} = \mathbf{y}\}) \\
        &= \sum_{\mathbf{y} \in A_{T(\mathbf{x})}} f(\mathbf{y} \mid \theta)
    \end{align*}
    $$

    따라서 $$\frac{f(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$는 아래와 같이 전개될 수 있다.

    $$
    \begin{align*}
        \frac{f(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} 
        &= \frac{f(\mathbf{x} \mid \theta)}{\sum_{\mathbf{y} \in A_{T(\mathbf{x})}} f(\mathbf{y} \mid \theta)} \\
        &= \frac{g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})}{\sum_{\mathbf{y} \in A_{T(\mathbf{x})}} g(T(\mathbf{y}) \mid \theta) h(\mathbf{y})} \quad (\because \text{factorization 가정}) \\
        &= \frac{g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})}{g(T(\mathbf{x}) \mid \theta) \sum_{\mathbf{y} \in A_{T(\mathbf{x})}} h(\mathbf{y})} \quad (\because \mathbf{y} \in A_{T(\mathbf{x})} \implies T(\mathbf{y}) = T(\mathbf{x})) \\
        &= \frac{h(\mathbf{x})}{\sum_{\mathbf{y} \in A_{T(\mathbf{x})}} h(\mathbf{y})}
    \end{align*}
    $$

    $$\frac{f(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$는 $$\theta$$에 대해 상수이므로 $$T(\mathbf{X})$$는 충분통계량이다.

연속 분포에서의 증명은 측도론 지식이 필요하므로 생략한다.
    
### Example1: Gaussian Distribution with known variance

$$X_1, \ldots, X_n$$이 평균 $$\mu$$와 분산 $$\sigma^2$$를 갖는 가우시안 분포에서 i.i.d.하게 샘플링된 것이라고 하자.

[이전 포스트](https://jazzdolphin.github.io/posts/data-reduction2/#example-2-gaussian-distribution-with-given-variance)에서 joint pdf가 다음과 같이 표현될 수 있음을 보았다.

$$
f(\mathbf{x} \mid \mu) = (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2 + n(\bar{x}-\mu)^2}{2\sigma^2} \right)
$$

$$g(t \mid \mu)$$와 $$h(\mathbf{x})$$를 다음과 같이 정의하자.

$$
g(t \mid \mu) = \exp\left(-\frac{n(t-\mu)^2}{2\sigma^2}\right)
$$

$$
h(\mathbf{x}) = (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2}{2\sigma^2} \right)
$$

$$f(\mathbf{x} \mid \mu) = g(t \mid \mu) h(\mathbf{x})$$으로 나타낼 수 있다.

$$g(t \mid \mu)$$는 $$T(\mathbf{x})$$를 통해서만 $$\mu$$에 의존하고 $$\mathbf{x}$$에 대한 함수인 $$h(\mathbf{x})$$는 $$\mu$$에 대해 상수이다.

따라서 factorization theorem에 의해 $$T(\mathbf{X})=\bar{X}$$는 $$\mu$$에 대한 충분통계량이 된다.

### Example 2: Gaussian Distribution with unknown mean and variance

$$X_1, \ldots, X_n$$이 평균 $$\mu$$와 분산 $$\sigma^2$$를 갖는 가우시안 분포에서 i.i.d.하게 샘플링된 것이라고 하자.

위의 예시와 다르게 이번에는 $$\mu$$와 $$\sigma^2$$가 모두 unknown parameter라고 가정하자. 즉 $$\boldsymbol{\theta} = (\mu, \sigma^2)$$이다.

따라서 factorization Theorem을 사용하기 위해서 $$\mu$$와 $$\sigma^2$$에 대해서 의존하는 함수는 모두 $$g$$ 함수에 포함되어야 한다.

위의 예시에서 본 것 처럼 $$f(\mathbf{x} \mid \mu, \sigma^2)$$는 다음과 같이 표현될 수 있다.

$$
f(\mathbf{x} \mid \mu, \sigma^2) = (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2 + n(\bar{x}-\mu)^2}{2\sigma^2} \right)
$$

이 pdf를 보면 sample mean $$\bar{x}$$와 sample variance $$s^2=\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})^2$$을 통해서만 샘플 $$\mathbf{x}$$에 대해서 의존한다는 사실을 알 수 있다.

$$\mathbf{t}=(t_1, t_2)$$라 할 때 $$g(\mathbf{t} \mid \mu, \sigma^2)$$와 $$h(\mathbf{x})$$를 다음과 같이 정의하자.

$$
\begin{align*}
    g(\mathbf{t} \mid \mu, \sigma^2) &= (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{n(t_1-\mu)^2 + (n-1)t_2}{2\sigma^2}\right) \\
    h(\mathbf{x}) &= 1
\end{align*}
$$

$$T_1(\mathbf{x}) = \bar{x}$$와 $$T_2(\mathbf{x}) = s^2$$로 정의하자.

그렇다면 $$f(\mathbf{x} \mid \mu, \sigma^2)$$는 다음과 같이 표현될 수 있다.

$$
f(\mathbf{x} \mid \mu, \sigma^2) = g((T_1(\mathbf{x}), T_2(\mathbf{x})) \mid \mu, \sigma^2) h(\mathbf{x})
$$

따라서 $$T(\mathbf{X}) = (\bar{X}, S^2)$$는 factorization theorem에 의해 $$\mu$$와 $$\sigma^2$$에 대한 충분통계량이 된다.

이 결과는 gaussian model에서 dataset을 sample mean과 sample variance로 요약하는 것이 parameter $$(\mu, \sigma^2)$$를 inference하는 데 필요한 모든 정보를 담고 있다는 것을 의미한다.

### Example 3: Discrete Uniform Distribution

$$X_1, \ldots, X_n$$이 $$\{1, 2, \ldots, \theta\}$$의 값을 갖는 discrete uniform distribution에서 i.i.d.하게 샘플링된 것이라고 하자. 

그렇다면 각 샘플의 pmf는 다음과 같다.

$$
f(x \mid \theta) = \begin{cases}
    \frac{1}{\theta} & x \in \{1, 2, \ldots, \theta\} \\
    0 & \text{otherwise.}
\end{cases}
$$

그렇다면 $$X_1, \ldots, X_n$$의 joint pmf는 i.i.d.가정에 의해 다음과 같이 표현될 수 있다.

$$
f(\mathbf{x} \mid \theta) = \begin{cases}
    \theta^{-n} & \text{if } x_i\in \{1, \ldots, \theta\} \text{ for all } i=1, \ldots, n \\
    0 & \text{otherwise.}
\end{cases}
$$

$$x_i\in \{1, \ldots, \theta\} \text{ for all } i=1, \ldots, n$$ 이라는 제약조건을 $$x_i \in \mathbb{N} \text{ for } i=1, \ldots, n \text{ and } \max_i x_i \leq \theta$$로 바꿀 수 있다. (여기서 $$\mathbb{N}$$은 자연수 집합을 의미한다.)

따라서 $$f(\mathbf{x} \mid \theta)$$는 다음과 같이 표현될 수 있다.

$$
f(\mathbf{x} \mid \theta) = \begin{cases}
    \theta^{-n} & \text{if } x_i \in \{1, 2, \ldots \} \text{ for } i=1, \ldots, n \text{ and } \max_i x_i \leq \theta \\
    0 & \text{otherwise.}
\end{cases}
$$

Indicator function $$\mathbb{I}(\cdot)$$를 사용하여 $$f(\mathbf{x} \mid \theta)$$를 다음과 같이 표현할 수 있다.

$$
f(\mathbf{x} \mid \theta) = \theta^{-n} \mathbb{I}(\max_i x_i \leq \theta) \prod_{i=1}^n \mathbb{I}(x_i \in \mathbb{N}) 
$$

그렇다면 $$g(t \mid \theta)$$와 $$h(\mathbf{x})$$를 다음과 같이 정의하자.

$$
\begin{align*}
    g(t \mid \theta) &= \theta^{-n} \mathbb{I}(t \leq \theta) \\
    h(\mathbf{x}) &= \prod_{i=1}^n \mathbb{I}(x_i \in \mathbb{N}) 
\end{align*}
$$

따라서 $$T(\mathbf{X}) = \max_i X_i$$는 factorization theorem에 의해 $$\theta$$에 대한 충분통계량이 된다.

## Reference
- Casella, G., & Berger, R. L. (2002). Statistical inference (2nd ed.). Duxbury.
