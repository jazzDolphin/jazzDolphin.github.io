---
title: "Principle of Data Reduction 2: Sufficiency Principle"
date: 2026-05-03 21:29:00 +0900
categories: [Statistics]
order: 5
math: true
---

[이전 포스트](https://jazzdolphin.github.io/posts/data-reduction1/)에서 통계량을 통해 data reduction을 수행할 때 $$\theta$$를 inference하는 데 필요한 정보는 보존하면서도 불필요한 정보는 버리는 통계량을 선택하는 기준이 필요하다고 언급하였다.

이번 포스트에서는 그 첫번째 기준인 Sufficiency Principle에 대해 살펴보고자 한다.

## Sufficiency Principle
Parameter $$\theta$$에 대한 충분통계량(sufficient statistic)을 대략적으로 설명하면 샘플이 가진 $$\theta$$에 대한 정보를 모두 담고 있는 통계량이다. 

즉 충분통계량을 제외한 샘플의 다른 추가적인 정보는 $$\theta$$를 inference하는 데 도움이 되지 않는다는 의미이다.

이러한 충분통계량의 개념을 이용해 Principle of Data Reduction의 첫 번째 원칙인 Sufficiency Principle을 다음과 같이 정의할 수 있다.

> **Sufficiency Principle**
> 
> 만약 $$T(\mathbf{X})$$가 $$\theta$$에 대한 충분통계량이라면, $$\theta$$에 대한 모든 inference는 샘플 $$\mathbf{X}$$의 다른 추가적인 정보 없이 $$T(\mathbf{X})$$의 값을 통해서만 이루어져야 한다.
>
> 즉, $$\mathbf{x}$$와 $$\mathbf{y}$$가 $$T(\mathbf{x})=T(\mathbf{y})$$를 만족한다면 $$\theta$$에 대한 inference는 $$\mathbf{X}$$가 $$\mathbf{x}$$로 관측되든 $$\mathbf{y}$$로 관측되든 동일해야 한다는 것이다.
{: .prompt-math }

이제 충분통계량의 개념을 formal하게 정의해보며 Principle of Data Reduction의 첫 번째 원칙인 Sufficiency Principle에 대해 좀 더 자세히 살펴보도록 하자.

## Sufficient Statistic(충분통계량)

> **Definition. Sufficient Statistic(충분통계량)**
>
> 통계량 $$T(\mathbf{X})$$의 값이 주어졌을 때의 샘플 $$\mathbf{X}$$의 조건부 분포가 $$\theta$$에 의존하지 않는다면, $$T(\mathbf{X})$$는 $$\theta$$에 대한 충분통계량이라고 한다.
{: .prompt-math }

이제 실제로 주어진 통계량이 충분통계량인지 어떻게 판별할 수 있는지 살펴보자.

## 충분통계량 판별법

$$T(\mathbf{X})$$가 이산 분포를 따른다고 가정하고 $$t$$가 $$T(\mathbf{X})$$의 image에 속하는 값이라고 하자. (연속 분포의 경우 $$P_\theta(T(\mathbf{X})=t) = 0$$이므로 조건부 확률 $$P_\theta(\mathbf{X}=\mathbf{x} \mid T(\mathbf{X})=t)$$가 단순히 정의되지 않는다.)

어떤 $$T(\mathbf{X})$$가 충분통계량임을 확인하려면 조건부 확률 $$P_\theta(\mathbf{X}=\mathbf{x} \mid T(\mathbf{X})=t)$$이 $$\theta$$에 의존하는지 확인하면 된다.

$$\mathbf{X}=\mathbf{x}$$로 관측되었다고 하자.

$$T(\mathbf{x}) \neq t$$인 경우는 $$P_\theta(\mathbf{X}=\mathbf{x} \mid T(\mathbf{X})=t)$$가 $$\theta$$의 값과 관계 없이 항상 0이다.

그러므로 $$T(\mathbf{x})=t$$인 경우, 즉 $$P_\theta(\mathbf{X}=\mathbf{x} \mid T(\mathbf{X})=T(\mathbf{x}))$$이 $$\theta$$의 값에 따라 변하는지 확인하면 된다.

$$
\begin{align*}
    P_\theta(\mathbf{X}=\mathbf{x} \mid T(\mathbf{X})=T(\mathbf{x})) 
    &= \frac{P_\theta(\mathbf{X}=\mathbf{x} \text{ and } T(\mathbf{X})=T(\mathbf{x}))}{P_\theta(T(\mathbf{X})=T(\mathbf{x}))} \\
    &= \frac{P_\theta(\mathbf{X}=\mathbf{x})}{P_\theta(T(\mathbf{X})=T(\mathbf{x}))} \quad (\because \{\mathbf{X}=\mathbf{x}\} \subseteq \{T(\mathbf{X})=T(\mathbf{x})\}) \\
    &= \frac{p(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} \\
\end{align*}
$$

위 식에서 $$p(\mathbf{x} \mid \theta)$$는 샘플 $$\mathbf{X}$$의 joint pmf이고, $$q(T(\mathbf{x}) \mid \theta)$$는 통계량 $$T(\mathbf{X})$$의 pmf이다.

만약 위 식에서 $$\theta$$가 전부 약분되어 사라진다면 $$T(\mathbf{X})$$는 $$\theta$$에 대한 충분통계량이 된다.

위에서 언급한 것처럼 $$\mathbf{X}$$나 $$T(\mathbf{X})$$가 연속 분포를 따른다면 위의 조건부 확률을 이용한 수식 전개를 따를 수 없지만 Casella 책에 따르면(p.274) $$\frac{p(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$를 이용한 충분통계량의 판별은 동일하게 할 수 있다고 한다.

> **Theorem. 충분통계량 판별법**
> 
> $$p(\mathbf{x} \mid \theta)$$가 $$\mathbf{X}$$의 joint pdf or pmf이고, $$q(t \mid \theta)$$가 $$T(\mathbf{X})$$의 pdf or pmf라고 할 때, 모든 $$\mathbf{x} \in \mathcal{X}$$에 대해 $$\frac{p(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$가 $$\theta$$에 대한 상수함수라면 $$T(\mathbf{X})$$는 $$\theta$$에 대한 충분통계량이 된다.
{: .prompt-math }

### Example 1. Bernoulli distribution
$$X_1, \ldots, X_n \stackrel{i.i.d}{\sim} \mathrm{Ber}(\theta)$$ ($$0 < \theta < 1$$)라 하자.

이제 $$T(\mathbf{X})=\sum_i X_i$$, 즉 샘플의 합이 $$\theta$$에 대한 충분통계량인지 확인해보자.

i.i.d. 가정에 의해 샘플의 joint pmf는 각 pmf의 곱으로 표현되므로 $$p(\mathbf{x} \mid \theta)=\prod_{i=1}^n \theta^{x_i} (1-\theta)^{1-x_i}$$이다.

$$T(\mathbf{X})=t$$는 $$X_i=1$$인 횟수를 세는 것이기에 binomial distribution을 따른다. 따라서 $$q(t \mid \theta) = \binom{n}{t} \theta^t (1-\theta)^{n-t}$$이다.

따라서 $$\frac{p(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$는 다음과 같이 계산된다.

$$
\begin{align*}
    \frac{p(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} 
    &= \frac{\prod_{i=1}^n \theta^{x_i} (1-\theta)^{1-x_i}}{\binom{n}{\sum_i x_i} \theta^{\sum_i x_i} (1-\theta)^{n-\sum_i x_i}} \\
    &= \frac{\prod_{i=1}^n \theta^{x_i} (1-\theta)^{1-x_i}}{\binom{n}{\sum_i x_i} \prod_{i=1}^n \theta^{x_i} (1-\theta)^{1-x_i}} \\
    &= \frac{1}{\binom{n}{\sum_i x_i}}
\end{align*}
$$

위 식에서 $$\theta$$가 모두 약분되어 사라진 것을 확인할 수 있다. 따라서 $$T(\mathbf{X})=\sum_i X_i$$는 $$\theta$$에 대한 충분통계량이 된다.

이는 베르누이 분포의 parameter를 inference할 때 샘플 개별의 정보 (ex: $$X_1=0, X_2=1 \ldots X_n=0$$)는 필요하지 않고 샘플의 합만 알고 있으면 충분하다는 상식적인 결과와 일치한다.

### Example 2. Gaussian distribution with given variance
$$X_1, \ldots, X_n \stackrel{i.i.d}{\sim} \mathcal{N}(\mu, \sigma^2)$$ ($$\sigma^2$$는 주어짐)라 하자.

이제 $$T(\mathbf{X})=\bar{X}$$, 즉 sample mean이 $$\mu$$에 대한 충분통계량인지 확인해보자.

샘플 $$\mathbf{X}$$의 joint pdf는 다음과 같이 표현된다.

$$
\begin{align*}
    f(\mathbf{x} \mid \mu)
    &= \prod_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\
    &= (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\mu)^2}{2\sigma^2} \right) \\
    &= (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x}+\bar{x}-\mu)^2}{2\sigma^2} \right) \\
    &= (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2 + 2(\bar{x} - \mu) \sum_{i=1}^n (x_i - \bar{x})+(\bar{x}-\mu)^2}{2\sigma^2} \right) \\
    &= (2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2 + n(\bar{x}-\mu)^2}{2\sigma^2} \right) \quad (\because \sum_{i=1}^n (x_i - \bar{x})=0) \\
\end{align*}
$$

$$\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$$이므로 $$q(t \mid \mu) = \frac{1}{\sqrt{2\pi \sigma^2/n}} \exp\left(-\frac{(t-\mu)^2}{2\sigma^2/n}\right)$$이다.

따라서 $$\frac{p(\mathbf{x} \mid \mu)}{q(T(\mathbf{x}) \mid \mu)}$$는 다음과 같이 계산된다.

$$
\begin{align*}
    \frac{p(\mathbf{x} \mid \mu)}{q(T(\mathbf{x}) \mid \mu)} 
    &= \frac{(2 \pi \sigma^2)^{-\frac{n}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2 + n(\bar{x}-\mu)^2}{2\sigma^2} \right)}{\frac{1}{\sqrt{2\pi \sigma^2/n}} \exp\left(-\frac{(\bar{x}-\mu)^2}{2\sigma^2/n}\right)} \\
    &= (2\pi\sigma^2)^{-\frac{n-1}{2}} n^{-\frac{1}{2}} \exp\left(-\frac{\sum_{i=1}^n (x_i-\bar{x})^2}{2\sigma^2}\right)
\end{align*}
$$

위 식에서 $$\mu$$가 모두 약분되어 사라진 것을 확인할 수 있다. 따라서 $$T(\mathbf{X})=\bar{X}$$는 $$\mu$$에 대한 충분통계량이 된다.

## 정의를 이용한 충분통계량 판별의 한계
충분통계량의 정의를 이용해 특정 모델의 충분통계량을 찾는 것은 어려울 수 있다.

정의를 이용하면 우리는
1. 먼저 $$T(\mathbf{X})$$를 선택하고
2. $$T(\mathbf{X})$$의 pmf or pdf인 $$q(t \mid \theta)$$를 구한 다음
3. $$\frac{p(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$$가 $$\theta$$에 대한 상수함수인지 확인해야 한다.

위에 예시들에서는 분포가 단순하고 직관적이기에 통계량을 쉽게 선택하고, pmf or pdf도 쉽게 구할 수 있었지만, 복잡한 모델에서는 충분통계량을 찾는 것 자체가 어려울 수 있다.

따라서 충분통계량을 찾는 다른 방법이 필요하다. 다음 포스트에서는 충분통계량을 찾는 다른 방법인 Factorization Theorem에 대해 살펴보도록 하자.

## Reference
- Casella, G., & Berger, R. L. (2002). Statistical inference (2nd ed.). Duxbury.
