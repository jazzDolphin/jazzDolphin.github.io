---
title: "Principle of Data Reduction 4: Likelihood Principle"
date: 2026-05-17 23:55:00 +0900
categories: [Statistical Inference, Foundations]
order: 7
math: true
---

이번 포스트에서는 Likelihood Principle에 대해 알아보겠다.

이 포스트의 내용과 증명은 직관적인 이해를 위해 이산 분포에 대해서만 설명하겠다. 연속 분포에 대해서도 동일한 결과가 성립하지만 추가적인 수학적인 도구가 필요하다.

## Likelihood Principle

Likelihood 함수의 정의는 [이전 포스트](https://jazzdolphin.github.io/posts/mle/#likelihood-%EC%9A%B0%EB%8F%84)에서 확인할 수 있다.

> **Likelihood Principle**
> 
> $$\mathbf{x}$$와 $$\mathbf{y}$$가 두 sample point이고, 모든 $$\theta$$에 대해 $$L(\theta \mid \mathbf{x})$$가 $$L(\theta \mid \mathbf{y})$$에 비례한다고 하자. 
> 
> 즉, 어떤 $$\theta$$에 의존하지 않는 상수 $$C(\mathbf{x}, \mathbf{y})$$가 존재하여
>
> $$L(\theta \mid \mathbf{x}) = C(\mathbf{x}, \mathbf{y}) L(\theta \mid \mathbf{y}) \quad \text{for all } \theta$$
>
> 를 만족하면, $$\mathbf{x}$$와 $$\mathbf{y}$$로부터 도출되는 결론은 동일해야 한다.
{: .prompt-math }

Likelihood Principle이 직관적으로 어떤 의미를 갖는지 알아보자.

만약 $$C(\mathbf{x}, \mathbf{y})$$가 1이라면 $$\mathbf{x}$$와 $$\mathbf{y}$$는 likelihood 값이 동일하므로 당연히 같은 결론이 도출되어야 한다. 

하지만 Likelihood Principle은 $$C(\mathbf{x}, \mathbf{y})$$가 1이 아니더라도, 즉 서로 다른 두 sample point에 대해 likelihood 함수가 비례하는 경우에도 같은 결론이 도출되어야 한다고 주장한다.

만약 $$L(\theta_2 \mid \mathbf{x}) = 2 L(\theta_1 \mid \mathbf{x})$$라면 $$\mathbf{x}$$로부터 도출되는 결론은 $$\theta_2$$가 $$\theta_1$$보다 두 배 그럴듯하다는 것이다.

Likelihood Principle이 성립한다면 $$L(\theta_2 \mid \mathbf{y}) = 2 L(\theta_1 \mid \mathbf{y})$$도 성립해야 한다. (Likelihood Principle을 직관적으로 생각해보면 $$L(\theta \mid \mathbf{y})$$의 그래프와 $$L(\theta \mid \mathbf{x})$$의 그래프가 shape는 동일하지만 scale만 다르다는 것이다.)

즉 $$\mathbf{y}$$를 통해서도 $$\theta_2$$가 $$\theta_1$$보다 두 배 그럴듯하다는 동일한 결론을 얻어야 한다.

## Birnbaum's Theorem

위에서는 Likelihood Principle이 어떤 것을 주장하는지를 살펴보았다..

이번 섹션에서는 Formal Sufficiency Principle과 Conditionality Principle이라는 더 기본적이고 직관적인 원리로부터 Formal Likelihood Principle이 도출된다는 Birnbaum's theorem에 대해 알아보겠다.

이전 포스트의 [Sufficiency Principle](https://jazzdolphin.github.io/posts/data-reduction2/)이나 방금 살펴본 Likelihood Principle은 같은 experiment에서 관측된 sample points를 다루는 것이었다. 

하지만 서로 다른 experiment에서 관측된 sample points에 대한 결론을 비교하려면, experiment와 그 결과로부터 도출되는 결론(evidence) 자체를 수학적으로 정의할 필요가 있다. 이를 통해 Sufficiency Principle과 Likelihood Principle을 formal하게 재정의할 수 있다.

### Experiment와 Evidence
먼저 experiment와 evidence를 formal하게 정의해보자.

Experiment E는 triple $$(\mathbf{X}, \theta, \{f(\mathbf{x} \mid \theta)\})$$로 정의된다.

여기서 $$\mathbf{X}$$는 parameter space $$\Theta$$에 있는 어떤 $$\theta$$에 대해 pmf $$f(\mathbf{x} \mid \theta)$$를 가지는 random vector이다.

Experiment E를 수행했음을 알고 $$\mathbf{X}=\mathbf{x}$$가 관측되었다면 $$\theta$$에 대한 어떠한 결론(ex: 점추정, confidence interval 등등)을 내릴 것이다. 

이렇게 내린 결론 즉 evidence를 $$\mathrm{Ev}(E, \mathbf{x})$$라고 하자.

### Evidence function의 예시

Experiment E가 $$X_1, \ldots, X_n$$을 $$\sigma^2$$를 알고 있는 가우시안 분포 $$\mathcal{N}(\mu, \sigma^2)$$에서 i.i.d. 관측하는 것이라고 하자.

Sample mean $$\bar{X}$$가 $$\mu$$에 대한 충분통계량이고, $$\mathbb{E}[\bar{X}] = \mu$$이므로 측정값 $$\bar{X}=\bar{x}$$를 $$\mu$$에 대한 점추정으로 사용할 수 있다.

추정의 정확도를 보고하기 위해 $$\bar{X}$$의 표준편차 즉 standard error인 $$\sigma / \sqrt{n}$$을 함께 제공한다. 

그렇다면 $$\mathrm{Ev}(E, \mathbf{x})$$는 점추정과 standard error의 쌍인 $$(\bar{x}, \sigma / \sqrt{n})$$이 될 것이다.

관측된 sample $$\mathbf{x}$$로부터 얻을 수 있는 정보 ($$\bar{x}$$)와 $$E$$에 대한 정보($$\sigma / \sqrt{n}$$)가 결합된 형태로 evidence가 표현된 것이다.

### Formal Sufficiency Principle

이제 [지난 포스트](https://jazzdolphin.github.io/posts/data-reduction2/)에서 배운 sufficiency principle을 experiment와 evidence function을 도입하여 formal하게 표현해보자.

> **Formal Sufficiency Principle**
> Experiment $$E=(\mathbf{X}, \theta, \{f(\mathbf{x} \mid \theta)\})$$가 있고, $$T(\mathbf{X})$$가 $$\theta$$에 대한 충분통계량이라고 하자.
>
> 만약 $$\mathbf{x}$$와 $$\mathbf{y}$$가 $$T(\mathbf{x})=T(\mathbf{y})$$인 sample points라면, $$\mathrm{Ev}(E, \mathbf{x}) = \mathrm{Ev}(E, \mathbf{y})$$가 되어야 한다.
{: .prompt-math }

기존의 Sufficiency Principle과 달리 Formal Sufficiency Principle은 experiment $$E$$를 정의하고 그 결론 $$\mathrm{Ev}$$가 $$E$$와 sample point $$\mathbf{x}$$에 의존하는 함수임을 명시적으로 표현하였다.

### Conditionality Principle

> **Conditionality Principle**
> 두 experiment $$E_1=(\mathbf{X}_1, \theta, \{f(\mathbf{x}_1 \mid \theta)\})$$와 $$E_2=(\mathbf{X}_2, \theta, \{f(\mathbf{x}_2 \mid \theta)\})$$가 있다고 하자. 두 experiment에서 미지의 parameter $$\theta$$는 동일해야 하지만 그 외의 구조(random vector와 pmf등)는 달라도 된다.
>
> 이제 random variable $$J$$를 관측하는 mixed experiment를 생각해보자. 이때 $$P(J=1) = P(J=2) = \frac{1}{2}$$이고 $$J$$는 $$\theta, \mathbf{X}_1, \mathbf{X}_2$$와 독립이라고 하자. $$J=j$$가 관측되면 experiment $$E_j$$가 수행된다고 가정하자.
>
> 이 mixed experiment를 formal하게 정의하면  $$E^*=(\mathbf{X}^*, \theta, \{f^*(\mathbf{x}^* \mid \theta)\})$$이며, $$\mathbf{X}^* =(j, \mathbf{X}_j)$$이고 $$f^*(\mathbf{x}^* \mid \theta) = f^*((j, \mathbf{x}_j) \mid \theta) = \frac{1}{2}f_j(\mathbf{x}_j \mid \theta)$$이다. 그렇다면 
> 
> $$\mathrm{Ev}(E^*, (j, \mathbf{x}_j)) = \mathrm{Ev}(E_j, \mathbf{x}_j)$$
{: .prompt-math }

Conditionality Principle은 동전을 던져 어떤 experiment를 수행할지 결정하든, 그냥 experiment를 결정하여 수행하든 결국 같은 experiment를 수행했다면 $$\mathrm{Ev}$$는 같아야 한다는 아주 직관적인 원리이다.

### Binomial/Negative Binomial experiment의 예시

Parameter $$p (0 < p < 1)$$를 inference하고자 하는 상황을 생각해보자. 여기서 $$p$$는 어떤 동전을 던졌을 때 앞면이 나올 확률이다.

$$E_1$$은 동전을 20번 던지고 나온 앞면의 수를 기록하는 실험이다. 즉 $$E_1$$은 binomial experiment이고, $$\{f_1(x_1 \mid p)\}$$는 $$\mathrm{binomial}(20, p)$$ pmf의 family이다. 

$$E_2$$는 동전을 앞면이 7번 나올 때까지 던지고, 7번째 앞면이 나오기 전에 나온 뒷면의 수를 기록하는 실험이다. 즉 $$E_2$$는 negative binomial experiment이다.

실험자는 random variable $$J$$를 관측해 어떤 experiment가 수행될지 결정한다고 하자.

이때 $$J=2$$가 관측되어 $$E_2$$가 수행되었다고 하자. 20번째 시행에서 7번째 앞면이 나왔다고 하자. 즉 $$x_2 = 13$$이다.

Conditionality Principle에 따르면 $$J=2$$를 관측하여 $$E_2$$를 수행했을 때의 $$p$$에 대한 evidence인 $$\mathrm{Ev}(E^*, (2, 13))$$는 그냥 $$E_2$$를 수행했을 때의 $$p$$에 대한 evidence인 $$\mathrm{Ev}(E_2, 13)$$와 동일해야 한다.

### Formal Likelihood Principle

> **Formal Likelihood Principle**
> 
> Experiment $$E_1=(\mathbf{X}_1, \theta, \{f_1(\mathbf{x}_1 \mid \theta)\})$$와 $$E_2=(\mathbf{X}_2, \theta, \{f_2(\mathbf{x}_2 \mid \theta)\})$$가 있다고 하자. 두 experiment에서 미지의 parameter $$\theta$$는 동일하다.
>
> $$\mathbf{x}^*_1, \mathbf{x}^*_2$$가 각각 $$E_1$$과 $$E_2$$에서 관측된 sample point라고 하자. 만약 모든 $$\theta$$에 대해 
>
> $$L(\theta \mid \mathbf{x}^*_2) = C \, L(\theta \mid \mathbf{x}^*_1)$$
> 
> 성립한다고 하자. (여기서 $$C$$는 $$\mathbf{x}^*_1$$과 $$\mathbf{x}^*_2$$에만 의존할 수 있고 $$\theta$$에는 의존하지 않는 상수이다.) 그렇다면 아래가 성립한다.
>
> $$\mathrm{Ev}(E_1, \mathbf{x}^*_1) = \mathrm{Ev}(E_2, \mathbf{x}^*_2)$$
{: .prompt-math }

글 처음에 설명한 Likelihood Principle은 같은 experiment의 두 sample points $$\mathbf{x}$$와 $$\mathbf{y}$$에 대하여 같은 결론이 도출되어야 한다고 주장한다.

반면에 Formal Likelihood Principle은 서로 다른 experiment에서의 각각의 sample point $$\mathbf{x}^*_1$$과 $$\mathbf{x}^*_2$$에 대하여 같은 evidence를 갖는다고 주장한다.

Formal Likelihood Principle에서 $$E_2$$를 $$E_1$$과 동일한 실험의 복제로 생각한다면 Formal Likelihood Principle은 글 처음에 설명한 Likelihood Principle과 같은 의미를 갖는다.

> **Likelihood Principle Corollary**
> 
> 만약 $$E=(\mathbf{X}, \theta, \{f(\mathbf{x} \mid \theta) \})$$가 experiment라면 $$\mathrm{Ev}(E, \mathbf{x})$$는 $$E$$와 $$\mathbf{x}$$에 $$L(\theta \mid \mathbf{x})$$를 통해서만 의존해야 한다.
{: .prompt-math }

위 Corollary의 증명은 아래와 같다.

어떤 experiment $$E=(\mathbf{X},\theta,\{f(\mathbf{x} \mid \theta)\})$$에서 관측된 두 sample points $$\mathbf{x}_1$$과 $$\mathbf{x}_2$$가 존재한다고 가정하자. 

그리고 parameter $$\theta$$에 의존하지 않는 어떤 상수 $$C$$에 대해 다음이 성립한다고 가정하자.

$$L(\theta \mid \mathbf{x}_2) = C \, L(\theta \mid \mathbf{x}_1) \quad \text{for all } \theta$$

$$E_1$$과 $$E_2$$를 $$E$$의 복제본으로 가정하자. 즉, $$E_1 = E_2 = E$$이다. 그리고 $$\mathbf{x}^*_1 = \mathbf{x}_1$$과 $$\mathbf{x}^*_2 = \mathbf{x}_2$$라고 하자.

그러면 Formal Likelihood Principle에 의해 

$$\mathrm{Ev}(E_1, \mathbf{x}^*_1) = \mathrm{Ev}(E_2, \mathbf{x}^*_2)$$

가 성립한다. 즉 

$$\mathrm{Ev}(E, \mathbf{x}_1) = \mathrm{Ev}(E, \mathbf{x}_2)$$

가 성립한다.

### Birnbaum's Theorem

> **Birnbaum's Theorem**
> 
> Formal Sufficiency Principle과 Conditionality Principle이 함께 참인 것과 Formal Likelihood Principle도 참인 것은 동치이다.
{: .prompt-math }

Birnbaum's Theorem의 증명은 다음과 같다.

1. Formal Sufficiency Principle & Conditionality Principle $$\implies$$ Formal Likelihood Principle

    FSP와 CP가 참이라고 가정할 때 FLP가 참임을 보이자

    $$\mathbf{x}^*_1, \mathbf{x}^*_2$$가 각각 $$E_1$$과 $$E_2$$에서 관측된 sample points라고 하자.

    그리고 $$L(\theta \mid \mathbf{x}^*_2) = C \, L(\theta \mid \mathbf{x}^*_1)$$이 성립한다고 하자.

    이제 mixed experiment $$E^*$$를 생각해보자. $$E^*$$는 conditionality principle의 정의에서 도입한 mixed experiment이다.

    $$E^*$$에 대한 통계량 $$T$$를 다음과 같이 정의하자.

    $$
    T(j, \mathbf{x}_j) = \begin{cases}
        (1, \mathbf{x}^*_1) & \text{if } (j, \mathbf{x}_j) = (2, \mathbf{x}^*_2) \\
        (j, \mathbf{x}_j) & \text{otherwise}
    \end{cases}
    $$

    즉 $$E^*$$에서 관측된 sample point가 $$(2, \mathbf{x}^*_2)$$인 경우에는 $$(1, \mathbf{x}^*_1)$$을 반환하고 그렇지 않은 경우에는 관측된 sample point를 그대로 반환하는 통계량이다.

    이제 $$T$$가 $$E^*$$에 대한 충분통계량임을 보이자.

    [factorization theorem](https://jazzdolphin.github.io/posts/data-reduction3/)에 의해 $$T$$가 충분통계량이 되기 위한 필요충분조건은 다음과 같다.

    $$f^*((j, \mathbf{x}_j) \mid \theta) = g(T(j, \mathbf{x}_j) \mid \theta) h(j, \mathbf{x}_j) \quad \text{for all } (j, \mathbf{x}_j)$$

    이제 모든 $$(j, \mathbf{x}_j)$$에 대해 위의 식이 성립함을 보이자.

    1. $$(j, \mathbf{x}_j) \neq (2, \mathbf{x}^*_2)$$인 경우

        $$g(T(j, \mathbf{x}_j) \mid \theta) = f^*(T(j, \mathbf{x}_j) \mid \theta)$$으로 정의하고, $$h(j, \mathbf{x}_j) = 1$$로 정의하면 $$T(j, \mathbf{x}_j) = (j, \mathbf{x}_j)$$이므로

        $$f^*((j, \mathbf{x}_j) \mid \theta) = \underbrace{g(T(j, \mathbf{x}_j) \mid \theta)}_{f^*((j, \mathbf{x}_j) \mid \theta)} \underbrace{h(j, \mathbf{x}_j)}_{1}$$        
        이 성립한다.

    2. $$(j, \mathbf{x}_j) = (2, \mathbf{x}^*_2)$$인 경우

        $$
        \begin{align*}
            f^*((j, \mathbf{x}_j) \mid \theta) 
            &= f^*((2, \mathbf{x}^*_2) \mid \theta) \\
            &= \frac{1}{2} \underbrace{f_2(\mathbf{x}^*_2 \mid \theta)}_{L(\theta \mid \mathbf{x}^*_2)} \\
            &= \frac{1}{2} C \underbrace{f_1(\mathbf{x}^*_1 \mid \theta)}_{L(\theta \mid \mathbf{x}^*_1)} \quad \because \theta \text{ 에 의존하지 않는 상수 } C \text{ 존재 가정} \\
            &= C f^*((1, \mathbf{x}^*_1) \mid \theta) \\
        \end{align*}
        $$

        $$g(T(j, \mathbf{x}_j) \mid \theta) = f^*(T(j, \mathbf{x}_j) \mid \theta)$$으로 정의하고, $$h(j, \mathbf{x}_j) = C$$로 정의하면 $$T(j, \mathbf{x}_j) = (1, \mathbf{x}^*_1)$$이므로

        $$f^*((j, \mathbf{x}_j) \mid \theta) = \underbrace{g(T(j, \mathbf{x}_j) \mid \theta)}_{f^*((1, \mathbf{x}^*_1) \mid \theta)} \underbrace{h(j, \mathbf{x}_j)}_{C}$$

        이 성립한다.

    즉 아래와 같이 $$g$$와 $$h$$를 정의하면 

    $$
    \begin{align*}
        g(t \mid \theta) &= f^*(t \mid \theta) \quad \text{for all } t \in \mathrm{Range}(T) \\
        h(j, \mathbf{x}_j) &= \begin{cases}
            C & \text{if } (j, \mathbf{x}_j) = (2, \mathbf{x}^*_2) \\
            1 & \text{otherwise}
        \end{cases}
    \end{align*}
    $$

    모든 $$(j, \mathbf{x}_j)$$에 대해 $$f^*((j, \mathbf{x}_j) \mid \theta) = g(T(j, \mathbf{x}_j) \mid \theta) h(j, \mathbf{x}_j)$$가 성립하므로 $$T$$는 $$E^*$$에 대한 충분통계량이다.

    그렇다면 $$T(1, \mathbf{x}^*_1) = T(2, \mathbf{x}^*_2)$$이므로
    
    Formal Sufficiency Principle에 의해 $$\mathrm{Ev}(E^*, (1, \mathbf{x}^*_1)) = \mathrm{Ev}(E^*, (2, \mathbf{x}^*_2))$$가 성립한다.

    그렇다면 Conditionality Principle에 의해 
    
    $$
    \begin{align*}
        \mathrm{Ev}(E^*, (1, \mathbf{x}^*_1)) &= \mathrm{Ev}(E_1, \mathbf{x}^*_1)\\
        \mathrm{Ev}(E^*, (2, \mathbf{x}^*_2)) &= \mathrm{Ev}(E_2, \mathbf{x}^*_2)
    \end{align*}
    $$

    이므로 

    $$\mathrm{Ev}(E_1, \mathbf{x}^*_1) = \mathrm{Ev}(E_2, \mathbf{x}^*_2)$$가 성립하여 Formal Likelihood Principle이 참임을 알 수 있다.

2. Formal Likelihood Principle $$\implies$$ Formal Sufficiency Principle & Conditionality Principle

    Formal Likelihood Principle이 참이라고 가정하자.

    두 experiment $$E^*=(\mathbf{X}^*, \theta, \{f^*(\mathbf{x}^* \mid \theta)\})$$와 $$E_j=(\mathbf{X}_j, \theta, \{f_j(\mathbf{x}_j \mid \theta)\})$$가 있다고 하자. $$E^*$$는 conditionality principle의 정의에서 도입한 mixed experiment이다.

    이때 $$f^*((j, \mathbf{x}_j) \mid \theta) = \frac{1}{2} f_j(\mathbf{x}_j \mid \theta)$$이므로 $$L(\theta \mid (j, \mathbf{x}_j)) = \frac{1}{2} L(\theta \mid \mathbf{x}_j)$$이다.

    $$\frac{1}{2}$$는 $$\theta$$에 대한 상수이므로 Formal Likelihood Principle에 의해 $$\mathrm{Ev}(E^*, (j, \mathbf{x}_j)) = \mathrm{Ev}(E_j, \mathbf{x}_j)$$가 성립한다. 따라서 Conditionality Principle이 참임을 알 수 있다.

    이제 Formal Sufficiency Principle이 참임을 보이자.

    통계량 $$T$$가 임의의 experiment $$E$$에 대한 충분통계량이라고 하고 $$\mathbf{x}$$와 $$\mathbf{y}$$가 $$T(\mathbf{x}) = T(\mathbf{y})$$인 sample points라고 하자. 

    그러면 factorization theorem에 의해 $$f(\mathbf{x} \mid \theta) = g(T(\mathbf{x}) \mid \theta) h(\mathbf{x}), f(\mathbf{y} \mid \theta) = g(T(\mathbf{y}) \mid \theta) h(\mathbf{y})$$인 함수 $$g$$와 $$h$$가 존재한다. 
    
    $$T(\mathbf{x}) = T(\mathbf{y})$$이므로 $$f(\mathbf{y} \mid \theta) > 0$$ 인 모든 sample point $$\mathbf{y}$$에 대해 아래가 성립한다.

    $$
    \begin{align*}
        \frac{f(\mathbf{x} \mid \theta)}{f(\mathbf{y} \mid \theta)}
        &= \frac{g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})}{g(T(\mathbf{y}) \mid \theta) h(\mathbf{y})} \\
        &= \frac{h(\mathbf{x})}{h(\mathbf{y})} 
    \end{align*}
    $$

    위 결과를 정리하면

    $$\underbrace{f(\mathbf{x} \mid \theta)}_{L(\theta \mid \mathbf{x})} = \underbrace{\frac{h(\mathbf{x})}{h(\mathbf{y})}}_{C} \underbrace{f(\mathbf{y} \mid \theta)}_{L(\theta \mid \mathbf{y})}$$

    $$L(\theta \mid \mathbf{x})$$와 $$L(\theta \mid \mathbf{y})$$가 $$\theta$$에 대한 상수 $$C = \frac{h(\mathbf{x})}{h(\mathbf{y})}$$에 비례하므로 Formal Likelihood Principle에 의해 $$\mathrm{Ev}(E, \mathbf{x}) = \mathrm{Ev}(E, \mathbf{y})$$가 성립한다. 이를 통해 Formal Sufficiency Principle이 참임을 알 수 있다.

따라서 Formal Sufficiency Principle과 Conditionality Principle이 함께 참인 것과 Formal Likelihood Principle도 참인 것은 동치이다.

### Binomial/Negative Binomial experiment의 예시 (cont.)

[위의 예시](https://jazzdolphin.github.io/posts/data-reduction4/#binomialnegative-binomial-experiment의-예시)와 동일하게 binomial experiment $$E_1$$과 negative binomial experiment $$E_2$$가 있다고 하자.

$$E_1$$에서는 $$x_1 = 7$$이 관측되었고, $$E_2$$에서는 $$x_2 = 13$$이 관측되었다고 하자.

그렇다면 두 experiment에서 관측된 sample points $$x_1$$과 $$x_2$$에 대한 likelihood function은 아래와 같다.

$$
\begin{align*}
    L(p \mid x_1) &= \binom{20}{7} p^7 (1-p)^{13} \\
    L(p \mid x_2) &= \binom{19}{6} p^7 (1-p)^{13}
\end{align*}
$$

($$L(p \mid x_2)$$는 19번의 시행에서는 정확하게 6번의 앞면이 나와야 하고, 마지막 20번째 시행에서는 앞면이 나와야 하므로 $$\binom{19}{6} p^6 (1-p)^{13} \cdot p$$가 된다.)

두 likelihood function이 parameter $$p$$에 의존하지 않는 상수 $$C = \frac{\binom{19}{6}}{\binom{20}{7}}$$에 대해 $$L(p \mid x_2) = C \, L(p \mid x_1)$$을 만족하므로  Formal Likelihood Principle에 의해 $$\mathrm{Ev}(E_1, 7) = \mathrm{Ev}(E_2, 13)$$가 성립한다.

이 결과는 동전을 20번 던져서 종료된 것인지, 아니면 7번째 앞면이 나올 때까지 던져서 종료된 것인지가 $$p$$에 대한 evidence에 영향을 미치지 않는다는 것을 의미한다.

따라서 Likelihood Principle은 sample이 어떤 stopping rule에 의해 얻어졌는지를 무시하고, 오직 likelihood function이 parameter에 무관한 상수에 비례하는지만으로 evidence의 동일성을 판단할 수 있다고 주장한다.

위에서는 예시에 FLP를 직접 적용해 결론을 얻었다. 이번에는 Birnbaum's theorem 증명의 FSP & CP $$\implies$$ FLP 부분이 구체적으로 어떤 의미를 갖는지 살펴보자.

Mixed experiment $$E^*$$에서의 통계량 $$T$$를 Birnbaum's theorem의 증명에서와 같이 정의하자.

$$
T(j, \mathbf{x}_j) = \begin{cases}
    (1, 7) & \text{if } (j, \mathbf{x}_j) = (2, 13) \\
    (j, \mathbf{x}_j) & \text{otherwise}
\end{cases}
$$

$$T$$는 증명에서 케이스를 나눠서 보인 것처럼 mixed experiment $$E^*$$에 대한 충분통계량이다.

$$T=(1,7)$$이 관측되었을 때, 충분통계량 $$T$$만으로는 다음 두 경우를 구분할 수 없다:

- $$E_1$$(binomial)이 수행되어 20번 시행 중 앞면이 7번 나온 경우
- $$E_2$$(negative binomial)가 수행되어 7번째 앞면이 20번째 시행에 나온 경우

즉 $$T=(1, 7)$$ 라는 정보만을 사용하면 어느 실험이 수행되었는지 알 수 없는 것이다.

$$E_1$$에서는 시행횟수 20이 고정되어있지만 $$E_2$$에서는 앞면의 횟수 7이 고정되게 experiment가 설계되었다. 그러나 $$T$$의 관점에서는 이 둘이 구분 불가능하므로 충분통계량 $$T$$만을 사용한다면 stopping rule에 관한 정보가 사라지게 된다. 

이것이 바로 Birnbaum's theorem 증명에서 $$T$$를 likelihood function이 비례하는 sample points가 동일한 $$T$$값으로 mapping되게 정의한 이유이다.

Formal Sufficiency Principle에 의해 충분통계량이 같은 sample points는 같은 evidence를 가져야 하므로, $$T(1,7)=T(2,13)=(1,7)$$라는 사실로부터
$$\mathrm{Ev}(E^*, (1, 7)) = \mathrm{Ev}(E^*, (2, 13))$$가 도출된다. 

여기에 Conditionality Principle을 적용하면 $$\mathrm{Ev}(E_1, 7) = \mathrm{Ev}(E_2, 13)$$이 되어 FSP & CP $$\implies$$ FLP임을 예시로 확인할 수 있다.

## Formal Likelihood Principle에 대한 논쟁

p-value를 포함한 많은 통계적 추론 방법은 Formal Likelihood Principle을 만족하지 않는다. 

### p-value가 Formal Likelihood Principle을 만족하지 않는 예시

p-value가 Formal Likelihood Principle을 만족하지 않는 예시로 동전이 앞면과 뒷면이 나올 확률이 같은지 검정하는 상황을 생각해보자.

- Parameter($$p$$): 동전을 던졌을 때 앞면이 나올 확률
- 귀무가설($$H_0$$): 동전은 앞면과 뒷면이 나올 확률이 같다. 즉 $$p=0.5$$이다.
- 대립가설($$H_1$$): 동전은 앞면이 나올 확률이 뒷면이 나올 확률보다 크다. 즉 $$p > 0.5$$이다.
- p-value가 0.05보다 작은 경우에 귀무가설을 기각한다.

가설 검정을 위해 experiment를 수행하여 앞면이 9번, 뒷면이 3번 나왔다고 하자. 그렇다면 실험 설계에는 아래와 같은 여러 개의 시나리오를 고려할 수 있다.

- $$E_1$$: 동전을 던지는 횟수를 12번으로 고정하여 동전을 던져서 나온 앞면의 수를 기록하는 실험이다. 즉 $$E_1$$은 binomial experiment이다.
  - $$x_1=9$$가 관측되었다고 하자. 즉 12번의 시행에서 9번의 앞면이 나왔다고 하자.
  - $$L(p \mid x_1)=\binom{12}{9} p^9 (1-p)^3$$
- $$E_2$$: 3번째 뒷면이 나올 때까지 동전을 던지는 실험이다. 즉 $$E_2$$는 negative binomial experiment이다.
  - $$x_2=9$$가 관측되었다고 하자. 즉 3번째 뒷면이 나오기 전에 9번의 앞면이 나왔다고 하자.
  - $$L(p \mid x_2)=\binom{11}{9} p^9 (1-p)^3$$
  
두 실험의 likelihood function은 $$p$$에 대한 상수 $$C=\frac{\binom{11}{9}}{\binom{12}{9}}$$에 대해 비례하므로 Formal Likelihood Principle이 성립한다면 $$\mathrm{Ev}(E_1, 9) = \mathrm{Ev}(E_2, 9)$$가 되어야 한다.

이제 각 실험의 p-value를 계산해보자.

- $$E_1$$의 p-value는 아래와 같이 계산된다.
  
    $$P(X_1 \geq 9 \mid p=0.5) = \sum_{k=9}^{12}\binom{12}{k} 0.5^k (1-0.5)^{12-k}\approx 0.073$$

- $$E_2$$의 p-value는 아래와 같이 계산된다.
    
    $$P(X_2 \geq 9 \mid p=0.5) = \sum_{k=9}^\infty \binom{k+2}{k} 0.5^k (1-0.5)^3 \approx 0.0327$$이다.
  
따라서 $$E_1$$에서는 p-value가 0.05보다 크므로 귀무가설을 기각하지 않지만, $$E_2$$에서는 p-value가 0.05보다 작으므로 귀무가설을 기각한다.

동전의 관측 결과가 동일해도 실험자가 어떤 의도로 실험을 설계하는지에 따라 evidence가 달라지므로 FLP를 위반하는 것이다.

p-value는 귀무가설이 참일 때 관측 결과보다 더 극단적인 결과가 나올 확률이므로, sample space가 달라지면 p-value도 달라질 수밖에 없다.

$$E_1$$에서는 $$X_1$$이 12번의 시행에서 나온 앞면의 횟수이므로 sample space가 $$\{0, 1, \ldots, 12\}$$이다. 

반면에 $$E_2$$에서는 $$X_2$$가 3번째 뒷면이 나오기까지의 앞면의 횟수이므로 sample space가 $$\{0, 1, 2, \ldots\}$$이다. 

이는 stopping rule이 sample space에 영향을 미쳐 p-value가 달라지는 원인이 된다.

### Formal Likelihood Principle에 대한 비판

왜 Formal Likelihood Principle에 대한 논쟁이 있는 것일까?

Casella & Berger의 책에서는 아래와 같은 이유가 있다고 설명한다.

- Formal Sufficiency Principle은 직관적이고 당연하다.
- Conditionality Principle도 직관적이고 당연하다.
- Birnbaum's theorem에 의해 FSP & CP $$\Leftrightarrow$$ FLP이다.
- 많은 통계적 추론 방법(ex: p-value)이 FLP를 위반한다. 이것은 이들이 FSP나 CP를 위반하는 것이다.

따라서 통계학자들은 1. FSP나 CP 중에 보편적이지 않은 원리가 있거나 2. Birnbaum's theorem의 증명에 문제가 있다고 주장하거나 3. FLP를 위반하는 통계적 추론 방법을 거부해야 한다고 주장한다.

3번 입장을 받아들이는 통계학자들(Bayesian 및 likelihoodist 학자들)은 FLP를 받아들이며, p-value 기반 추론 등 FLP를 위반하는 통계적 추론을 거부한다. 이 관점에서는 Birnbaum's Theorem이 frequentist 추론에 대한 근본적 비판의 도구가 된다. 

반면 frequentist 학자들은 1번 또는 2번 입장을 통해 자신들의 추론 방법을 정당화하려 한다.

1. FSP가 항상 성립하지는 않는다는 주장

    어떤 통계량 $$T$$가 충분통계량이 되려면 factorization theorem에 의해 $$f(\mathbf{x} \mid \theta) = g(T(\mathbf{x}) \mid \theta) h(\mathbf{x})$$로 분해되어야 한다.

    따라서 충분통계량이라는 개념 자체가 모델의 family $$\{f(\mathbf{x} \mid \theta) : \theta \in \Theta\}$$를 가정한 상태에서 정의되는 것이다.

    이는 동일한 sample point $$\mathbf{x}$$에 대해서도 모델이 달라지면 충분통계량이 달라질 수 있다는 것을 의미한다.

    결론적으로 FSP의 $$T(\mathbf{x}) = T(\mathbf{y}) \implies \mathrm{Ev}(E, \mathbf{x}) = \mathrm{Ev}(E, \mathbf{y})$$는 모델이 옳다는 전제 하에서만 성립한다.

    예를 들어 $$X_1, \ldots, X_7 \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$$ ($$\sigma^2$$ known) model에서 sample $$\mathbf{x}=(6, 8, 10, 10, 10, 12, 14), \mathbf{y}=(7, 8, 9, 10, 11, 12, 13)$$가 관측되었다고 하자.

    Sample mean $$T(\mathbf{X})= \bar{X}$$는 parameter $$\mu$$에 대한 충분통계량이다.
    
    $$T(\mathbf{x}) = T(\mathbf{y})=10$$이므로 FSP에 의해 $$\mathrm{Ev}(E, \mathbf{x}) = \mathrm{Ev}(E, \mathbf{y})$$가 성립한다.

    즉, 두 sample은 충분통계량 값이 같으므로 parameter $$\mu$$에 대한 inference를 위한 동일한 정보를 가지고 있어야 한다.

    하지만 $$\mathbf{y}$$는 거의 uniform하게 퍼져있다는 점에서 관측한 통계학자가 가우시안 분포 모델 가정에 의심을 품을 수 있다.

    이 결과는 충분통계량이 아닌 sample의 다른 측면 (분포의 모양 등)을 통해 모델 가정을 바꾸는 등 inference에 영향을 미칠 수 있다는 것을 의미한다.

    즉 통계학자들은 FSP가 직관적이고 당연하다고 생각하지만 충분통계량이 아닌 정보를 이용해 "model checking"을 하며 FSP를 위반하고 있는 셈이다.

2. Birnbaum's theorem의 증명에 문제가 있다는 주장

    Birnbaum's theorem의 FSP & CP $$\implies$$ FLP의 증명의 순서를 다시 살펴보자.

    1. Mixed experiment $$E^*$$을 도입하고 $$E^*$$에 대한 충분통계량 $$T(J, \mathbf{X}_J)$$를 정의한다.
    2. $$T(1, \mathbf{x}^*_1) = T(2, \mathbf{x}^*_2)$$이므로 FSP에 의해 $$\mathrm{Ev}(E^*, (1, \mathbf{x}^*_1)) = \mathrm{Ev}(E^*, (2, \mathbf{x}^*_2))$$가 성립한다.
    3. CP에 의해 $$\mathrm{Ev}(E^*, (j, \mathbf{x}^*_j)) = \mathrm{Ev}(E_j, \mathbf{x}^*_j)$$가 성립한다.
    4. 따라서 $$\mathrm{Ev}(E_1, \mathbf{x}^*_1) = \mathrm{Ev}(E_2, \mathbf{x}^*_2)$$가 성립한다.

    위 증명은 FSP를 적용하고 CP를 적용한다. 하지만 Kalbfleisch (1975)는 아래처럼 CP를 적용하고 FSP를 적용해야 한다고 주장한다.

    1. $$E^*$$를 도입하고 수행하여 $$J=j$$를 관측한다.
    2. CP에 의해 $$\mathrm{Ev}(E^*, (j, \mathbf{x}^*_j)) = \mathrm{Ev}(E_j, \mathbf{x}^*_j)$$가 성립한다. 이제부터 $$E^*$$는 잊고 $$E_j$$에 대해서만 생각하자.
    3. $$E_j$$에 대한 충분통계량을 정의하여 이후 분석을 진행한다.

    만약 $$J=1$$이 관측된다면 $$E_1$$에 대한 충분통계량 $$T_1$$을 정의하여 분석을 진행하고, $$J=2$$가 관측된 경우에는 $$E_2$$에 대한 충분통계량 $$T_2$$를 정의하여 분석을 진행한다.

    이 두 충분통계량은 서로 다른 sample space위에서 정의된 충분통계량이므로 이전과 같이 FSP를 적용해 $$T_1(\mathbf{x}^*_1)$$과 $$T_2(\mathbf{x}^*_2)$$의 값을 비교하는 것이 의미가 없다.

    Birnbaum's theorem의 증명에서 $$E^*$$에 대한 충분통계량 $$T(J, \mathbf{X}_J)$$는 사실상 $$E_1$$과 $$E_2$$의 sample space를 $$\mathbf{X}_J$$에 퉁쳐서 domain으로 갖는다. 

    물론 해당 증명이 수학적인 모순은 없다. 하지만 수행되지 않을 가능성이 있는 experiment의 sample space를 domain으로 갖는 충분통계량을 도입하는 것은 수행된 experiment만이 inference에 영향을 미친다고 주장하는 CP의 정신과는 맞지 않는다.

이러한 비판에도 불구하고 FLP는 수학적으로 매력적이고, 효율적인 data reduction을 가능하게 하는 원리이므로 많은 통계학자들이 FLP를 받아들이고 있다. 

특히 베이지안 inference를 수행할 때 posterior $$\propto$$ likelihood $$\cdot$$ prior이므로 사전 믿음(prior)의 사후믿음(posterior)으로의 변화는 likelihood에만 의존한다. 

이는 베이지안 inference가 항상 FLP를 만족한다는 것을 의미한다.

## Reference
- Casella, G., & Berger, R. L. (2002). Statistical inference (2nd ed.). Duxbury.

