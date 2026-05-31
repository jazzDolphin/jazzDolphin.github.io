---
title: "Principle of Data Reduction 5: Equivariance Principle"
date: 2026-05-24 23:55:00 +0900
categories: [Statistical Inference, Foundations]
order: 8
math: true
---

이번 포스트에서는 Equivariance Principle에 대해 알아보도록 하자. 

[Sufficiency Principle](https://jazzdolphin.github.io/posts/data-reduction2/)은 $$\theta$$에 대한 충분통계량 값이 같은 sample points들 간에는 $$\theta$$에 대한 inference 결과에 차이가 없다고 주장한다.

이는 충분통계량 값이 같은 sample points 중 어떤 것이 관측되었는지에 관한 정보를 고려하지 않고 inference를 수행하여 data reduction을 하는 것이다.

[Likelihood Principle](https://jazzdolphin.github.io/posts/data-reduction4/)은 관측된 두 sample points $$\mathbf{x}, \mathbf{y}$$의 likelihood function이 $$\theta$$에 무관한 상수에 대해 서로 비례한다면 ($$L(\theta \mid \mathbf{x})=C(\mathbf{x},\mathbf{y}) \ L(\theta \mid \mathbf{y})$$) 동일한 inference를 수행해야 한다고 주장한다.

이는 likelihood function이 비례한다는 정보만으로 inference를 수행하여 data reduction을 하는 것이다.

Equivariance Principle은 위 두 Principle과 다른 방식으로 data reduction을 수행한다.

## Equivariance Principle

Equivariance Principle은 어떤 함수 $$T$$에 대해 $$T(\mathbf{x})=T(\mathbf{y})$$라면 $$\mathbf{x}$$가 관측되었을 때의 inference와 $$\mathbf{y}$$가 관측되었을 때의 inference가 "어떤 관계"가 있어야 한다고 주장한다. (위 두 principle과 다르게 꼭 inference의 결과가 같아야 하는 것은 아니다.)

이러한 inference의 관계는 고려해야할 추정량의 집합을 축소해 data reduction을 하는 것이다.

Equivariance Principle은 아래의 두 원칙을 하나로 묶은 것이다.

1. Measurement Equivariance:

    inference는 측정 단위를 바꾼다고 바뀌면 안 된다.

    예를 들어 나무의 지름의 평균을 추정하는 상황에서 inch를 사용하든, meter를 사용하든 같은 추정치가 나와야 한다.

2. Formal Invariance:

    두 inference problem이 같은 수학적 모델을 사용하면 같은 inference procedure가 사용되어야 한다는 것이다.

    같아야 하는 모델의 요소는 1. parameter space $$\Theta$$, 2.family of pdf or pmf $$\{f(\mathbf{x} \mid \theta):\theta \in \Theta\}$$ 3. 모든 가능한 inference의 집합(이 글에서는 inference를 $$\Theta$$의 element를 정하는 점추정으로 가정해서 서술하겠다.) 
    
    formal invariance는 experiment의 물리적인 실체가 중요한 것이 아니라 수학적인 요소들만 중요하다는 것이다.
    예를 들어 계란 한 판의 평균가격을 추정하는 문제와 기린의 평균키를 추정하는 문제가 $$\Theta=\{\theta:\theta>0\}$$이므로 같은 inference procedure를 사용해야 한다는 것이다.

Equivariance와 invariance는 이름이 비슷하지만 서로 다른 개념이다.

equivariance는 data가 어떤 변환을 거치면 estimate가 이에 대응하는 방식으로 변환된다는 의미이고, invariance는 data가 어떤 transformation을 거쳐도 estimate가 변하지 않아야 한다는 의미이다.

> **Equivariance Principle**: 
>
> 만약 $$\mathbf{Y}=g(\mathbf{X})$$가 측정 단위의 변화이고 $$\mathbf{Y}$$의 model이 $$\mathbf{X}$$의 모델과 같은 formal structure를 가지고 있다면 inference procedure이 measurement equivariant하고 formally invariant 해야한다.
{: .prompt-math }

### Example: Equivariance Principle in Binomial Model

$$X \sim \mathrm{binom}(n, p)$$ where $$n$$ known이라고 하자.

그리고 $$T(x)$$가 $$X=x$$가 관측되었을 때의 $$p$$의 추정이라고 하자.

$$p$$를 inference 하기 위해 성공 횟수 대신에 실패 횟수 $$Y=n-X$$를 사용한다고 하자. 

그렇다면 $$Y \sim \mathrm{binom}(n, q=1-p)$$ 이다. 

여기서 측정 단위의 변경 $$g(X)=n-X$$가 도입된다.

$$T^*(y)$$가 $$Y=y$$로 관측되었을 때 $$q$$의 추정이라고 하자.

그렇다면 $$X=x$$가 관측되면 $$Y=n-x$$가 관측된 것이고, $$q$$의 추정치는 $$T^*(n-x)$$인데 $$p=1-q$$이므로 $$p$$의 추정치는 $$1-T^*(n-x)$$가 되어야 한다.

같은 데이터에서 다른 측정 단위로 추정한 $$p$$의 추정치가 동일해야한다는 measurement equivariance에 의해 $$T(x)=1-T^*(n-x)$$가 성립한다.

그리고 $$p$$를 추정하는 문제와 $$q$$를 추정하는 문제 모두 $$\mathrm{binom}(n, \theta), \theta \in [0,1]$$이라는 동일한 formal structure를 가진다는 것이다.

따라서 formal invariance에 의해 $$T(z)=T^*(z)$$ for all $$z=0, \dots, n$$에 대해 성립한다.

이에 따라 $$T(x)=1-T^*(n-x)=1-T(n-x)$$가 성립한다.

만약 Equivariance Principle을 적용하지 않는다면 $$T$$의 domain인 $$\{0, \dots, n\}$$에서 $$T$$의 값을 지정해야 한다.

하지만 Equivariance Principle에 의해 $$T(x)=1-T(n-x)$$가 성립하므로 $$T(0), \dots, T(\lfloor n/2 \rfloor)$$의 값만 지정하면 나머지 값들은 자동으로 결정된다.

구체적으로 고려해야할 추정량의 집합이 어떻게 축소되는지 살펴보자.

먼저 $$T_1(x)=x/n$$이 Equivariance Principle을 만족하는지 확인해보자.

$$1-T_1(n-x)=1-\frac{n-x}{n}=\frac{x}{n}=T_1(x)$$

이므로 조건을 만족한다. 따라서 $$T_1$$은 equivariant하다.

다음으로 표본 비율을 $$0.5$$ 쪽으로 끌어당기는 추정량 $$T_2(x)=0.9(x/n)+0.1(0.5)$$를 보자. $$p$$가 $$0.5$$ 근처일 것이라는 사전 정보가 있을 때 합리적일 수 있는 추정량이다.

$$
\begin{align*}
    1-T_2(n-x)
    &=1-\left[0.9\cdot\frac{n-x}{n}+0.1(0.5)\right]\\
    &=1-0.9+0.9\cdot\frac{x}{n}-0.05\\
    &=0.9\cdot\frac{x}{n}+0.05\\
    &=0.9(x/n)+0.1(0.5)=T_2(x)
\end{align*}
$$

이므로 $$T_2$$ 역시 equivariant하다.

반면에 $$T_3(x)=0.8(x/n)+0.2(1)$$은 equivariant하지 않다. 

$$x=0$$에서 확인해보면

$$T_3(0)=0.2 \neq 0 = 1-T_3(n)$$

이므로 조건 $$T(x)=1-T(n-x)$$를 위반한다.

이처럼 Equivariance Principle을 적용하면 $$T_3$$과 같은 추정량은 고려할 필요가 없게 된다.

## Group of Transformations

Equivariance Principle을 적용할 때 어떤 transformation $$g$$를 선택하는지가 모든 equivariance argument의 핵심이다. 

그리고 이때 사용되는 transformation들은 아래에 정의할 group of transformations에 속하는 함수여야 한다

> **Group of Transformations**
>
> Sample space $$\mathcal{X}$$에서 $$\mathcal{X}$$로 가는 함수의 집합 $$\mathcal{G}$$가 다음 조건들을 만족할 때 $$\mathcal{G}$$를 group of transformations라고 한다.
>
> 1. (Inverse) 모든 $$g \in \mathcal{G}$$에 대해 $$g'(g(x))=x$$ for all $$x \in \mathcal{X}$$인 $$g'\in\mathcal{G}$$가 존재한다. (즉, $$g$$의 역함수 $$g'$$가 $$\mathcal{G}$$에 존재한다.)
>
> 2. (Composition) 모든 $$g, g' \in \mathcal{G}$$에 대해 $$g''(x)=g'(g(x))$$ for all $$x \in \mathcal{X}$$인 $$g''\in\mathcal{G}$$가 존재한다. (즉, $$\mathcal{G}$$는 함수의 합성에 대해 닫혀 있다.)
>
> 3. (Identity) Identity function $$e(x)=x$$ for all $$x \in \mathcal{X}$$가 $$\mathcal{G}$$에 존재한다.
{: .prompt-math }

Identity 조건은 inverse와 composition 조건을 만족하는 집합에는 자동으로 성립한다. 

Identity에 의해 $$g \in \mathcal{G}$$대해 $$g'(g(x)) = x$$ for all $$x$$인 $$g' \in \mathcal{G}$$가 존재한다. 이때 composition에 의해 $$g \circ g' \in \mathcal{G}$$ 인데, $$g \circ g' = e$$이므로 identity 조건이 만족한다.

### Example: Group of Transformations의 예시

$$\mathcal{G}=\{g_1, g_2\}$$ where $$g_1(x)=n-x$$ and $$g_2(x)=x$$라고 하자.

1. $$g_1$$의 역함수는 $$g_1$$ 자기 자신이고, $$g_2$$의 역함수는 $$g_2$$ 자기 자신이므로 $$\mathcal{G}$$에는 모든 원소의 역함수가 존재한다. 

2. $$\mathcal{G}$$는 함수의 합성에 대해 닫혀 있는지 확인해보자
    
    모든 $$x \in \mathcal{X}$$에 대해 다음이 성립한다.

   1. \$$g_1(g_1(x))=g_1(n-x)=n-(n-x)=x=g_2(x)$$
   2. \$$g_1(g_2(x))=g_1(x)$$
   3. \$$g_2(g_1(x))=g_1(x)$$
   4. \$$g_2(g_2(x))=g_2(x)$$
   
   따라서 $$\mathcal{G}$$는 함수의 합성에 대해 닫혀 있다.


그러므로 $$\mathcal{G}$$는 group of transformations이다.

Equivariance Principle을 사용하려면 transformation이 적용된 문제에 formal invariance을 적용해야 한다.

즉 $$\mathbf{Y}=g(\mathbf{X})$$와 같이 transformation이 적용된 문제의 모델이 $$\mathbf{X}$$의 모델과 같은 formal structure를 가져야 한다.

> **Invariant Model under a Group of Transformations**:
>
> $$\mathcal{F}=\{f(\mathbf{x} \mid \theta):\theta \in \Theta\}$$가 $$\mathbf{X}$$의 pdf or pmf의 집합이고, $$\mathcal{G}$$가 $$\mathbf{X}$$의 sample space의 group of transformations라고 하자.
>
> 모든 $$g \in \mathcal{G}$$와 모든 $$\theta \in \Theta$$에 대해 $$\mathbf{X}$$가 $$f(\mathbf{x}\mid \theta)$$의 분포를 갖고 $$\mathbf{Y}=g(\mathbf{X})$$가 $$f(\mathbf{y} \mid \theta')$$의 분포를 갖게하는 유일한 $$\theta' \in \Theta$$가 존재한다면 $$\mathcal{F}$$는 $$\mathcal{G}$$에 대해 invariant하다고 한다.
{: .prompt-math }

### Example: Binomial Model에서의 Invariant Model

위의 예시에서 $$\mathcal{G}=\{g_1, g_2\}$$ where $$g_1(x)=n-x$$ and $$g_2(x)=x$$이 group of transformations임을 보였다.

$$\mathcal{F}=\{f(x \mid p):p \in [0,1]\}$$ where $$f(x \mid p) = \binom{n}{x}p^x(1-p)^{n-x}$$이 $$\mathcal{G}$$에 대해 invariant한지 확인해보자.

$$g_1(X)=n-X \sim \mathrm{binom}(n, 1-p)$$이므로 $$\theta' = 1-p$$가 존재한다.

$$g_2(X)=X \sim \mathrm{binom}(n, p)$$이므로 $$\theta' = p$$가 존재한다.

따라서 $$\mathcal{F}$$는 $$\mathcal{G}$$에 대해 invariant하다.

### Example: Normal Model에서의 Invariant Model

$$X_1, \dots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$$라고 하자. 이때 $$\mu$$와 $$\sigma^2$$는 모두 unknown이다.

$$\mathcal{G}=\{g_a(\mathbf{x}): a \in \mathbb{R}\}$$ where $$g_a(\mathbf{x})=(x_1+a, \dots, x_n+a)$$이라고 정의하자.

일단 먼저 $$\mathcal{G}$$가 group of transformations인지 확인해보자.

1. $$g_a$$의 역함수는 $$g_{-a}$$인데 $$g_{-a} \in \mathcal{G}$$이므로 $$\mathcal{G}$$에는 모든 원소의 역함수가 존재한다.

2. $$g_{a_2}(g_{a_1}(\mathbf{x}))=g_{a_2}(x_1+a_1, \dots, x_n+a_1)=(x_1+a_1+a_2, \dots, x_n+a_1+a_2)=g_{a_1+a_2}(\mathbf{x})$$ 인데 $$g_{a_1+a_2} \in \mathcal{G}$$이므로 $$\mathcal{G}$$는 함수의 합성에 대해 닫혀 있다.

따라서 $$\mathcal{G}$$는 group of transformations이다.

$$\mathcal{F}=\{f(\mathbf{x} \mid \mu, \sigma^2):\mu \in \mathbb{R}, \sigma^2 > 0\}$$라고 하자. 이때 $$f$$는 $$\mathbf{X}$$의 joint pdf로 정의된다.

$$\mathbf{Y}=g_a(\mathbf{X})=(X_1+a, \dots, X_n+a)$$라고 하자.

그렇다면 $$\mathbf{Y}$$의 각 원소는 $$\mathcal{N}(\mu+a, \sigma^2)$$의 분포를 가지고, $$g_a$$는 상수를 더하기만 하기에 \mathbf{X}의 iid 성질의 보존되므로 $$\mathbf{Y}$$의 joint pdf는 $$f(\mathbf{y} \mid \mu+a, \sigma^2)$$로 표현할 수 있다.

따라서 $$\theta'=(\mu+a, \sigma^2)\in \Theta$$가 존재한다.

정리하면, Equivariance Principle을 적용하려면 먼저 transformation의 집합이 group of transformations인지 확인해야 하고, 그 group of transformations 안에서 모델의 family of pdf or pmf가 invariant한지 확인해야 한다.

이 두 조건이 만족된다면 measurement equivariance와 formal invariance가 적용할 수 있고 고려해야할 추정량의 집합이 축소되어 data reduction이 가능하다.

다만 Casella & Berger의 책에서는 measurement equivariance는 직관적으로 타당하기에 많은 사람들이 Equivariance Principle에 대해 생각할 때 measurement equivariance만 떠올린다고 한다.

반면에 formal invariance는 설명하고자 하는 물리적 현실이 다르더라도 수학적인 구조만 같으면 같은 inference procedure가 사용되어야 한다고 하므로 직관적으로 타당하지 않다고 받아들이는 사람들이 많다고 한다.

하지만 Sufficiency Principle과 Likelihood Principle이 data reduction을 하는 것처럼 Equivariance Principle도 허용 가능한 inference의 집합을 제한하며 data reduction을 해서 분석을 단순화할 수 있다.

## Reference
- Casella, G., & Berger, R. L. (2002). Statistical inference (2nd ed.). Duxbury.
