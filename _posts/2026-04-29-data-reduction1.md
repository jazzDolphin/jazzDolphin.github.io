---
title: "Principle of Data Reduction 1: 통계량과 Data Reduction"
date: 2026-04-29 17:30:00 +0900
categories: [Statistics]
order: 4
math: true
---

CS229 강의나 머피 책에서 지수족과 GLM을 설명할 때 sufficient statistic에 대한 설명이 부족하다고 느껴서 이를 Statistical Inference(Casella and Berger) 책을 참고하여 공부하였다.

## Statistic and Data Reduction

통계학에서는 샘플을 바탕으로 모집단의 parameter를 inference한다.

Sample $$\mathbf{x} = x_1, \dots, x_n$$의 정보를 이용하여 parameter $\theta$를 inference하는 상황을 가정해 보자. 

(여기서 $$\mathbf{X} = [X_1, \dots, X_n]$$는 random variables, $$\mathbf{x} = [x_1, \dots x_n]$$는 관측된 샘플을 의미한다.)

샘플의 크기 $$n$$이 매우 커진다면 아래와 같은 문제가 생긴다.
1. 개별 샘플의 나열을 해석하기 매우 어렵다.
2. 샘플들을 그대로 저장하고 연산하는 것은 시간 및 공간 복잡도를 크게 증가시킨다.

따라서 많은 샘플들의 정보를 요약하기 위해 통계량(Statistic)의 개념이 도입되었다.

### 통계량 (Statistic)

통계량은 단순히 샘플을 input으로 받는 함수이다. 예를 들어, 표본 평균, 샘플의 최댓값, 샘플의 합 등이 모두 통계량에 해당한다.

단, 통계량을 계산하는 수식 자체에는 inference하고자 하는 파라미터 $$\theta$$가 포함되어서는 안 된다. 즉, $$T(\boldsymbol{X}) = T(X_1, X_2, \dots, X_n)$$와 같이 데이터로만 구성된 함수여야 한다.

또한 통계량은 sample space $$\mathcal{X}$$를 분할하는 것으로 해석할 수 있다.

$$\mathcal{T} = \{t: t=T(\mathbf{x}) \text{ for some } \mathbf{x} \in \mathcal{X} \}$$ 라고 정의하자. 즉, $$\mathcal{T}$$는 통계량 $$T$$에 대한 $$\mathcal{X}$$의 image이다.

그렇다면 $$T$$는 sample space $$\mathcal{X}$$를 다음과 같은 부분 집합들로 분할하는 역할을 한다.

$$
A_t = \{\mathbf{x}: T(\mathbf{x}) = t \in \mathcal{T} \}
$$

예를 들어 주사위를 3번 던지는 상황을 생각해보자.

통계량 $$T(\mathbf{x})=x_1+x_2+x_3$$ 을 정의한다면, $$A_9$$는 관측값들의 합이 $$9$$가 되는 샘플들의 집합이다.

그렇다면 $$A_9$$에는 $$[1, 3, 5]$$, $$[2, 3, 4]$$, $$[3, 3, 3]$$ 등은 포함되지만, $$[3, 4, 5]$$는 포함되지 않는다.

![Partition of sample space using sum statistic](/assets/img/posts/data-reduction1/partitioning_sample_space.png){: w="430px"}
_Figure1: 통계량 $$T(\mathbf{x})=x_1+x_2+x_3$$이 sample space $$\mathcal{X}$$를 분할하는 모습_ 

결과적으로 $$T(\mathbf{x})=9$$이라는 사실은 $$A_9$$에 속하는 수많은 원소 중 각 관측치가 구체적으로 어떤 값들을 가졌는지에 대한 raw data의 정보는 버리고, 오직 관측값들의 합이 9라는 요약된 정보만을 남긴다는 의미이다.

이것이 바로 통계량이 data reduction을 수행한다는 의미이다.

### Principle of Data Reduction

통계량을 이용해 data reduction을 수행할 때 임의의 통계량을  사용하게 된다면 parameter $$\theta$$를 inference하는 데 필요한 정보까지 버려질 수 있다.

따라서 통계량을 이용한 data reduction을 수행할 때 $$\theta$$를 inference하는 데 필요한 정보는 보존하면서도 불필요한 정보는 버리는 통계량을 선택하는 기준이 필요하다.

이를 위해 Casella 책에서는 다음과 같은 세 가지 원칙을 제시한다.

1. Sufficiency Principle
2. Likelihood Principle
3. Equivariance Principle

[다음 포스트](https://jazzdolphin.github.io/posts/data-reduction2/)에서는 Sufficiency Principle에 대해 알아보도록 하자.

## Reference
- Casella, G., & Berger, R. L. (2002). Statistical inference (2nd ed.). Duxbury.
