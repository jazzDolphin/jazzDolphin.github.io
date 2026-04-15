---
title: Logistic Regression (로지스틱 회귀)
date: 2026-04-15 19:24:00 +0900
categories: [Machine Learning, Linear Models]
order: 2
math: true
---

## Logistic Regression (로지스틱 회귀)

Logistic Regression은 class label $$y \in \{1, \dots, C\}$$에 대해 input data $$\boldsymbol{x} \in \mathbb{R}^D$$와 parameter $$\boldsymbol{\theta}$$가 주어졌을 때 $$p(y \mid \boldsymbol{x}, \boldsymbol{\theta})$$를 모델링하여 이를 바탕으로 classification하는 것이다.

즉, 특정 데이터와 파라미터가 주어졌을 때, 해당 데이터가 특정 클래스에 속할 확률을 모델링하는 것이다.

$$C=2$$인 경우에는 binary logistic regression이라고 하고, $$C>2$$인 경우에는 multinomial logistic regression이라고 한다.

## Binary Logistic Regression
Binary logistic regression은 다음과 같은 모델로 정의된다.

$$
p(y \mid \boldsymbol{x},\boldsymbol{\theta})=\mathrm{Ber}(y \mid \sigma(\boldsymbol{w}^\top \boldsymbol{x}+b))
$$

여기서 파라미터 $$\boldsymbol{\theta}=(\boldsymbol{w},b)$$에서 $$\sigma$$는 sigmoid 함수, $$\boldsymbol{w}\in\mathbb{R}^D$$는 weights, $$b\in\mathbb{R}$$는 bias이다.

즉, 주어진 데이터 $$\boldsymbol{x}$$가 클래스 $$y=1$$에 속할 확률은 다음과 같이 계산된다.

$$
p(y=1 \mid \boldsymbol{x},\boldsymbol{\theta})=\sigma(a)=\frac{1}{1+e^{-a}}
$$

이때 $$a=\boldsymbol{w}^\top \boldsymbol{x}+b=\log\left(\frac{p}{1-p}\right)$$이며, 이를 log-odds라고 부른다. $$(p = p(y=1 \mid \boldsymbol{x},\boldsymbol{\theta}))$$. 

ML에서 $$a$$는 logit 또는 pre-activation라는 용어로도 불린다.

만약 각 클래스를 잘못 분류했을 때 발생하는 loss가 동일하다면, 최적의 결정 규칙은 클래스 1일 확률이 클래스 0일 확률보다 클 때 $$y=1$$로 예측하는 것이다.

따라서 예측 함수는 다음과 같이 정의된다.

$$
f(\boldsymbol{x}) = \mathbb{I}(p(y=1 \mid \boldsymbol{x})>p(y=0 \mid \boldsymbol{x}))
=\mathbb{I}\left(\log\frac{p(y=1 \mid \boldsymbol{x})}{p(y=0 \mid \boldsymbol{x})}>0\right)
=\mathbb{I}(\boldsymbol{w}^\top\boldsymbol{x}+b>0)
$$

이 수식에서 $$\boldsymbol{w}^\top \boldsymbol{x}=\langle \boldsymbol{w},\boldsymbol{x}\rangle$$는 가중치 벡터 $$\boldsymbol{w}$$와 특성 벡터 $$\boldsymbol{x}$$ 간의 내적(inner product)을 의미한다.

즉, normal vector가 $$\boldsymbol{w}$$이고 bias가 $$b$$인 affine hyperplane을 경계로 예측값이 변하게 된다.

이러한 affine hyperplane을 logistic regression의 decision boundary라 부른다.

하지만 실제 데이터는 항상 이러한 hyperplane으로 완벽하게 분류되지 않으므로 주어진 데이터가 각 클래스에 속할 확률을 모델링해야 한다. 

### Maximum Likelihood Estimation (MLE)

$$p(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}) = \sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)= \phi_n$$ 이라고 하면 ($$\boldsymbol{x}$$에서 $$x_0=1$$이라 가정해 bias $$b$$를 $$w_0$$에 포함시킨다.) 데이터셋 $$\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n=1}^N$$에 대한 negative log likelihood는 다음과 같이 계산된다.

$$
\begin{align*}
    \mathrm{NLL}(\boldsymbol{w}) 
    &= -\frac{1}{N} \log p(\mathcal{D} \mid  \boldsymbol{w}) = -\frac{1}{N} \log \prod_{n=1}^N \mathrm{Ber}(y_n \mid \phi_n) \\
    &= -\frac{1}{N} \sum_{n=1}^N [y_n \log \phi_n + (1-y_n) \log(1-\phi_n)] \\
    &= \frac{1}{N} \sum_{n=1}^N \mathbb{H}_{ce}(y_n, \phi_n)
\end{align*}
$$

NLL을 최소화하는 parameter $$\boldsymbol{w}$$는 다음과 같이 계산된다.

$$
\arg \min_\boldsymbol{w} \mathrm{NLL}(\boldsymbol{w}) = \arg \min_\boldsymbol{w} \frac{1}{N} \sum_{n=1}^N \mathbb{H}_{ce}(y_n, \phi_n)
$$

즉 binary logistic regression에서 MLE는 cross-entropy loss를 최소화하는 것과 같다.

따라서 $$y_n$$이 1인 데이터는 $$\phi_n$$이 1에 가까워지도록 $$\boldsymbol{w}^\top \boldsymbol{x}_n$$값이 커지게 $$\boldsymbol{w}$$이 조정되고, $$y_n$$이 0인 데이터는 $$\phi_n$$이 0에 가까워지도록 $$\boldsymbol{w}^\top \boldsymbol{x}_n$$값이 작아지도록 $$\boldsymbol{w}$$이 조정된다.

### Gradient of NLL in Binary Logistic Regression

Gradient descent와 같은 gradient based optimization algorithm을 사용하여 $$\mathrm{NLL}$$을 최소화하는 $$\boldsymbol{w}$$를 구할 수 있다. 

그러기 위해 $$\nabla_\boldsymbol{w} \mathrm{NLL}(\boldsymbol{w})$$을 계산해보자.

$$
\begin{align*}
    \nabla_\boldsymbol{w} \mathrm{NLL}(\boldsymbol{w}) 
    &= -\frac{1}{N} \sum_{n=1}^N \nabla_\boldsymbol{w} [y_n \log \phi_n + (1-y_n) \log(1-\phi_n)] \\
    &= -\frac{1}{N} \sum_{n=1}^N \nabla_\boldsymbol{w} [y_n \log \sigma(\boldsymbol{w}^\top \boldsymbol{x}_n) + (1-y_n) \log(1-\sigma(\boldsymbol{w}^\top \boldsymbol{x}_n))] \\
    &= -\frac{1}{N} \sum_{n=1}^N \left[
        y_n \frac{1}{\sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)} \sigma'(\boldsymbol{w}^\top \boldsymbol{x}_n) \boldsymbol{x}_n + (1-y_n) \frac{1}{1-\sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)} (-\sigma'(\boldsymbol{w}^\top \boldsymbol{x}_n)) \boldsymbol{x}_n
    \right] \\
    &= -\frac{1}{N} \sum_{n=1}^N \left[
        y_n (1-\sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)) - (1-y_n) \sigma(\boldsymbol{w}^\top \boldsymbol{x}_n) 
    \right]\boldsymbol{x}_n \quad (\because \sigma'(a) = \sigma(a)(1-\sigma(a))) \\
    &= -\frac{1}{N} \sum_{n=1}^N (y_n - \sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)) \boldsymbol{x}_n \\
    &= -\frac{1}{N} \sum_{n=1}^N (y_n - \phi_n) \boldsymbol{x}_n
\end{align*}
$$

이것을 matrix form으로 표현하면 다음과 같다. (matrix vector multiplication을 vector의 각 원소가 matrix의 각 행에 weighted sum하는 것으로 생각하면 쉬움)

$$
\nabla_\boldsymbol{w} \mathrm{NLL}(\boldsymbol{w}) = -\frac{1}{N} \mathbf{X}^\top (\boldsymbol{y} - \boldsymbol{\phi})
$$

여기서 $$\mathbf{X}$$는 n번째 행이 $$\boldsymbol{x}_n^\top$$인 design matrix이다. 

### Convexity Of NLL in Binary Logistic Regression

NLL이 convex function인지 확인하기 위해 NLL의 Hessian을 계산해보자.

$$
\begin{align*}
    \mathbf{H}(\boldsymbol{w}) &= \nabla_\boldsymbol{w} \nabla_\boldsymbol{w}^\top \mathrm{NLL}(\boldsymbol{w})  \\
    &= -\frac{1}{N} \sum_{n=1}^N \nabla_\boldsymbol{w} [(y_n - \sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)) \boldsymbol{x}_n^\top] \\
    &= -\frac{1}{N} \sum_{n=1}^N \left[
        -\sigma'(\boldsymbol{w}^\top \boldsymbol{x}_n) \boldsymbol{x}_n \boldsymbol{x}_n^\top
    \right] \\
    &= \frac{1}{N} \sum_{n=1}^N \sigma(\boldsymbol{w}^\top \boldsymbol{x}_n)(1-\sigma(\boldsymbol{w}^\top \boldsymbol{x}_n))\boldsymbol{x}_n\boldsymbol{x}_n^\top \\
    &= \frac{1}{N} \mathbf{X}^\top \mathbf{S} \mathbf{X} \quad (\text{where } \mathbf{S} = \mathrm{diag}(\phi_1(1-\phi_1), \dots, \phi_N(1-\phi_N)))
\end{align*}
$$

$$
    \boldsymbol{v}^\top \mathbf{H}(\boldsymbol{w}) \boldsymbol{v} = \frac{1}{N} \boldsymbol{v}^\top \mathbf{X}^\top \mathbf{S} \mathbf{X} \boldsymbol{v} = \frac{1}{N} (\boldsymbol{v}^\top \mathbf{X}^\top \mathbf{S}^{\frac{1}{2}}) (\mathbf{S}^{\frac{1}{2}} \mathbf{X} \boldsymbol{v}) = \frac{1}{N} \|\mathbf{S}^{\frac{1}{2}} \mathbf{X} \boldsymbol{v}\|_2^2 \geq 0
$$

따라서 $$\mathbf{H}(\boldsymbol{w})$$는 positive semi-definite이므로 $$\mathrm{NLL}(\boldsymbol{w})$$는 convex function이다.

그러므로 $$\mathrm{NLL}(\boldsymbol{w})$$의 local minimum은 global minimum이 되기에 gradient descent와 같은 optimization algorithm을 사용하여 global minimum을 찾을 수 있다.

## Multinomial Logistic Regression

Multinomial logistic regression은 다음과 같은 모델로 정의된다.

$$
p(y \mid \boldsymbol{x},\boldsymbol{\theta})=\mathrm{Cat}(y \mid \mathrm{softmax}(\mathbf{W}\boldsymbol{x}+\boldsymbol{b}))
$$

여기서 $$\boldsymbol{\theta}=(\mathbf{W},\boldsymbol{b})$$에서 $$\mathbf{W}\in\mathbb{R}^{C\times D}$$는 weights, $$\boldsymbol{b}\in\mathbb{R}^C$$는 bias이다.

$$\mathbf{W}$$의 $$c$$번째 행을 $$\mathbf{W}_c$$라고 하면, $$y$$가 어떤 클래스 $$c$$에 속할 확률은 다음과 같이 계산된다. ($$\boldsymbol{x}$$에서 $$x_0=1$$이라 가정해 bias $$\boldsymbol{b}$$를 $$\mathbf{W}$$의 첫번째 열로 추가한다.)

$$
p(y=c \mid \boldsymbol{x},\boldsymbol{\theta})=\frac{\exp(\mathbf{W}_c \boldsymbol{x})}{\sum_{k=1}^C \exp(\mathbf{W}_k \boldsymbol{x})}
$$

### Maximum Likelihood Estimation (MLE)

$$\boldsymbol{\phi}_n \in \mathbb{R}^C$$을 n번째 데이터가 각 클래스에 속할 확률을 나타내는 vector라고 하자 즉 $$\phi_{nc} = p(y_n=c \mid \boldsymbol{x}_n, \boldsymbol{\theta})$$ 이다.

그리고 $$\boldsymbol{y}_n \in \mathbb{R}^C$$을 $$y_n$$이 $$c$$일 때 $$c$$번째 원소만 1인 one-hot vector라고 하면 데이터셋 $$\mathcal{D} = \{(\boldsymbol{x}_n, y_n)\}_{n=1}^N$$에 대한 negative log likelihood는 다음과 같이 계산된다.

$$
\begin{align*}
    \mathrm{NLL}(\mathbf{W}) 
    &= -\frac{1}{N} \log p(\mathcal{D} \mid \mathbf{W})\\
    &= -\frac{1}{N} \log \prod_{n=1}^N \prod_{c=1}^C \phi_{nc}^{y_{nc}} \\
    &= -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{nc} \log \phi_{nc} \\
    &= -\frac{1}{N} \sum_{n=1}^N \boldsymbol{y}_n^\top \log \boldsymbol{\phi}_n \\
    &= \frac{1}{N} \sum_{n=1}^N \mathbb{H}_{ce}(\boldsymbol{y}_n, \boldsymbol{\phi}_n)
\end{align*}
$$

### Gradient of NLL in Multinomial Logistic Regression

Gradient based optimization algorithm을 사용하기 위해 $$\nabla_\mathbf{W} \mathrm{NLL}(\mathbf{W})$$을 계산해보자.

우선 $$\nabla_{\mathbf{W}_j} \phi_{nc}$$을 계산해보자. 

1. $$j=c$$인 경우

    $$
    \begin{align*}
      \nabla_{\mathbf{W}_j} \phi_{nc} 
      &= \nabla_{\mathbf{W}_j }\frac{
        \exp(\mathbf{W}_c \boldsymbol{x}_n)
      }{\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n)} \\
      &= \frac{
        \exp(\mathbf{W}_c \boldsymbol{x}_n) \boldsymbol{x}_n (\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n)) - \exp(\mathbf{W}_c \boldsymbol{x}_n) \exp(\mathbf{W}_c\boldsymbol{x}_n) \boldsymbol{x}_n
      }{(\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n))^2} \\
      &= \frac{
        \exp(\mathbf{W}_c \boldsymbol{x}_n) (\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n) - \exp(\mathbf{W}_c\boldsymbol{x}_n))  \boldsymbol{x}_n
      }{(\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n))^2} \\
      &= \phi_{nc} (1-\phi_{nc}) \boldsymbol{x}_n
    \end{align*}
    $$
  
2. $$j \neq c$$인 경우

    $$
    \begin{align*}
      \nabla_{\mathbf{W}_j} \phi_{nc} 
      &= \nabla_{\mathbf{W}_j }\frac{
        \exp(\mathbf{W}_c \boldsymbol{x}_n)
      }{\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n)} \\
      &= -\frac{
        \exp(\mathbf{W}_c \boldsymbol{x}_n) \exp(\mathbf{W}_j\boldsymbol{x}_n) \boldsymbol{x}_n
      }{(\sum_{k=1}^C \exp(\mathbf{W}_k\boldsymbol{x}_n))^2} \\
      &= -\phi_{nc} \phi_{nj} \boldsymbol{x}_n
    \end{align*}
    $$

따라서 Kronecker Delta $$\delta_{cj}$$를 사용하여 $$\nabla_{\mathbf{W}_j} \phi_{nc}$$을 다음과 같이 표현할 수 있다.

$$
\nabla_{\mathbf{W}_j} \phi_{nc} = \phi_{nc} (\delta_{cj} - \phi_{nj}) \boldsymbol{x}_n
$$ 

이를 matrix form으로 표현하면 다음과 같다.

$$
\nabla_{\mathbf{W}} \phi_{nc} = \phi_{nc} (\boldsymbol{e}_c - \boldsymbol{\phi}_n)\boldsymbol{x}_n^\top
$$

여기서 $$\boldsymbol{e}_c \in \mathbb{R}^C$$는 $$c$$번째 원소만 1인 one-hot vector이다. ($$\mathbb{R}^C$$의 $$c$$번째 standard basis vector라 생각하자)

따라서 $$\nabla_\mathbf{W} \mathrm{NLL}(\mathbf{W})$$는 다음과 같이 계산된다.

$$
\begin{align*}
    \nabla_\mathbf{W} \mathrm{NLL}(\mathbf{W}) 
    &= \nabla_\mathbf{W} \left(
        -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{nc} \log \phi_{nc}
    \right) \\
    &= -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{nc} \frac{1}{\phi_{nc}} \nabla_\mathbf{W} \phi_{nc} \\
    &= -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{nc} (\boldsymbol{e}_c - \boldsymbol{\phi}_n)\boldsymbol{x}_n^\top \\
    &= -\frac{1}{N} \sum_{n=1}^N  \left(
        \sum_{c=1}^C y_{nc} \boldsymbol{e}_c- \boldsymbol{\phi}_n \sum_{c=1}^C y_{nc}
    \right)\boldsymbol{x}_n^\top \\
    &= -\frac{1}{N} \sum_{n=1}^N (\boldsymbol{y}_n - \boldsymbol{\phi}_n)\boldsymbol{x}_n^\top \\
\end{align*}
$$

Numpy와 같은 라이브러리를 사용하여 구현할 때는 반복문보다는 matrix form으로 표현된 수식을 사용하는 것이 효율적이므로, 여기서 각 행렬을 다음과 같이 정의하면

* $$\mathbf{X} \in \mathbb{R}^{N\times D}$$: $$n$$번째 행이 $$\boldsymbol{x}_n^\top$$인 design matrix
* $$\mathbf{Y} \in \mathbb{R}^{N\times C}$$: $$n$$번째 행이 $$\boldsymbol{y}_n^\top$$인 matrix
* $$\boldsymbol{\Phi} \in \mathbb{R}^{N\times C}$$: $$n$$번째 행이 $$\boldsymbol{\phi}_n^\top$$인 matrix
  
$$\nabla_\mathbf{W} \mathrm{NLL}(\mathbf{W})$$를 아래와 같이 표현할 수 있다.

$$
\nabla_\mathbf{W} \mathrm{NLL}(\mathbf{W}) = -\frac{1}{N} (\mathbf{Y} - \boldsymbol{\Phi})^\top \mathbf{X}
$$

### Convexity Of NLL in Multinomial Logistic Regression

이제 gradient까지 계산했으니, $$\mathrm{NLL}(\mathbf{W})$$가 convex function인지 확인하기 위해 NLL의 Hessian을 계산해보자.

$$\mathrm{NLL}(\mathbf{W})$$의 Hessian은 4차원 tensor지만, 다루기 쉽게 block matrix로 Hessian matrix를 표현한다.

우선 $$\mathbf{W}$$를 다음과 같이 $$\tilde{\mathbf{W}} \in \mathbb{R}^{CD}$$인 block vector로 표현하자. ($$\mathbf{W}$$를 쫙 폈다고 생각하자)

$$
\tilde{\mathbf{W}}= \begin{bmatrix}
    \mathbf{W}_{11} \\
    \vdots \\
    \mathbf{W}_{1D} \\
    \mathbf{W}_{21} \\
    \vdots \\
    \mathbf{W}_{2D} \\
    \vdots \\
    \vdots \\
    \mathbf{W}_{C1} \\
    \vdots \\
    \mathbf{W}_{CD}
\end{bmatrix}
$$

그리고 $$\mathbf{H}(\mathbf{W}) = \nabla_\mathbf{W} \nabla_\mathbf{W}^\top \mathrm{NLL}(\mathbf{W})$$을 

다음과 같이 각 원소 $$\tilde{\mathbf{H}}_{ij}$$가 $$\nabla_{\mathbf{W}_i} \nabla_{\mathbf{W}_j}^\top \mathrm{NLL}(\mathbf{W}) \in \mathbb{R}^{D \times D}$$인 block matrix $$\tilde{\mathbf{H}}(\tilde{\mathbf{W}}) \in \mathbb{R}^{CD \times CD}$$로 표현하자.

$$
\tilde{\mathbf{H}}(\tilde{\mathbf{W}}) = \begin{bmatrix}
    \nabla_{\mathbf{W}_1} \nabla_{\mathbf{W}_1}^\top \mathrm{NLL}(\mathbf{W}) & \cdots & \nabla_{\mathbf{W}_1} \nabla_{\mathbf{W}_C}^\top \mathrm{NLL}(\mathbf{W}) \\
    \vdots & \ddots & \vdots \\
    \nabla_{\mathbf{W}_C} \nabla_{\mathbf{W}_1}^\top \mathrm{NLL}(\mathbf{W}) & \cdots & \nabla_{\mathbf{W}_C} \nabla_{\mathbf{W}_C}^\top \mathrm{NLL}(\mathbf{W})
\end{bmatrix}
$$

이렇게 표현하면 4차원이었던 Hessian이 2차원으로 표현되어 다루기가 편해진다.

이제 $$\tilde{\mathbf{H}}_{ij}=\nabla_{\mathbf{W}_i} \nabla_{\mathbf{W}_j}^\top \mathrm{NLL}(\mathbf{W})$$을 계산해보자.

$$
\begin{align*}
    \nabla_{\mathbf{W}_i} \nabla_{\mathbf{W}_j}^\top \mathrm{NLL}(\mathbf{W}) 
    &= \nabla_{\mathbf{W}_i} \left(
        -\frac{1}{N} \sum_{n=1}^N (y_{nj} - \phi_{nj}) \boldsymbol{x}_n^\top
    \right) \\
    &= \frac{1}{N} \sum_{n=1}^N \nabla_{\mathbf{W}_i} \phi_{nj} \boldsymbol{x}_n^\top \\
    &= \frac{1}{N} \sum_{n=1}^N \phi_{nj} (\delta_{ij} - \phi_{ni}) \boldsymbol{x}_n \boldsymbol{x}_n^\top
\end{align*}
$$

이제 Hessian이 PSD인지 확인해보자.

원래는 Hessian이 4차원 tensor이기에 $$\mathbf{H}(\mathbf{W})$$가 PSD인지는 $$\boldsymbol{v}^\top \mathbf{H} \boldsymbol{v} \geq 0$$이 모든 $$\boldsymbol{v} \in \mathbb{R}^{C\times D}$$에 대해 성립하는지 확인해야 한다.

하지만 $$\mathbf{H}$$를 block matrix로 표현했으므로 $$\boldsymbol{v}$$을 $$\tilde{\boldsymbol{v}} \in \mathbb{R}^{CD}$$인 block vector로 표현하자.

$$
\tilde{\boldsymbol{v}} = \begin{bmatrix}
    \boldsymbol{v}_1 \\
    \vdots \\
    \boldsymbol{v}_C
\end{bmatrix}
$$

이제 $$\tilde{\boldsymbol{v}}^\top \tilde{\mathbf{H}} \tilde{\boldsymbol{v}} \ge 0$$이 모든 $$\tilde{\boldsymbol{v}} \in \mathbb{R}^{CD}$$에 대해 성립하는지 확인해보자.

$$
\begin{align*}
    \tilde{\boldsymbol{v}}^\top \tilde{\mathbf{H}} \tilde{\boldsymbol{v}}
    &= \sum_{i=1}^C \sum_{j=1}^C \boldsymbol{v}_i^\top
    \left(
        \frac{1}{N} \sum_{n=1}^N \phi_{nj} (\delta_{ij} - \phi_{ni}) \boldsymbol{x}_n \boldsymbol{x}_n^\top
    \right)
    \boldsymbol{v}_j \\
    &= \frac{1}{N} \sum_{n=1}^N \left[
        \sum_{i=1}^C \sum_{j=1}^C 
        \phi_{nj} (\delta_{ij} - \phi_{ni}) (\boldsymbol{v}_i^\top \boldsymbol{x}_n) (\boldsymbol{v}_j^\top \boldsymbol{x}_n)
    \right] \\
    &= \frac{1}{N} \sum_{n=1}^N \left[
        \sum_{i=1}^C \sum_{j=1}^C \phi_{nj}\delta_{ij} (\boldsymbol{v}_i^\top \boldsymbol{x}_n) (\boldsymbol{v}_j^\top \boldsymbol{x}_n) 
        - \sum_{i=1}^C \sum_{j=1}^C \phi_{nj} \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n) (\boldsymbol{v}_j^\top \boldsymbol{x}_n)
    \right] \\
    &= \frac{1}{N} \sum_{n=1}^N \left[
        \sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)^2 
        - \sum_{i=1}^C \phi_{ni}(\boldsymbol{v}_i^\top \boldsymbol{x}_n) \sum_{j=1}^C \phi_{nj} (\boldsymbol{v}_j^\top \boldsymbol{x}_n)
    \right] \\
    &= \frac{1}{N} \sum_{n=1}^N \left[
        \sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)^2 
        - \left(\sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)\right)^2
    \right] \\
\end{align*}
$$

$$f(x)=x^2$$라 하고, 확률변수 $$K$$가 $$\phi_{ni}$$의 확률로 $$\boldsymbol{v}_i^\top \boldsymbol{x}_n$$의 값을 갖는다고 가정하자. (즉 $$K$$는 $$\mathrm{Cat}(\phi_{n1}, \dots, \phi_{nC})$$에서 샘플링됨)

그럼 이제 대괄호 안의 항들을 각각 아래와 같이 표현할 수 있다.

$$
\sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)^2 = \mathbb{E}[f(K)]
$$

$$
\left(\sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)\right)^2 = f(\mathbb{E}[K])
$$

$$f$$를 convex function으로 가정했으므로 Jensen's inequality에 의해 아래의 부등식이 성립한다.

$$
\mathbb{E}[f(K)] \geq f(\mathbb{E}[K])
$$

$$
\tilde{\boldsymbol{v}}^\top \tilde{\mathbf{H}} \tilde{\boldsymbol{v}}
=\frac{1}{N} \sum_{n=1}^N \underbrace{\left[
    \sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)^2 
    - \left(\sum_{i=1}^C \phi_{ni} (\boldsymbol{v}_i^\top \boldsymbol{x}_n)\right)^2
\right]}_{\geq 0 \text{ (by Jensen's inequality)}} \ge 0
$$

따라서 $$\tilde{\mathbf{H}}$$는 positive semi-definite이므로 $$\mathrm{NLL}(\mathbf{W})$$는 convex function이다.

## Reference
* Murphy, K. P. (2022). [Probabilistic Machine Learning: An Introduction.](https://probml.github.io/pml-book/book1.html) MIT press.
