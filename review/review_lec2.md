# Lecture 2.

There are different types of learning problems such as supervised learning, unsupervised learning and reinforcement learning and so on.

### Supervised learning
In supervised learning, It matters that how we represent $f_\theta(x)$, how we measure difference between $f_theta(x_i)$ and $y_i$ and how we optimize the process of finding the best $\theta$.


### Unsupervised learning
- generative modeling: GANs, VAEs, pixel RNN, etc.
- self-supervised representation learning  


### Reinforcement Learning
generalized supervised learning. (지도학습은 Y를 매칭하는 f를 찾는거고, 강화학습은 그냥 reward를 최대화하는 건데 정답에 가까워지는 걸 reward로 정의할 수 있으니까)

![img](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

많은 분야로 확장 가능. 로봇이나, ad placement나, recommendation system 같은 것들.



<br>
<br>
## Supervised Learning

라벨 값이 존재하는 데이터, 데이터를 기반으로 라벨을 추론함.  
But, 라벨값 그 자체를 추론하는 것보다는 해당 라벨일 **확률**을 추론하는 것이 조금 더 합리적임.  

Why? ***smoothness*** 때문에 조금 더 학습하기가 ***쉬워지기 때문***  
(smootheness → 라벨값을 조금씩 변경하는 것은 불가능하지만, 확률값은 조금씩 조정하는 것이 가능함)

그럼 우리는 무슨 확률값을 예측하는 걸까? $p(y|x)$ => 데이터 x가 주어졌을 때의 Y가 y일 확률을 예측하는 것임. 이 조건부 확률은 $p(x,y) over p(x)$ 로 정의됨.  

그럼 우리는 어떤 식으로 $p(y|x)$를 표현해낼까?
각 데이터가 어떤 클래스에 속할 확률을 softmax를 취한 값.  

softmax란 exp(f_1(x)) / sum(exp~~) 임. (각 함수값이 모두 positive이고, 합으로 나눠줌으로써 정규화)


Till now, supervised learning에서의 task가 확률값을 예측하는 것이고 그 확률값은 각 클래스에 대한 softmax함수로 정의되는 것을 알았다.
하지만 x에서 softmax의 인자로 들어가는 $f$를 어떻게 정할까? f를 정하는 것 자체가 중요하다기보다는, f를 구성하는 파라미터 $\theta$를 정하는 것이 훨씬 중요함.

 뉴럴넷에서는 별다를 거 없고 $\theta$와 X의 선형결합으로 f를 정의하는데, 이때 $\theta$가 업데이트되면서 학습이 되는 것. 따라서 **어떻게 $\theta$를 업데이트시킬 것인가** 가 중요한 것이라고 볼 수 있음.  
→ 이는 결국 **loss function의 정의와 optimizer의 선택**으로 이어지는 것임.


<br>
<br>
## Loss functions

데이터셋 D에서 각 인스턴스는 iid라고 가정함 (independent and identically distributed). 즉, 각 인스턴스는 독립적이지만 각각 같은 확률분포를 갖는다.

이러한 iid 가정 하에서 우리는 각 인스턴스의 확률을 곱하면 전체 데이터셋의 확률이라고 말할 수 있음.


$p(x_i, y_i) = p(x_i)*p(y_i|x_i)$

우리는 $p_\theta(y|x)$를 학습하는데, 이는 p(y|x)를 모형화한 것. p(y|x)가 그럴 듯하게 되기 위해 우리는 좋은 \theta를 추정해야함.

이때 추정하는 방식이 mle (maximum likelihood estimation)인 것. 하지만 이를 loss function에 맞게 해석한다면, log-likelihood에 negative만 붙여주면 됨. (그래야 최소화 문제로 바뀌니까)

즉, NLL (negative log likelihood) 가 우리의 loss function이 되는 거고, 이건 cross-entropy와 동일하게 됨.

결론: 우리가 흔히 사용하는 loss function은 NLL, zero-one loss, MSE, cross entropy 등이 있다. 이때 NLL = croos entropy (이때는 label의 분포가 discrete하다고 가정), MSE = NLL (regression의 경우) 이다.



<br>
<br>
## Optimization

지금까지 무엇을 기준으로 $\theta$ 가 학습이 되는지에 대해서 알아보았고(loss func),  
앞으로는 그래서 **어떤 방식으로** $\theta$를 업데이트할 것인가에 대해 알아볼 것이다.
→ Optimizer 선택  


<br>
### Gradient Descent

Loss Function이 어느 theta에서 가장 작은값을 갖는가의 문제로 귀결됨. 즉, 최적화 문제 중 최소화 문제임.

즉 미분값이 0이 되는 값을 찾으면 되는데, loss function이 복잡해지고 파라미터 스페이스가 커질 수록 함수의 최적화를 완벽하게 찾아내는 것은 불가능함. 따라서 반복적으로 계산을 수행해가며 값을 최소화하는 $\theta$값을 찾아야함.

이러한 방식 중 하나가 **Gradient Descent**이며, Gradient는 편미분 벡터이며, Descent란 Loss Function 평면 위에서 계속 내려가면서 최저점을 찾는다는 것 정도로 이해할 수 있음.

수식적으로 표현하면,

1. Compute $\nabla_\thetaL(\theta)$
2. $\theta ← \theta - \alpha\nabla_\thetaL(\theta)$


<br>
#### Logistic Regression
logreg의 경우, 조건부확률 $p(y|x)$는 softmax로 정의되며, loss function은 cross entropy 혹은 nll로 정의된다. 하지만 클래스가 2개밖에 없는 binary classification의 경우에는 조건부확률을 softmax로 정의하는 것은 redundancy가 발생하므로, softmax가 아닌 sigmoid함수를 사용한다.
