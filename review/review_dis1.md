# Discussion 1 Review

## Machine Learning Overview
### Formulating Learning Problems
1. Supervised Learning
  : 라벨값이 같이 주어지고, $f_/theta(X)$는 label $y$에 수렴하도록 학습된다.

2. Unsupervised Learning
  : 데이터셋은 unlabeled. 모델은 데이터셋 $D$의 잠재된 분포 특성을 학습하게 됨.

3. Reinforcement Learning
  : 정해져있는 데이터셋이 있는 것이 아니라, 데이터를 얻기 위해 계속 interaction이 일어나는 형태.
  rewards 함수를 정의해놓고 이를 최대화하는 문제로 변환. (다만 시계열적으로 데이터가 수집이 되는 듯하다)

### Solving Machine Learning Problems
1) model class 결정 (어떤 model을 사용할 것인가)  
2) loss function 결정 (어떤 식으로 모델 performance의 나쁨을 결정할 것인가)  
3) optimizer 결정 (어떤 식으로 파라미터 $\theta$를 결정할 것인가)  

### Dataset Splits During Training
- 하이퍼 파라미터 튜닝 시, validation set을 따로 둬서 퍼포먼스를 측정해야 함.  
- 모델 학습은 only train set에 대해서만, validation과 test는 단순 performance measure 용  
- Test set은 model이 최종적으로 결정된 후에 사용해야 함.  
- 새로운 테스트셋을 사용한다면 꼭 새로운 모델을 학습시킨 후에!


## Statistics Review
### Estimators

- Bias and Variance of Estimator
1) Bias: estimator의 기댓값이 실제 distribution과 얼마나 떨어져있는가를 측정하는 값. $E(f_\theta(x) - y)$ 로 정의함.  
2) Variance: estimator의 값이 estimator의 기댓값과 평균적으로 얼마나 차이가 나는지를 측정하는 값. $Var(f_\theta(x)) = E[(f_\theta(x) - E[f_\theta(x)])^2]$ 로 formulate된다.

- unbiased estimator (불편추정량)
 불편 추정량이란 estimator의 기댓값이 estimand y와 일치하는 것을 의미함. 즉 위에서 언급한 bias가 0인 것을 의미.

 - Bias-Variance Tradeoff
 가장 좋은 esitmator는 bias와 variance가 동시에 작은 값을 갖는 estimator라고 할 수 있다. 다만 이 둘 사이에는 tradeoff가 존재하므로 bias와 variance를 동시에 매우 작은 값을 갖도록 하는 추정량은 존재하기 힘들다. (어느 정도 타협이 필요함)


## Function Approximation and Risk functions
 딥러닝은 이러니 저러니 말해도 function approximation. Classification의 경우에는 y가 Discrete Variable, Regression의 경우네느 y가 continuous variable일 때 $P(y|x)$를 학습하는 것이다. y의 분포를 학습한다는 것은 결국 true $y$에 대해 데이터 $X$를 사용하여 추정량 $f_\theta(X)$ 를 구한다는 의미이다.  

  신경망에서의 파라미터는 weight과 bias. 이때 이러한 파라미터는 true distribution $y$와 추정 distribution인 $f_\theta(x)$의 거리를 최소화하는 방향으로 결정되어야 함.

  이때 "거리"를 정의하는 것이 바로 Risk Function을 결정하는 것.

### Loss Functions & Risk functions
 Loss란 estimator가 얼마나 별로인지를 측정하는 것. 이러한 Loss는 실제값과 estimate 간의 거리를 정의하는 데에 사용된다. Risk는 이러한 Loss의 expectation을 의미하는데, 파라미터에 대한 함수로 정의된다. formula를 확인해보면 다음과 같다.

 $ R(\theta; f(.)) = E_(x,y)[L(x,y,\theta)]

그러나 이를 바로 최적화할 수는 없는데 이는 바로 y의 true distribution을 알 수 없기 때문이다. 따라서 y의 true distribution을 데이터로부터 추정하여 empirical distribution으로 이를 대체하게 된다. 이를 사용한 risk function을 empirical risk minimization이라고 한다.


- Empirical Risk
 Empirical Risk는 true distribution으로부터 추출된 sample을 바탕으로 결정된 risk 함수이다. 개별 라벨값에 대한 Loss의 산술평균으로 정의된다.
 다만 이것이 true risk minimization과 동일한가의 문제는 further ado이다.
