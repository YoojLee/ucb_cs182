# Lecture 1. Introduction


### What is Machine Learning?

input - output 간의 rule을 사용자가 직접 찾아 정의하기엔 패턴이 너무 복잡해지고, 예외가 너무 많아진다면? *Acquire the rule from ***data****

Data $X$ & parameter $\theta$, output $y$
→ $f(X|\theta) = y$

parameters to be trained so we call $f$ as parametrized program (or function).

Then , **How** are we going to **parametrize** function?
Not with just the binary classification (that is, not just defining decision boundaries that separate O from X)

### Shallow Learning
 fixed function for extracting *features* from $x$
ex. HOG (Histogram of the gradients)
→ could name it a "compromise" solution (hand-program the features)

In this case, learning process is not as complex as much (we could use 2D binary classification here) , but coming up with good featues is very difficult to reach a satisfaction.

### From Shallow to Deep
We parametrized also the program that extracts features from input not only the part that predicts label from features. That is, ***Feature are learned***, not Extracted!

 For Image Classification, passing thru multiple layers, program learns multiple level representations. In latter layers, program learns higher level representations.

 cf.) Higher level representations
 Those are
 - more abstract
 - more invariant to nuisances (that is not crucial to predict targets, such as background color and so on)
 - more contibutions to predict label (?)

### What is *Deep* Learning?

Machine learning with ***multiple layers of learned representations***
(not just mere machine learning)

The **function** that represents the transformation from input to internal representation to output is usually a deep neural netwrok

The parameters for every layer are usually trained with respect to overall task objective (e.g. accuracy)
→ end-to-end learning

###  The More Layers, The Better?
Resnet 152, the SOTA, has 152 layers in itself. (its error is even lower than that of human)
→more layers help. (But always?)

### The underlying themes
(themes: things that is important for whom to make their deep learning systems work well)

- Acquire representations by using **high-capacity** models and lots of **data**, without requiring manual engineering of features or representations
(Automation and better performance)
→ Model capacity: how many different functions a particular model can represent
(So, with high capacity model, we mean that the model is highly generalized with a decent performance)

- Learning vs. inductive bias
 : minimalize the designer insight that is put into the model. (cannot make it to zero, tho)
	 - Inductive bias: built-in knowledge or biases in a model designed to help it learned. (Why called bias? since it makes some solutions *more likely* or some *less likely*)

- Algorithms that scale
	Scaling: ability for an algorithm to work better as more data and more capacity is added. (with capacity, we mean representational capcity, and compute)
