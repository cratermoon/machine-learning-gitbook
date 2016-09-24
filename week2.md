# Week 2: Linear Regression with Multiple Variables

## Multivariate Linear Regression
Multiple features (or variables)

### Notation

* $$n$$ = number of features
* $$x^{\left(i\right)}$$ input (features) of the $i$th training example
* $$x_j^{\left(i\right)}$$: value of feature $$j$$ in the $$i$$th training example
* $$y$$ price

### Example

| size (feet<sup>2</sup>) | # of bedrooms | # of floors | age of home (years) | price ($1000)|
| :------------- | :------------- | :------------- | :------------- | :------------- |
| $$x_1$$ | $$x_2$$ | $$x_3$$ | $$x_4$$ | $$y$$ |
| 2104 | 5 | 1 | 45 | 460 |
| 1416 | 3 | 2 | 40 | 232 |
| 1534 | 3 | 2 | 30 | 315 |
| 852 | 2 | 1 | 36 | 178 |
| ... | ... | ... | ... | ... |

$$x^{\left(2\right)} = [ 1416; 3; 2; 40]$$

$$x_j^{\left(2\right)} = 3$$

hypothesis becomes sum of n features
For convenience we set the zeroth feature = 1 for all examples.
$$X = [x_0; x_1; x_2; ... ; x_n]$$ size is n+1

$$\theta = [\theta_0; \theta_1; \theta_2; ... ; \theta_n]$$ size is also n+1

We can also write the hypothesis as $$\theta'x$$ (transpose)

### Feature Scaling
Make sure features on similar scales (take on a similar range of values) so gradient descent will converge more quickly.

Example: size range 0-2000 ft sq and # of bedrooms 1-5 results in a 2000:5 ratio of features, a tall and skinny contour plot that gradient descent can take a long time traversing.

Solution: scale features to a common range such as $$0 \leq x \leq 1$$ by dividing  by max. An ideal is $$-1 \leq x \leq 1$$, but a rule of thumb is  range limits of 1/3 to 3 are acceptable.

Also scale very small values up -- e.g a range of $$-0.0001 \leq x \leq 0.0001$$ should be scaled up to a normalized range.

### Mean Normalization
Normalizing the mean to ensure all features have a mean around 0. We can use $$max(x) - min(x)$$ or standard deviation of X.

Example $$x_1 \leftarrow \dfrac{(x_1- \mu_1)}{\sigma_1}$$

## Learning Rate: Gradient Descent in Practice

### Debugging Learning Rate
Plot $$J\left(\theta\right)$$ as a function of the number of iterations to get the value of the cost function after each iteration. The value should decrease after every iteration at a rate that is not too slow.  If the plot shows $$J\left(\theta\right)$$ increasing then the gradient descent is not working, so use a smaller $$\alpha$$. If the it is decreasing slowly, use a larger $$\alpha$$. Tip: use values that differ by about a factor of 3: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3....

## Features and Polynomial Regression

Housing price prediction features:
$$h_\theta(x) = \theta_0 + \theta_1 * frontage + \theta_2*depth$$

New feature Area X = frontage * depth then quadratic is a better fit than a straight line.

Polynomial regression
$$\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3$$ where $$x_1 = size, x_2 = size^2, x_3 = size^3$$  makes it *very* important to use feature scaling
as $$x_1 = 1-1000; x^2=1-1,000,000, \text{and} x^3 = 1-10^9$$

## Normal Equation
For some linear regression problems the normal euqatoin will giv us a much better ay to solve for the optimal value of $$\theta$$.  While Gradient Descent takes many iterations, the normal equation solves it in one step.

Intuitively, for a 1-dimensional $$\theta \in \mathbb R$$, or scalar:
$$J(\theta) = a\theta^2 +b\theta +c$$ then we set the derivative $$\frac{d}{d\theta}J(\theta)$$ to 0 and solve for $$\theta$$. When $$\theta \in \mathbb R^n+1$$ we must use the partial derivative.

$$\frac{\partial}{\partial \theta_j} J(\theta) = ...$$ for every $$j$$

$$
J(\theta_0,\theta_1,...,\theta_m) = \frac{1}{2m}\sum_{k=1}^m(h_\theta(x^{\left(i\right)} - y^{\left(i\right)})^2
$$ The derivation is involved.

### Example
| | size ft$$^2$$  | # of bedrooms | # of floors | age (years) | price ($1000)|
| :------------- | :------------- | :------------- | :------------- | :------------- | :------------- |
| $$x_0$$ | $$x_1$$ | $$x_2$$ | $$x_3$$ | $$x_4$$ | $$y$$ |
| 1       | 2104 | 5 | 1 | 45 | 450 |
| 1       | 1416 | 3 | 2 | 40 | 232 |
| 1       | 1534 | 3 | 2 | 30 | 315 |
| 1       | 852  | 2 | 1 | 45 | 178 |

$$
X  = \left[\begin{matrix}
1 & 2104 & 5 & 1 & 45 \cr
1 & 1416 & 3 & 2 & 40 \cr
1 & 1534 & 3 & 2 & 30 \cr
1 & 852 & 2 & 1 & 36
\end{matrix} \right]
y =\left[\begin{matrix}
460 \cr 232 \cr 315 \cr 178
\end{matrix}\right]
$$
$$X$$ is $$m$$x$${n+1}$$, $$y$$ is m-dimensional and
$$\theta = (X^{T}X)^{-1}X^{T}y $$

We use ```pinv``` in case $$X^TX$$ is *noninvertible*: features are linearly dependent or $$m\le n$$. Delete features or use regularization.
```MatLab
pinv(X'*X)*X'*y
```
No need to use features scaling with gradient descent.
