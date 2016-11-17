# Week 3: Logistic Regression

## Classification and Representation

### Classification

Variable $$y$$ is a discrete value in the model, such as is this email spam or not?

$$ y \in \left\{ 0,1 \right\} $$ 0 is the negative class, 1 is the positive

$$ h\left(\theta\right) = \theta^\intercal x $$

Threshold classifier output $$h_\theta(x) = 0.5 $$

If

$$
\begin{aligned}
h_\theta(x) \geq 0.5 y = 1 \\
h_\theta(x) \lt 0.5 y = 0
\end{aligned}
$$

### Hypothesis Representation

We want $$ 0 \leq h_\theta\left(x\right) \leq 1 $$.

Logistic regression sigmoid function.

$$
\begin{aligned}
h_\theta\left(x\right) = g\left(\theta^\intercal x\right) \\
 \\
g\left(z\right) = \frac{1}{\left(1+e^-z\right)}
\end{aligned}
$$

Note $$g\left(z\right)$$ approaches 1 as $$z$$ gets larger towards $$\infty$$ and approaches 0 as $$z$$ gets smaller towards $$-\infty$$.  So $$g(z)$$ and  $$h(x)$$ are bounded at 0 and 1.

We interpret $$h_\theta\left(x\right)$$ as the estimated *probability* that $$y=1$$ for for input $$x$$.

Example: for $$ x = \left[\begin{matrix}x_0 \cr x_1\end{matrix} \right] = \left[\begin{matrix}1 \cr  tumorSize\end{matrix} \right]$$ (a toy model where were size of tumor is the sole criteria for malignancy) we take $$h_\theta\left(x\right) = 0.7$$ as meaning there is a 70% chance the tumor is malignant.

probability that $$y=1$$ given $$x$$ paramaterized by $$\theta$$ is written as $$h_\theta\left(x\right) = P\left(y=1|x; \theta\right)$$

Some properties:
$$P\left(y=0|x; \theta\right) + P\left(y=1|x; \theta\right) = 1$$
and
$$P\left(y=0|x; \theta\right) = 1 - P\left(y=1|x; \theta\right)$$
### Decision Boundary
We predict that $$y = 1$$ if $$h_\theta\left(x\right) \ge 0.5$$
We predict that $$y = 0$$ if $$h_\theta\left(x\right) \lt 0.5$$

When exactly is $$h_\theta\left(x\right) \ge 0.5$$?


$$g(x) \ge 0.5$$ when $$z \ge 0$$
$$h_\theta(x) = g(\theta^\intercal) \ge 0.5 $$ whenever $$ \theta^\intercal x \ge 0$$ when $$ z = \theta^\intercal x$$

The decision boundary is a property of the hypothesis and the parameters **not** the training set.
## Logistic Regression Model

### Cost Function
### Simplified Cost Function and Gradient Descent
### Advanced Optimization

## Multiclass Classification
### Multiclass Classification: One-vs-all
