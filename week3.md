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
g\left(z\right) = \frac{1}{\left(1+e^-z\right)}
\end{aligned}
$$

Note $$g\left(z\right)$$ approaches 1 as $$z$$ gets larger towards $$\infty$$ and approaches 0 as $$z$$ gets smaller towards $$-\infty$$.  So $$g(z)$$ and  $$h(x)$$ are bounded at 0 and 1.

### Decision Boundary

## Logistic Regression Model

### Cost Function
### Simplified Cost Function and Gradient Descent
### Advanced Optimization

## Multiclass Classification
### Multiclass Classification: One-vs-all
