---
layout: cover
background: ./waves.png
class: text-center
info: |
  ## Weighted likelhood filter
title: WoLF — (W)eighted (L)ikelihood (F)ilter
---


# WoLF
## (W)eighted-(o)bservation (L)ikelihood (F)ilter

Gerardo Duran-Martin

 <small>
 Joint work with:
 Matias Altamirano,
 Alexander Y. Shestopaloff,
 Leandro Sanchez-Betancourt,
 Jeremias Knoblauch,
 Matt Jones,
 François-Xavier Briol, and
 Kevin Murphy.
 </small>

---
layout: center
---

# State space models (SSMs)

---

## SSMs
Measurements  $\bm y_t \in \R^d$ are modelled by an unobserved (latent) state process $\bm\theta_t \in \R^p$:

$$
\begin{aligned}
p(\bm\theta_t \vert  \bm\theta_{t-1}) &= {\cal N}(\bm\theta_t \vert f_t(\bm\theta_{t-1}), {\bf Q}_t),\\
p(\bm y_t \vert \bm\theta_t) &= {\cal N}(\bm y_t \vert h_t(\bm\theta_t), {\bf R}_t).
\end{aligned}
$$


With

- ${\bf Q}_t$ the state covariance,
- ${\bf R}_t$ the measurement covariance,
- $f_t: \R^p \to \R^p$ the state-transition function, and
- $h_t: \mathbb{R}^p \to \mathbb{R}^d$ the measurement function.

---

## The goal of an SSM*

*For purposes of this talk

1. Estimate state $\mathbb{E}[\bm\theta_{t} \vert {\bm y}_{1:t}]$, or
2. one-step-ahead forecast $\mathbb{E}[\bm y_t \vert {\bm y}_{1:t-1}]$.

---
layout: center
---

# What for?

---

## (i) Tracking a moving object

State transition $\bm\theta_t$: linear and known |  $\bm y_t$ measurement: linear and known.
<img class="horizontal-center" width=500
     src="/attachment/b9809328fd8bed9621d4a7f20821ced7.gif">

---

## (ii) Weather forecasting

State transition $\bm\theta_t$: non-linear and known | measurement $\bm y_t$: linear and known.

<img class="horizontal-center" width=75%
     src="/attachment/950ffd8667e5a9dc18844ea7bb69b612.gif">

---

## (iii) Sequential training of non-linear models (neural networks)

State transition $\bm\theta_t$: linear | measurement function $h_t$: non-linear (misspecified).
State has no physical interpretation.
Here, $h_t(\theta) = \theta^{(3)}\,g\Big(\theta^{(2)}\,g\left(\theta^{(1)}{\bf x}_t + b^{(1)}\right) + b^{(2)}\Big) + b^{(3)}$ is a two-layered neural network with activation function $g$.


<img class="horizontal-center" width=500
     src="/attachment/1e9c7bc1936b8d406c0d6fe557d7666f.gif">

---

# How to solve the filtering problem?

If the SMM is linear, i.e.,

$$
\begin{aligned}
p(\bm\theta_t \vert \bm\theta_{t-1}) &= {\cal N}(\bm\theta_t \vert {\bf F}_t\bm\theta_{t-1}, {\bf Q}_t)\\
p({\bf y}_t \vert \bm\theta_t) &=
{\cal N}({\bf y}_t \vert {\bf H}_t\bm\theta_t, {\bf R}_t)
\end{aligned}
$$

the Kalman filter (KF) is the *optimal* filter in terms of residual mean squared error.

---

## The Kalman filter
Estimates a predict and an update step
$$
\begin{aligned}
p(\bm\theta_t \vert \bm y_{1:t-1})
&= {\cal N}(\bm\theta \vert \bm\mu_{t|t-1}, \bm\Sigma_{t|t-1}) & \text{(Predict)}\\
p(\bm\theta_t \vert \bm y_{1:t})
&= {\cal N}(\bm\theta_t \vert \bm\mu_t, \bm\Sigma_t) & \text{(Update)}
\end{aligned}
$$

---

### The Kalman filter (cont'd)
**Predict step**
$$
\begin{aligned}
\bm\Sigma_{t|t-1} &= {\bf F}_t^\intercal\bm\Sigma_{t-1}{\bf F}_t + {\bf Q}_t\\
\bm\mu_{t|t-1} &= {\bf F}_t\bm\mu_{t-1}
\end{aligned}
$$

**Update step**
$$
\begin{aligned}
\hat{\bm y}_t &= {\bf H}_t\bm\mu_{t|t-1}\\

{\bf S}_t &= {\bf H}_t\bm\Sigma_{t|t-1}{\bf H}_t^\intercal + {\bf R}_t \\

{\bf K} &= \bm\Sigma_{t|t-1}{\bf H}_t^\intercal{\bf S}_t^{-1}\\

\bm\mu_t &=
\bm\mu_{t|t-1} + {\bf K}_t({\bm y}_t - \hat{\bm y}_t)\\

\bm\Sigma_t &=
\bm\Sigma_{t|t-1} - {\bf K}_t{\bf S}_t{\bf K}_t^\intercal
\end{aligned}
$$

---
layout: center
---

# KF extensions

---

## KF extensions cont’d

Vast literature on non-linear / non-Gaussian / non-Markovian SSMs, but we focus on closed-form update rules for computational tractability. 
In particular, we consider
1. Extended Kalman filter (EKF), and
2. Ensemble Kalman filter (EnKF).

---

## Extended Kalman filter (EKF)
For non-linear state and/or measurement functions.
Replace functions with first-order approximations centred around previous mean.
$$
\begin{aligned}
p(\bm\theta_t \vert  \bm\theta_{t-1}) &= {\cal N}(\bm\theta_t \vert \bar f_t(\bm\theta_{t-1}), {\bf Q}_t)\\
p(\bm y_t \vert \bm\theta_t) &= {\cal N}(\bm y_t \vert \bar h_t(\bm\theta_t), {\bf R}_t) 
\end{aligned}
$$

with

- $\bar f_t(\bm\theta_{t-1}) = f(\bm\mu_{t-1}) + {\bf F}_t(\bm\theta_{t-1} - \bm\mu_{t-1})$,
- $\bar h_t(\bm\theta_t) = h_t(\bm\mu_{t|t-1}) + {\bf H}_t(\bm\theta_t - \bm\mu_{t|t-1})$,
- ${\bf F}_t = \text{Jac}(f_t)(\bm\mu_{t-1})$, ${\bf H}_t = \text{Jac}(h_t)(\bm\mu_{t|t-1})$

---

## Ensemble Kalman filter (EnKF)

Sample-based predict step

$$
\begin{aligned}
\hat{\bm\theta}_t^{(i)} &\sim {\cal N}(f_{t}\big(\bm\theta_{t-1}^{(i)}), {\bf Q}_t\big)\\
\hat{\bm y_t}^{(i)} &\sim {\cal N}\big(h_t(\bm\theta_t^{i}), {\bf R}_t\big)
\end{aligned}
$$

Update follows by computing a sample-based gain matrix $\hat{\bf K}_t$.

---

## Extensions to the KF  --- cont’d

- Experiment (i) is filtered using the KF,
- experiment (ii) is filtered using the EnKF, and
- experiment (iii) is filtered (trained) using the EKF.

---
layout: center
---

# The problem with KF-like methods

Sensitive to outliers and misspecified measurement models

---

## (i) Tracking problem

Measurements sampled from a Student-t distribution with 2.01 degrees of freedom.
<img class="horizontal-center" width=500
     src="/attachment/887c8cd1689889df9c7a6b29f1977fac.gif">

---

## (ii) Atmosphere model

Any component has probability 0.2% of taking the value 100.

<img class="horizontal-center" width=500
     src="/attachment/f430213d98f7d16e7f86cf143bd32f7a.gif">

---

### (ii) Atmosphere model (cont'd)

<img class="horizontal-center" width=500
     src="/attachment/2565ad74b269bcbdf55e7e7f385a7196.png">

---

## (iii) Sequential training of non-linear models

Any measurement has 15% probability of taking value between -50 and 50.

<img class="horizontal-center" width=500
     src="/attachment/36ef70bb76373ec5358575a6807825f0.gif">

---

# Robustifying the KF (via variational Bayes)
Literature in recent years has focused in making robust versions of the KF by increasing the state-space and finding a variational Bayes (VB) posterior.

---

## Agamenoni et.al (2012) --- compensation-based
Extend state-space to include the measurement noise covariance matrix ${\bm R}_t$.
$$
\begin{aligned}
    p({\bm \theta}_t\vert{\bm \theta}_{t-1}) &=
    {\cal N}({\bm \theta}_t \vert f({\bm \theta}_{t-1}), {\bf Q}_t)\\
    p({\bm R}_t) &= {\cal W}^{-1}({\bm R}_t \vert \nu{\bm\Lambda}, \nu)\\
    p({\bm y}_t \vert {\bm\theta}_t, {\bm R}_t) &= 
    {\cal N}({\bm y}_t \vert h_t({\bm \theta_t}), {\bm R}_t)
\end{aligned}
$$

VB posterior is
$$
q_R^*, q_\theta^* =
\arg\min_{q_\theta, q_R}\mathbb{KL}\Big(
q_R({\bm R}_t)\,q_\theta({\bm \theta}_t)
\,\|\,
p({\bm R}_t, {\bm \theta}_t \vert {\bm y}_{1:t})
\Big)
$$

---

## Wang et.al (2018) --- detect-and-reject
Extend state-space to include a binary variable $w_t$ that indicates whether the measurement is informative or not.
$$
\begin{aligned}
    p({\bm \theta}_t\vert{\bm \theta}_{t-1}) &=
    {\cal N}({\bm \theta}_t \vert f({\bm \theta}_{t-1}), {\bf Q}_t)\\
    p(\xi_t) &= {\rm Beta}(\xi_t \vert \alpha_0, \beta_0) \\
    p(w_t \vert \xi_t) &= {\rm Bern}(w_t \vert \xi_t)\\
    p({\bm y}_t \vert {\bm\theta}_t, {\bm R}_t) &= 
    \begin{cases}
        {\cal N}({\bm y}_t \vert h_t({\bm \theta_t}), {\bm R}_t)&
        \text{if } w_t=1\\
        1 & \text{if } w_t = 0 
    \end{cases}
\end{aligned}
$$

VB posterior is

$$
q_\xi^*, q_w^*, q_\theta^* =
\arg\min_{q_\xi, q_w, q_\theta}\mathbb{KL}\Big(
q_\xi(\xi_t)\,q_w(w)\,q_\theta({\bm \theta}_t)
\,\|\,
p(\xi_t, w_t, {\bm \theta}_t \vert {\bm y}_{1:t})
\Big)
$$

---

## Robust KF via VB --- cont’d

The methods above:
1. are not straightforward to implement,
2. are not provably robust,
3. are at least $I$ times (number of inner iteration) slower than KF methods.

---
layout: center
---

# The weighted likelihood filter (WoLF)

----

# The weighted likelihood filter — WoLF
Replace log-likelihood with loss function of the form
$$
\begin{aligned}
\ell_t(\bm\theta_t) &= -W_t(\bm y_{1:t})\,\log{\cal N}(\bm y_t \vert h_t(\bm\theta_t), {\bf R}_t).
\end{aligned}
$$
We estimate
$$
    q(\bm\theta_t  \vert \bm y_{1:t})
    \propto \exp(-\ell_t(\bm\theta_t))q(\bm\theta_t \vert \bm y_{1:t-1})
$$

---

## Why?
1. For some choices of $W_t$, we show that our method is **provably robust**,
2. straightforward to implement, and
3. orders of magnitude faster than alternative robust methods.

---

## WoLF — cont’d
For a linear SSM, WoLF is shown to be a modification of the KF update step.

**Update step**

$$
\begin{aligned}
\hat{\bm y}_t &= {\bf H}_t\bm\mu_{t|t-1}\\

{\bf S}_t &= 

{\bf H}_t\bm\Sigma_{t|t-1}{\bf H}_t^\intercal + {\bf R}_t
{\color{red}
/ W_t(\bm y_{1:t})} \\

{\bf K} &= \bm\Sigma_{t|t-1}{\bf H}_t^\intercal{\bf S}_t^{-1}\\

\bm\mu_t &=
\bm\mu_{t|t-1} + {\bf K}_t({\bm y}_t - \hat{\bm y}_t)\\

\bm\Sigma_t &=
\bm\Sigma_{t|t-1} - {\bf K}_t{\bf S}_t{\bf K}_t^\intercal
\end{aligned}
$$


---

## Our choice of weighting function: the IMQ

Inverse multi-quadratic

$$
W_t({\bm y}_{1:t}) = \left(1 + \frac{\|\bm y_t - \hat{\bm y}_t\|_2^2}{c^2}\right)^{-1/2}
$$

with $c > 0$ the soft-threshold.

Measures our tolerance to the worst-case inlier.

---

## Robustness

**Definition: posterior influence function (PIF)**  
$$
\text{PIF}(\bm y_t^c, \bm y_{1:t}) = \text{KL}(
    q(\bm\theta_t  \vert \bm y_t^c, \bm y_{1:t-1})\,\|\,
    q(\bm\theta_t  \vert \bm y_t, \bm y_{1:t-1})
    ).
$$

<br>

**Definition: outlier-robust filter**  
A filter is outlier-robust if the PIF is bounded, i.e.,
$$
\sup_{\bm y_t^c\in\R^d}|\text{PIF}(\bm y_t^c, \bm y_{1:t-1})| < \infty.
$$

---

### Theorem
If $\sup_{\bm y_t\in\R^d} W_t(\bm y_t) < \infty$ and $\sup_{\bm y_t\in\R^d} W_t(\bm y_{1:t})^2 \|\bm y_t\|^2 < \infty$ then the PIF is bounded.

<br> 

**Remarks**
1. The KF is not outlier robust
2. WoLF with IMQ weighting function is outlier robust

---

### Robustness --- cont’d
<img class="horizontal-center" width=75%
     src="/attachment/20240501-2d-tracking-grid-pif.png">

---
layout: center
---

# Examples

---

## (i) 2d-tracking problem

<img class="horizontal-center" width=75%
     src="/attachment/27222fee22ea59e8ff7e133f22f0e38c.gif">

---

## (ii) Atmosphere model

<img class="horizontal-center" width=500
     src="/attachment/268c89c57e0a44328071c441e96d3ade.png">

---

## (iii) Sequential training of neural networks

<img class="horizontal-center" width=500
     src="/attachment/b90056b22de044b15657d12ad3258d12.gif">


---

### What about alternative methods?
Online training of neural networks in corrupted UCI datasets.

<figure>
<img class="horizontal-center" width=500
     src="/attachment/1d5a92fed176545c125bc7e8f5661b84.png">
<figcaption>
     Results are shown relative to online gradient descent (OGD) with multiple inner iterations.
</figcaption>
</figure>

    
---
layout: end
---
gerdm.github.io/weighted-likelihood-filter