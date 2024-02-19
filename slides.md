---
background: ./waves.png
class: text-center
info: |
  ## Weighted likelhood filter
title: WoLF — (W)eighted (L)ikelihood (F)ilter
mdc: true
---

# WoLF
## (W)eighted (L)ikelihood (F)ilter

Gerardo Duran-Martin

---
layout: center
---

# (Markovian) State space models (SSM)

---

## SSMs (con'd)

Measurements  $\bm y_t \in \reals^d$ are modelled by an unobserved (latent) state process $\bm\theta_t \in \reals^p$:

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

1. Estimate state $\mathbb{E}[\bm\theta_{t} \vert {\cal D}_{1:t}]$, or
2. one-step-ahead forecast $\mathbb{E}[\bm y_t \vert {\cal D}_{1:t-1}]$.

---
layout: center
---

# What for?*

*See paper for details.

---

## (i) Tracking a moving object

State $\bm\theta_t$: linear and known |  $\bm y_t$ measurement: linear and known.

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/2d-tracking-state.gif">

---

## (ii) Weather forecasting

State $\bm\theta_t$: non-linear and known | measurement $\bm y_t$: linear and known.


<img class="horizontal-center" width=75%
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/waves.gif">

---

## (iii) Sequential training of non-linear models (neural networks)

State $\bm\theta_t$: linear and unknown | measurement $\bm y_t$: non-linear and unknown.

Estimate next value in the sequence using a neural network. One update per iteration. State has no physical interpretation.

We observe measurements from an unknown function and estimate it using a neural network. Suppose $h_t(\bm\theta_t) = h(\bm\theta, x_t)$ with $x_t \in \reals$

---

### (iii) cont’d

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/1dregression-corrupted00-EKF.gif">

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

then the Kalman filter (KF) is the *optimal* filter in terms of residual mean squared error.

Vast literature, for non-linear / non-Gaussian / non-Markovian SSMs, but we focus on closed-form update rules for computational tractability. 

---

# The Kalman filter
Linear filtering problem.

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

# Extensions to the KF

---

## Extended Kalman filter (EKF)

Replace measurement and state model with first-order approximations

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

## Extensions to the KF cont’d

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
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/2d-tracking-KF.gif">

---

## (ii) Atmosphere model

Any component has probability 0.2% of taking the value 100.

<img class="horizontal-center" width=70%
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/waves-corrupt.gif">

---

### (ii) Atmosphere model (cont'd)

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/lorenz96-enkf-sample.png">

---

## (iii) Sequential training of non-linear models

Any measurement has 15% probability of taking value between -50 and 50.

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/1dregression-corrupted15-EKF.gif">

---

# The weighted likelihood filter — WoLF

Replace measurement model with

$$
\begin{aligned}
\log q(\bm y_t \vert \bm\theta_t) &= W_t(\bm y_{1:t})\,\log{\cal N}(\bm y_t \vert h_t(\bm\theta_t), {\bf R}_t) 
\end{aligned}
$$

---

### WoLF — cont’d

**Update step**

$$
\begin{aligned}
\hat{\bm y}_t &= {\bf H}_t\bm\mu_{t|t-1}\\

{\bf S}_t &= 

{\bf H}_t\bm\Sigma_{t|t-1}{\bf H}_t^\intercal + {\bf R}_t
{\color{teal}
/ W_t(\bm y_{1:t})} \\

{\bf K} &= \bm\Sigma_{t|t-1}{\bf H}_t^\intercal{\bf S}_t^{-1}\\

\bm\mu_t &=
\bm\mu_{t|t-1} + {\bf K}_t({\bm y}_t - \hat{\bm y}_t)\\

\bm\Sigma_t &=
\bm\Sigma_{t|t-1} - {\bf K}_t{\bf S}_t{\bf K}_t^\intercal
\end{aligned}
$$

---

### Why?

1. For some choices of $W_t$, we show that our method is provably robust — in a posterior influence function (PIF) sense—,
2. straightforward to implement, and
3. orders of magnitude faster than alternative robust methods.

---

## Our choice of weighting function: the IMQ

Inverse multi-quadratic

$$
W_t({\bm y}_{1:t}) = \left(1 + \frac{\|\bm y_t - \hat{\bm y}_t\|_2^2}{c^2}\right)^{-1/2}
$$

with $c > 0$ the soft-threshold.

Measures our tolerance to the worst-case inlier.

---

## Examples

---

### (i) 2d-tracking problem

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/2d-tracking.gif">

---

### (ii) Atmosphere model

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/lorenz96-wenkf-enkf-sample-inflation-covariance.png">

---

### (iii) Sequential training of neural networks

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/1dregression-corrupted15-WLF-IMQ.gif">

---

## What about alternative methods?

Probabilistic-based methods to filtering usually solve the fixed-point solution to a variational-Bayes problem. This makes them

1. Not scalable to high-dimensional problems,
2. sensitive to the choice of hyperparameters, 
3. not provably robust.

---

### What about alternative methods? (cont’d)

<img class="horizontal-center" width=500
     src="/WoLF%20%E2%80%94%20talk%20f37b4510d28444f69ff7f03aaf36192a/relative-ogd-metrics-datasets.png">
    
---
layout: end
---
Done.