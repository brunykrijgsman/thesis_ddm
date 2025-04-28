# Directed model (7 parameters)

$$
r_i, x_i \sim DDM(\alpha, \tau, \delta_i, \beta) \\
z_i \sim N(\mu_{z}, \sigma^2) \\
\delta_i \sim N(\lambda z_i + b, \eta)
$$

This model is a variation of the Drift Diffusion Model (DDM).

---

### 1. The Observed Data

- **$r_i$**: the observed response time (RT) on trial $i$.
- **$x_i$**: the observed choice/response (a binary variable indicating which boundary was reached) on trial $i$.

Both $\{r_i, x_i\}$ are assumed to come from a _Drift Diffusion Model_ (DDM) with four parameters $\alpha,\;\tau,\;\delta_i,\;\beta$.

---

### 2. The Drift Diffusion Model (DDM)

The standard (or "basic") Drift Diffusion Model typically includes:

1. **Boundary separation $\alpha$**:  
   Controls the distance between the two absorbing decision boundaries. Large $\alpha$ implies that more evidence is required before making a decision, often resulting in slower but more accurate responses.

2. **Non-decision time $\tau$**:  
   Accounts for sensory encoding and motor execution components of the RT that are not part of the decision process itself. It is essentially the time added (from stimulus presentation) onto the diffusion process before a response is observed.

3. **Drift rate $\delta_i$**:  
   Governs how quickly evidence is accumulated (on average) toward one boundary or the other. In many DDMs this is denoted by $v_i$, but here it is $\delta_i$. A higher drift rate implies a stronger “pull” toward one of the boundaries, typically reflecting an easier decision or stronger biasing evidence for one choice over the other.

4. **Starting point / bias $\beta$**:  
   Determines if the diffusion process is biased toward one boundary or the other from the onset. If $\beta = 0.5$, there is no a priori bias, but if $\beta \neq 0.5$, the process starts closer to one boundary than the other.

Hence, writing $\{r_i, x_i\} \sim \mathrm{DDM}(\alpha, \tau, \delta_i, \beta)$ means “the trial-level response time and decision outcome are distributed according to a DDM with these parameters.”

---

### 3. The Latent Variable $z_i$

- **$z_i$** is a latent (unobserved) variable for each trial $i$.
- It is assumed to follow a normal distribution with mean $\mu_{z}$ and variance $\sigma^2$:
  $$
  z_i \sim N(\mu_{z}, \sigma^2).
  $$
  This distribution captures whatever latent factor might influence the drift rate across trials—e.g., difficulty, attention, or some subject-level or trial-level covariate.

---

### 4. Linking the Latent Variable to the Drift Rate

- The drift rate for trial $i$, $\delta_i$, depends _linearly_ on $z_i$:
  $$
  \delta_i = \lambda \, z_i + b.
  $$
  Here:
  - **$b$** is an intercept or “baseline” drift rate.
  - **$\lambda$** is a slope that scales how strongly the latent factor $z_i$ influences the drift.

In other words, if $z_i$ is large (perhaps the trial is “easier” or the subject is more alert), then $\delta_i$ will be higher, thus leading on average to faster and more accurate decisions. Conversely, if $z_i$ is small (or negative), the drift rate $\delta_i$ is lower, leading to slower and potentially less accurate decisions.

---

### 5. Putting It All Together

1. **Draw the latent factor:** For trial $i$, sample $z_i$ from $N(\mu_{z}, \sigma^2)$.

2. **Compute the drift rate:** Given that draw, set $\delta_i = \lambda\, z_i + b$.

3. **Generate RT and choice:** Conditioned on $\alpha$, $\tau$, $\beta$, and $\delta_i$, the DDM describes how $\{r_i, x_i\}$ are distributed.

This results in a hierarchical or multi-level model:

- **7 parameters** in total:

  1. $\alpha$ (boundary separation),
  2. $\tau$ (non-decision time),
  3. $\beta$ (starting bias),
  4. $\mu_{z}$ (mean of the latent variable),
  5. $\sigma$ (standard deviation of the latent variable),
  6. $\lambda$ (slope linking $z_i$ to the drift),
  7. $b$ (intercept/baseline drift rate).

- **Trial-level random effects:** $\delta_i$ are not independent parameters but determined by $\lambda, b,$ and the latent draw $z_i$.

- **Observations:** The observed data ($r_i, x_i$) on each trial are thus driven by the underlying latent factor $z_i$.

---

## Intuitive Interpretation

1. You have a “baseline” decision process described by $\alpha$, $\tau$, and $\beta$.
2. On each trial, there is a hidden factor $z_i$ that shifts the drift rate $\delta_i$.
3. If $z_i$ is high (e.g., an “easy” trial), then the drift $\delta_i$ is large and the decision accumulates evidence quickly.
4. If $z_i$ is low (e.g., a “hard” trial), then $\delta_i$ is small or even negative, slowing down the accumulation or favoring a particular outcome.
5. Over many trials, $\{z_i\}$ follows a Normal($\mu_{z}, \sigma^2$) distribution, capturing overall difficulty or other latent influences.

Thus, this model extends a standard DDM by allowing the drift rate to vary from trial to trial (or person to person) through a latent normally distributed variable $z_i$.
