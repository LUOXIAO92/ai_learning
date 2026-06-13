# From History-Based RL to Resampling and a Physical Picture of RL

## 0. Goal

This text develops a **physical picture** for RL. It does not try to rewrite RL as a strict field theory. The discussion focuses on physical methods for expanding path sampling. It has two main lines:
- **Resampling**: under the original RL definition of cumulative return, resample or re-estimate the observation $o_{t+1}$ and reward $r_t$ to improve rollout, return estimation, advantage estimation, and PPO / GRPO / GSPO updates
- **Path integral**: under a path-integral picture, introduce an effective Hamiltonian, Boltzmann weights, Gibbs sampling, MCMC / Langevin, and inverse-temperature annealing, then use them to expand sampling in path space and search for low-Hamiltonian paths

RL and physics share two basic ingredients:
- history-based RL can be written as a path integral along a one-dimensional time direction
- the return $G[\tau]$ in the original expectation is an observable on the path

---

## 1. The Original Trajectory Integral of History-Based RL

Start from the most primitive form. The interaction history is
```math
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\cdots,a_{t-1},r_{t-1},o_t)
```

The policy is $\pi(a_t\mid h_t)$, and the environment conditional density is $\mu(o_{t+1},r_t\mid a_t,h_t)$. Within a finite horizon $T$, the expected return over a whole trajectory can be written as
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
```

The whole path can then be written as
```math
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
```

The path return is
```math
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s
```

So the original RL objective integrates over all possible paths with weights supplied by the policy and the environment. Each path receives its value from the discounted return $G[\tau]$.

### 1.1. Dirac Delta Degeneration in a Deterministic Environment

If the environment is deterministic, then after we fix $a_t,h_t$, deterministic functions give the next observation and reward:
```math
o_{t+1}=O(a_t,h_t),\quad r_t=R(o_{t+1},a_t,h_t)
```

The environment conditional density degenerates into Dirac deltas:
```math
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))
```

Substitute this into the original trajectory integral:
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
```

Use the basic integration property of the Dirac delta:
```math
\int \delta(x-x_0)f(x)\,dx=f(x_0)
```

After the environment part collapses, only action sampling remains:
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right]
```

This means:
- the deterministic environment no longer supplies path branches; only policy sampling supplies them
- if the policy is also deterministic, the whole path collapses to a single path

### 1.2. Finite Horizon, Finite Reward
To prevent reward divergence, besides setting the discount to $0<\gamma\leq 1$, we can also
- limit the maximum path length $T\le T_{\max}$, so that the path integral runs only on a finite time interval:
```math
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right)
```

- clip rewards:
```math
\bar r_t=\mathrm{clip}(r_t,-r_{\max},r_{\max}),\quad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t
```

### 1.3. Path Sampling and Hidden Actions in LLM RL

For LLM RL, we need to separate external trajectory steps from autoregressive token steps. This text uses the following convention:
- $t=0,\ldots,T$ denotes external trajectory steps, such as one question-answer turn, one tool call, one environment interaction, or one agent step
- $i$ denotes the token position generated autoregressively inside the external step $t$

Let the token interval corresponding to the external step $t$ be $L_t\le i\le L_{t+1}-1$. Given the external-step history $h_{t} = [h_{t-1}, a_{L_{t-1}}, \cdots a_{L_t-1}, o_{t}, r_{t-1}] $, define the token-prefix history inside the same external-step interval as $h_{i,t} \equiv \text{concat}(h_t, [a_{L_t},a_{L_t+1},\ldots,a_{i-1}])$. In this interval, the model generates tokens as follows:
- predict the logit of token $i$:
```math
z_i^{\mathrm{logit}} = f_\theta(h_{i,t})
```
- sample, with top-p, top-k, or other decoding strategies available at this step:
```math
\pi_{\theta,T_{\mathrm{dec}}}(a_i\mid h_{i,t}) =  \frac{\exp(z^{\mathrm{logit}}_{i,a_i}/T_{\mathrm{dec}})} {\sum_{a'}\exp(z^{\mathrm{logit}}_{i,a'}/T_{\mathrm{dec}})}
```
- concatenate the new token according to the definition of $h_{i,t}$ above


The original coarse-grained action $a_t$ should therefore be read as a whole generated segment in the autoregressive LLM case. The token sequence follows the joint distribution
```math
\pi_\theta(a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}|h_t) = \prod_{i=L_t}^{L_{t+1}-1} \pi_\theta(a_i|h_{i,t})
```
So the coarse-grained action $a_t$ can be represented as a sample from this joint distribution:
```math
a_t \sim \prod_{i=L_{t}}^{L_{t+1}-1} \pi_\theta(\cdot|h_{i,t}), \quad a_t = [a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}]
```

The full action measure of the autoregressive structure in integral form becomes
```math
\prod_{t=0}^{T} da_t~ \pi_\theta(a_t|h_{t})  \equiv \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1} da_i~ \pi_\theta(a_i|h_{i,t}) 
```

When the LLM generates a complete token sequence $a_t$ and interacts with the environment to obtain the observation $o_{t+1}$ and return $r_t$, we can still write the history as $h_{t+1} = [h_t, (a_t, o_{t+1}, r_t)]$. Under this convention, the LLM trajectory keeps the original coarse-grained form:
```math
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
```

To avoid a wrong reading, $a_t$ here is a segment generated autoregressively, not a single token. In short, the autoregressive LLM structure expands one external action $a_t$ into an intra-block token integral. After integration, the outer path variable remains the stage-level action $a_t$. If we want to sample paths in a continuous space, we can treat token hiddens as continuous degrees of freedom. Write the intra-block hidden as
```math
z_t\equiv[z_{L_t},z_{L_t+1},\ldots,z_{L_{t+1}-1}],\quad z_i \in \mathbb{R}^{d_\mathrm{model}}
```

The continuous-action path can be written as
```math
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T)
```

Here $z_t$ also denotes a whole segment of hidden variables inside the external step $t$. If $b$ denotes the $b$-th sample, the sampled trajectory is
```math
\tau_b=(a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
```

At this point, the reward can be written as
```math
\begin{aligned}
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t}
\end{aligned}
```
$\omega^{i-L_t}$ and $\gamma^t$ are the token-level discount and task-step-level discount. $R(h_{i,t}, a_{i})$ is the token-level reward, while $\phi(h_{i,t}, a_{i}, o_{i+1})$ can serve as a stage reward. When $t=T$, it can reduce to the terminal reward.

---

## 2. The Physical Picture of the Original History-Based RL

### 2.1. From Trajectory Integral to Path Integral

First write the product of the original probability densities as a path density:
```math
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T} \pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)
```

Then the expected cumulative return $J$ can be written as a functional integral:
```math
J(\pi)=\int \mathcal{D}\tau ~ P_{\pi,\mu}[\tau]G[\tau],\quad \mathcal{D}\tau = \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
```

Now define the action $S_{\pi,\mu}[\tau]$:
```math
S_{\pi,\mu}[\tau] = - \log P_{\pi,\mu}[\tau] = - \sum_{t=0}^T \left[ \log \pi(a_t|h_t) + \log \mu(o_{t+1},r_t | h_t, a_t) \right]
```
The original trajectory integral can be described as a path integral:
```math
J(\pi)=\int \mathcal{D}\tau ~ e^{-S_{\pi,\mu}[\tau]} G[\tau]
```

For LLM autoregressive generation, following the convention in Section 1.3:
```math
da_t\equiv\prod_{i=L_t}^{L_{t+1}-1}da_i,\quad
\pi(a_t\mid h_t)\equiv\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
```

The LLM path density is therefore
```math
P^{AR}_{\pi,\mu}[\tau] = \prod_{t=0}^{T} \left[ \prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t}) \right]
\mu(o_{t+1},r_t\mid h_t,a_t)
```

The action is
```math
S^{AR}_{\pi,\mu}[\tau] = - \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \left( \log \pi(a_i|h_{i,t}) + \delta_{i,L_{t+1}-1} \log \mu(o_{t+1},r_t | h_t, a_t) \right)
```

The propagator at time $t$ is
```math
\begin{aligned}
f(h_{t+1}, h_t) &= \left[\prod_{i=L_t}^{L_{t+1}-1} \pi(a_i|h_{i,t})\right]\mu(o_{t+1},r_t|h_t,a_t)\\
\Delta h_t &= [a_t,o_{t+1},r_t] ,~ h_{t+1} = \text{concat}(h_t, \Delta h_t)
\end{aligned}
```

The corresponding measure is
```math
\mathcal{D}\tau = \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \equiv \prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right] do_{t+1}\,dr_t
```

Under the path-integral representation, reinforcement learning has the following physical interpretation:
- $a_t,o_t,r_t$: degrees of freedom of fields on a one-dimensional time path. The fields here are scalar fields, namely an action field, an observation field, and a return field
- $\mathcal{D}\tau$: the measure over fields on all lattice sites
- $G[\tau]$: the physical observable
- $f(h_{t+1},h_t)$: the propagator. Because history-based RL defines the state through history, the variation here is a sequence $\Delta h_t = h_{t+1}-h_{t}\equiv [a_t, o_{t+1}, r_t]$, not an infinitesimal field variation
- $e^{-S_{\pi,\mu}[\tau]}$: the path-integral weight, where $S_{\pi,\mu}$ is the action
- History-based RL can be viewed as a complex one-dimensional single-particle system with three degrees of freedom: $a,o,r$. Its complexity comes from long-range coupling based on the historical trajectory $h_t$. This coupling is not a simple coefficient such as $w_{o_0,a_0,o_1,r_0,\cdots,a_t,o_{t+1},r_t}$; the policy $\pi$ and environment measure $\mu$ determine it.
- Although we define the negative log path density $- \log P_{\pi,\mu}$ as the action, we can redefine the action because the inverse temperature $\beta$ can control the strength of the global coupling:
```math
S_{\pi,\mu}[\tau] = \beta H_{\pi,\mu}[\tau],\quad H_{\pi,\mu}[\tau] \equiv - \log P_{\pi,\mu}
```

### 2.2. Discount Factor and Laplace Regularization

If the continuous-time return is written as

```math
G[\tau]=\int_0^\infty r(t)\,dt
```

it may diverge. After adding exponential decay:

```math
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt
```

In discrete time:

```math
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t
```

Let the time step be $\Delta t$. The correspondence is $\gamma=e^{-\lambda\Delta t}, ~\gamma^t=e^{-\lambda t\Delta t}$. If the reward is bounded, $|r_t|\le r_{\max}$, then the discounted return is bounded:
```math
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}
```

---

## 3. Resampling Observations and Rewards Under the Original RL Definition

Starting from the original RL definition of cumulative return, we can use resampling to expand the local sampling space of the observation $o_{t+1}$ and reward $r_t$ without changing the policy updater. This helps the model explore paths. From the definition of the environment measure, we can treat latent variables as random perturbations to measurements and rewards, which introduces stochasticity. In LLM RL, observations and rewards are usually deterministic, so path exploration depends on the policy model and the exploration capacity becomes limited. We therefore need random perturbations for observations and rewards to give the LLM a larger exploration space.
- Note: for some tasks, especially tool calls, it is hard or impossible to add random perturbations to the observation, meaning the returned value

More complex approaches can use repeated sampling from the real environment, a world model, a reward model, a verifier, SMC, or CEM proposals. This section does not discuss those resampling methods. It uses the basic Gaussian-noise proposal.

### 3.1. Approximation of Observations $o$ and Rewards $r$ Under Deterministic Conditions
In Section 1, the $\delta$ function let random observations and rewards degenerate into deterministic values, namely values obtained from deterministic physical or mathematical modeling. Here we use the definition of the $\delta$ function:
```math
\delta(x-x_0) = \lim_{\sigma\rightarrow 0}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( -\frac{(x-x_0)^2}{2\sigma^2} \right)
```
We first take a small $\sigma$ and expand the deterministic value into an approximate "probability distribution." Thus a proposal with very small variance can approximate a zero-variance Dirac delta. At this point, $o_{t+1}, r_t$ approximately follow
```math
o_{t+1},r_t \sim \mu_\sigma(\cdot|h_t,a_t) \quad\text{or}\quad o_{t+1}\sim \mu_{\sigma_O}(\cdot|h_t,a_t),~ r_t \sim \mu_{\sigma_R}(\cdot|h_t, a_t, o_{t+1})
```
Choose a Gaussian distribution as the approximation, with the deterministic observation $o$ and reward $r$ as the means. The transition probabilities of the random processes $o\rightarrow o', r\rightarrow r'$ are
```math
q(o'|o) = \frac{1}{\sqrt{2\pi\sigma^2_O}}\exp\left( -\frac{(o'-o)^2}{2\sigma^2_O} \right) ,\quad q(r'|r) = \frac{1}{\sqrt{2\pi\sigma^2_R}}\exp\left( -\frac{(r'-r)^2}{2\sigma^2_R} \right)
```

This approximation makes simulated annealing natural. First consider the **scalar** case. Let $\beta$ be the inverse temperature. Then
```math
q_{\beta} (x'|x) = \sqrt{\frac{\beta}{2\pi\sigma^2}} \exp\left( - \beta \frac{(x'-x)^2}{2\sigma^2} \right)
```
The effective variance is $\sigma^2_{\text{eff}}(\beta) = \sigma^2 / \beta$, so simulated annealing can tune the sampling amplitude by adjusting the inverse temperature. Rewrite the approximate Gaussian distribution as a normal distribution:
```math
q(\xi) = \frac{1}{\sqrt{2\pi}} \exp\left(- \frac{\xi^2}{2} \right), \quad \xi^2 = \beta\frac{(x'-x)^2}{\sigma^2}
```

Now we can perform Gaussian sampling with the new variable $x' = x + \xi / \sqrt{\beta} ,~ \xi \sim \mathcal{N}(0, \sigma^2)$. The sampling order is: 1. generate a random number from the standard normal distribution; 2. compute the new $x'$.

Extend the discussion to the multivariate Gaussian case. Let $\boldsymbol{x} = (x_1, \cdots, x_d)$, and let $\Sigma$ be a positive definite covariance matrix:
```math
q_{\beta}(\boldsymbol{x}'|\boldsymbol{x}) = \frac{\beta_k^{d/2}} {(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{\beta}{2} (\boldsymbol{x}'-\boldsymbol{x})^T\Sigma^{-1}(\boldsymbol{x}'-\boldsymbol{x}) \right)
```

When all eigenvalues of $\Sigma$ approach 0, the expression degenerates at $\beta=1$ into the multidimensional Dirac delta $q_{\beta}(\boldsymbol{x}'|\boldsymbol{x}) \rightarrow \delta^{(d)}(\boldsymbol{x}'-\boldsymbol{x})$. If we need simulated annealing for a deterministic vector, we can construct a covariance matrix $\Sigma = \sigma^2 I + \epsilon^2 (U U^{T} - \mathrm{diag}{UU^T})$, where $I$ is the identity matrix, $U$ is a normalized random real matrix, $\epsilon$ gives a small off-diagonal correlation, and $0 < \epsilon \ll \sigma$. This gives the new distribution $\boldsymbol{x}' = \boldsymbol{x} + \boldsymbol{\eta} / \sqrt{\beta},~ \boldsymbol{\eta}\sim \mathcal{N}(0, \Sigma)$.

The discussion above concerns deterministic vector-valued observations and rewards. It may deviate from the **current** real training environment, especially for LLM reinforcement learning, but it remains useful as a reference.


### 3.2. Coordination Between Random Resampling and PPO / GRPO / GSPO

The previous section approximated observations and rewards under deterministic conditions as an effective environment measure with small variance. Now connect this approximation back to the original RL update process. The original PPO / GRPO / GSPO updater does not need to change. The change happens during rollout: feedback originally supplied by the environment measure $\mu$ is now supplied by the small-variance effective environment measure $\mu_\sigma$.

Under the original environment measure, the policy and environment jointly generate a path:
```math
\tau_b = (a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
```

Here $b$ is the sample index. Under the Gaussian-approximated effective environment measure, the corresponding path is
```math
\tau'_b = (a'_{b,0},o'_{b,1},r'_{b,0},a'_{b,1},o'_{b,2},r'_{b,1},\ldots,a'_{b,T_b},o'_{b,T_b+1},r'_{b,T_b})
```

The prime marks a path generated under perturbed environment feedback. The perturbed observation enters the next-step history:
```math
h'_{b,t+1}=\mathrm{concat}(h'_{b,t},a'_{b,t},o'_{b,t+1},r'_{b,t})
```

The policy still generates the next action, but now conditions on the perturbed history:
```math
a'_{b,t}\sim \pi_\theta(\cdot\mid h'_{b,t})
```

In the scalar approximation, perturbations of observations and rewards can be written as
```math
\begin{aligned}
o'_{b,t+1} &= o_{b,t+1} + \sigma_O\xi_{O,b,t}, \quad \xi_{O,b,t} \sim \mathcal N(0,1) \\
r'_{b,t} &= r_{b,t} + \sigma_R\xi_{R,b,t}, \quad \xi_{R,b,t} \sim \mathcal N(0,1)
\end{aligned}
```

If the reward is recomputed from the perturbed observation, write
```math
r'_{b,t} = R(o'_{b,t+1},a'_{b,t},h'_{b,t})
```

The resulting path return is
```math
G[\tau'_b] = \sum_{t=0}^{T_b} \gamma^t r'_{b,t}
```

If we use the LLM autoregressive return form in Section 1.3, then
```math
G[\tau'_b] = \sum_{t=0}^{T_b} \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r'_{b,i,t}
```

Here $r'_{b,i,t}$ is the training reward assigned, under perturbed feedback, to token $i$ inside external step $t$. If we decompose reward into token-level reward and stage reward, then
```math
r'_{b,i,t} = \omega^{i-L_t}R(h'_{b,i,t},a'_{b,i}) + \delta_{i,L_{t+1}-1}\phi(h'_{b,t},a'_{b,t},o'_{b,t+1})
```

From this point on, omit the prime for brevity. All perturbed quantities below use the original symbols $h, a, r, o$, and so on. When using PPO, the perturbed path $\tau_b$ supplies new returns and advantages. Let the advantage on the perturbed path be $A_{b,i,t}$:
```math
\begin{aligned}
A_{b,i,t} &= Q(h_{b,i,t},a_{b,i}) - V(h_{b,i,t}) \\
Q(h_{b,i,t},a_{b,i}) &\simeq \widehat{G}_{b,i,t} = \sum_{j=i}^{L_{t+1}-1} r_{b,j,t} + \sum_{s=t+1}^{T} \gamma^{s-t} \sum_{j=L_s}^{L_{s+1}-1} r_{b,j,s}
\end{aligned}
```


The policy ratio still uses the probability ratio of the current policy and old policy on the same action:
```math
\rho_{b,t}(\theta) = \frac{ \pi_\theta(a_{b,t}\mid h_{b,t}) }{ \pi_{\theta_{\mathrm{old}}}(a_{b,t}\mid h_{b,t}) }
```

Then the PPO clipped objective can be written as
```math
L_{\mathrm{PPO}} = \mathbb E_{b,t} \left[ \min \left( \rho_{b,t}(\theta)A_{b,t}, \mathrm{clip}(\rho_{b,t}(\theta),1-\epsilon,1+\epsilon)A_{b,t} \right) \right]
```

For LLM autoregressive generation, the coarse-grained action $a_{b,t}$ is a token sequence, so the policy ratio needs to expand into a product of token-level probability ratios:
```math
\rho_{b,t}^{\mathrm{AR}}(\theta) = \prod_{i=L_t}^{L_{t+1}-1} \frac{ \pi_\theta(a_{b,i}\mid h_{b,i,t}) }{ \pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t}) }
```

Equivalently, write the log-ratio as a sum:
```math
\log\rho_{b,t}^{\mathrm{AR}}(\theta)=\sum_{i=L_t}^{L_{t+1}-1}\left[\log\pi_\theta(a_{b,i}\mid h_{b,i,t})-\log\pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t})\right]
```

GRPO / GSPO can use the same structure. For the same input $x$, the policy generates a group of paths under perturbed feedback, $\tau_1,\tau_2,\ldots,\tau_{K_s}$. Each path has its own return $G_b = G[\tau_b],~ b=1, \ldots, K_s$. The within-group mean and variance are
```math
\bar G = \frac{1}{K_s} \sum_{b=1}^{K_s}G_b, \quad (\sigma_G)^2 = \frac{1}{K_s} \sum_{b=1}^{K_s} (G_b-\bar G)^2
```

The within-group normalized advantage is
```math
A_b = \frac{G_b-\bar G}{\sigma_G+\epsilon}
```

GSPO can treat the whole generated sequence as the sampling unit. The return $G_b$ of each sample comes from the full sequence path $\tau_b$, while the policy update still acts on model parameters through the log-prob or log-ratio of each token in the sequence. If we expand the tokens inside sequence $b$, the sequence-level log-ratio is
```math
\log\rho_b^{\mathrm{seq}}(\theta) = \sum_{t=0}^{T_b} \sum_{i=L_t}^{L_{t+1}-1} \left[ \log\pi_\theta(a_{b,i}\mid h_{b,i,t}) - \log\pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t}) \right]
```

Thus the Gaussian-approximated environment feedback can connect directly to PPO / GRPO / GSPO. It changes the observations, rewards, and returns on the rollout path, while the policy update still uses the original ratio, advantage, clipping, or group-normalization structure.

### 3.3. Simulated Annealing

The previous section expanded deterministic observations and rewards into a small-variance Gaussian proposal. Now introduce simulated annealing and use a temperature scheduler to control the perturbation amplitude of this proposal. It acts on Gaussian perturbations of observations and rewards and controls the exploration width of rollout in feedback space.

Let the inverse temperature in round $k$ be
```math
\beta_k=\mathcal B(k),\quad \beta_k>0
```

where $B(k)$ is a manually specified annealing scheduler. For a scalar variable $x$, the Gaussian proposal in round $k$ can be written as
```math
q_{\beta_k}(x'\mid x) = \sqrt{\frac{\beta_k}{2\pi\sigma^2}} \exp\left( -\frac{\beta_k(x'-x)^2}{2\sigma^2} \right)
```

The corresponding sampling form is
```math
x^{(k)} = x+\frac{\sigma}{\sqrt{\beta_k}}\xi_k, \quad \xi_k \sim \mathcal N(0,1) 
```

The effective variance is
```math
\sigma_{\mathrm{eff}}^2(k) = \frac{\sigma^2}{\beta_k}
```

So smaller $\beta_k$ gives a wider proposal, and rollout deviates more in the observation and reward space. Larger $\beta_k$ gives a narrower proposal, and the path stays closer to the original deterministic feedback. If we use temperature $T_k$ rather than inverse temperature, then
```math
T_k=\frac{1}{\beta_k}, \quad \sigma_{\mathrm{eff}}^2(k)=\sigma^2T_k
```

For a multidimensional variable $\boldsymbol{x}$ with covariance matrix $\Sigma$, the annealed proposal in round $k$ is
```math
q_{\beta_k}(\boldsymbol{x}'\mid\boldsymbol{x}) = \frac{\beta_k^{d/2}} {(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{\beta_k}{2} (\boldsymbol{x}'-\boldsymbol{x})^T\Sigma^{-1}(\boldsymbol{x}'-\boldsymbol{x}) \right)
```

The corresponding effective covariance is
```math
\Sigma_{\mathrm{eff}}(k) = \frac{1}{\beta_k}\Sigma
```

So simulated annealing fixes the shape of the base Gaussian proposal and uses the scheduler $\mathcal{B(k)}$ to control its global scale. Monotone cooling can give early rollout larger feedback perturbations and then shrink them toward the original feedback. Cyclic heating and cooling can enlarge the perturbation again after local contraction, simulating quenching and increasing the chance of escaping a local region.

### 3.4. Statistical-Mechanical Interpretation

From the physical picture in Section 2, the path weight of original history-based RL can already be written as a Boltzmann weight:
```math
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)=\exp(-\beta H_0[\tau])
```

Here $H_0[\tau]$ is the base Hamiltonian induced by the original policy and environment, and $\beta$ is the inverse temperature. In the LLM autoregressive form, the policy part expands as
```math
P_{\pi,\mu}^{\mathrm{AR}}[\tau]= \prod_{t=0}^{T} \left[ \prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t}) \right] \mu(o_{t+1},r_t\mid h_t,a_t) =\exp(-\beta H_0^{\mathrm{AR}}[\tau])
```

Sections 3.1 to 3.3 discuss resampling $o_{t+1}$ and $r_t$ without changing the token-generation measure itself. In the LLM case, $a_t$ only needs to be understood as an autoregressive token sequence:
```math
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
```

The return remains a statistical observable on the path:
```math
G[\tau_b]=\sum_{t=0}^{T}\gamma^t r_{b,t} \quad \text{or} \quad G[\tau_b] = \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{b,i,t}
```

A group of paths sampled for the same input can be viewed as a path distribution under the original or rollout-augmented process, including local perturbations: $\tau_b\sim P_{\pi,\mu}[\tau],~ b=1,\ldots,K_s$, where $b$ is the sample index. The within-group return mean and variance are
```math
\bar G=\frac{1}{K_s}\sum_{b=1}^{K_s}G[\tau_b],\quad \sigma_G^2=\frac{1}{K_s}\sum_{b=1}^{K_s}(G[\tau_b]-\bar G)^2
```

Here $\bar G$ is the mean return observable within the group, and $\sigma_G^2$ is the fluctuation strength of the return observable.

In Section 3.3, $\beta_k$ directly serves as the inverse-temperature parameter of simulated annealing in round $k$. To align this with the thermodynamic picture in Section 2, we can treat $\beta_k$ as the product of the inverse temperature $\beta$ and the local resampling strength $\alpha_{\,\mathrm{res}}^{(k)}$ in round $k$:
```math
\beta_k=\beta\alpha_{\mathrm{res}}^{(k)}
```

Here $\alpha_{\mathrm{res}}^{(k)}$ is the strength of the local observation / reward perturbation potential. For brevity, the text below still writes it as $\beta_k$. Given an original rollout path $\tau$, Gaussian resampling gives the path $\tau^k=(a_0,o_1^{(k)},r_0^{(k)},\ldots,a_T,o_{T+1}^{(k)},r_T^{(k)})$. If this is an LLM autoregressive path, then each $a_t$ remains the same token sequence, and resampling acts only on $o_{t+1}$, $r_t$, or their continuous representations. Gaussian resampling of observations corresponds to a quadratic Hamiltonian at each local time $t$. With independent Gaussian perturbations of the same scale, write
```math
H_o(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{(o_{t+1}^{(k)}-o_{t+1})^2}{2\sigma_o^2}
```

If we use an observation-noise covariance matrix $\Sigma_o$, then

```math
H_o(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{1}{2}(o_{t+1}^{(k)}-o_{t+1})^\top\Sigma_o^{-1}(o_{t+1}^{(k)}-o_{t+1})
```

If we also add Gaussian perturbations directly to rewards, we can add a reward-perturbation Hamiltonian:
```math
H_r(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{(r_t^{(k)}-r_t)^2}{2\sigma_r^2}
```

The resampling Hamiltonian then takes the form
```math
H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)=H_o(\tau^{(k)}\mid\tau)+H_r(\tau^{(k)}\mid\tau)
```

If reward is not noised directly but recomputed from the perturbed observation, we can keep only $H_o$ and set $r_t^{(k)}=R(o_{t+1}^{(k)},a_t,h_t)$.

The joint weight of the original path and the resampled path is therefore
```math
P_k(\tau,\tau^{(k)})\propto \exp\left(-\beta H_0[\tau]-\beta_k H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)\right)
```

So the sampling-augmented action is

```math
S_{\mathrm{augmented}}^{(k)}[\tau,\tau^{(k)}]=\beta H_0[\tau]+\beta_k H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)
```

This gives the following statistical-mechanical interpretation:
- the base Hamiltonian $H_0[\tau]$ weights the original path, and observation and reward resampling near this path produces thermal fluctuations through the local perturbation Hamiltonian $H_{\mathrm{resampled}}$
- $\beta_k$ controls the strength of local fluctuations. Small $\beta_k$ corresponds to local high temperature, so the resampled path has larger thermal fluctuations near the original path. Large $\beta_k$ later corresponds to local low temperature, so the resampled path shrinks toward the original observations and rewards

### 3.5. Related Work

- $(o,r)$ resampling
  - Research direction: Gaussian noise proposal / noisy environment augmentation / observation noise / reward noise
  - Directly related:
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
    - [[arXiv:2106.11420] Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/abs/2106.11420)
    - [[arXiv:1810.01032] Reinforcement Learning with Perturbed Rewards](https://arxiv.org/abs/1810.01032)
    - [[PMLR 2020] Deep Reinforcement Learning with Robust and Smooth Policy](https://proceedings.mlr.press/v119/shen20b.html)
  - Same direction:
    - [[arXiv:2310.00344] HarmonyDream: Task Harmonization Inside World Models](https://arxiv.org/abs/2310.00344)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)

- Within-group statistics
  - Research direction: GRPO / GSPO
  - Directly related:
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
  - Same direction:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- Proposal expansion
  - Research direction: SMC policy optimization / CEM / iCEM
  - Directly related:
    - [[arXiv:2402.07963] SPO: Sequential Monte Carlo Policy Optimisation](https://arxiv.org/abs/2402.07963)
    - [[arXiv:2505.16732] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](https://arxiv.org/abs/2505.16732)
    - [[arXiv:2008.06389] Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/abs/2008.06389)
    - [[arXiv:2112.07746] CEM-GD: Cross-Entropy Method with Gradient Descent Planner for Model-Based Reinforcement Learning](https://arxiv.org/abs/2112.07746)
  - Same direction:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)

- Connecting the original RL estimator to PPO / GRPO / GSPO
  - Research direction: PPO-family + noisy rollout augmentation / group-level RL
  - Directly related:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
  - Same direction:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)


---

## 4. Path Sampling Based on the Path Integral

Section 3 discussed local Gaussian perturbations of the observation $o_{t+1}$ and reward $r_t$ while keeping the original RL updater unchanged, thereby expanding the feedback space of rollout. This method still augments paths near those generated by the original policy. It mainly changes environment feedback, rather than directly searching for better paths across the whole path space.

This section changes the view. Starting from the path-integral representation, treat the whole trajectory $\tau$ itself as the sampling object. Path quality no longer depends only on single-step rewards or local perturbations. The whole path action and target observable decide it together. Intuitively, we want sampled paths not only to complete the task, but also to satisfy constraints such as KL, length, format, and resources. Such paths should correspond to lower effective action, or lie in more stable low-energy regions. This is the core reason for introducing the path integral.

To do this, add a source term to the original action $S_0[\tau]$ and organize rewards, penalties, and constraints into the target observable $F[\tau]$. This gives a Gibbs distribution with a source term. The distribution biases path sampling toward high-return and low-penalty regions. We then use sampling methods such as MCMC or Langevin, not to treat $a,o,r$ separately as states for local random walks, but to search for low-action paths in the whole path space. MCMC lets a path trapped in a local low point still have a chance to jump out. Langevin combines action descent and random perturbation on continuous hidden paths. A stable high-quality path should belong to a path basin that returns to a task-completing region after perturbation, rather than drifting away from the target.

### 4.1. Review: The Path-Integral Representation of RL

Section 2 gave the path-integral representation of RL:
```math
J(\pi)=\int \mathcal{D}\tau ~ e^{-S_{\pi,\mu}[\tau]} G[\tau]
```
It has autoregressive and non-autoregressive forms. The autoregressive form corresponds to next-token prediction in LLMs, where multiple new tokens form a new action $a_t$. The non-autoregressive form corresponds to traditional next-action prediction, where the model directly generates action $a_t$. The action, measure, and observable in the two forms are as follows.
- Autoregressive:
```math
\begin{aligned}
S^{AR}_{\pi,\mu}[\tau] &= - \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \left( \log \pi(a_i|h_{i,t}) + \delta_{i,L_{t+1}-1} \log \mu(o_{t+1},r_t | h_t, a_t) \right) \\
\mathcal{D}\tau &= \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \equiv \prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right] do_{t+1}\,dr_t\\
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t}
\end{aligned}
```

- Non-autoregressive:
```math
\begin{aligned}
S_{\pi,\mu}[\tau] &= - \sum_{t=0}^T \left[ \log \pi(a_t|h_t) + \log \mu(o_{t+1},r_t | h_t, a_t) \right] \\
\mathcal{D}\tau &= \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \\
G[\tau] &= \sum_{t=0}^T \gamma^t r_{t}
\end{aligned}
```

Under this representation, reinforcement learning has the following physical interpretation:
- $a_t,o_t,r_t$: degrees of freedom of fields on a one-dimensional time path. The fields here are scalar fields, namely an action field, an observation field, and a return field
- $\mathcal{D}\tau$: the measure over fields on all lattice sites
- $G[\tau]$: the physical observable
- $e^{-S_{\pi,\mu}[\tau]}$: the path-integral weight, where $S_{\pi,\mu}$ is the action
- History-based RL: a complex one-dimensional single-particle system with three degrees of freedom, $a,o,r$. Its complexity comes from long-range coupling based on the historical trajectory $h_t$. The policy $\pi$ and environment measure $\mu$ determine the coupling.
- Equivalent original Hamiltonian: $S_{\pi,\mu}[\tau] = \beta H_{\pi,\mu}[\tau] \equiv S_0[\tau] = \beta H_0[\tau],~ H_{\pi,\mu}[\tau] = - \log P_{\pi,\mu}$


### 4.2 Source Term & Gibbs Distribution
Introduce the field-theory source term to build the physical-picture correspondence for RL. The path integral with a source term is
```math
Z[\eta] = \int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]}
```
Differentiating with respect to the source coefficient gives
```math
\begin{aligned}
\frac{\partial}{\partial \eta} \log Z[\eta] &= \frac{1}{Z[\eta]} \int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} F[\tau] = \mathbb{E}_\eta [F[\tau]] \\
\frac{\partial^2}{\partial^2 \eta} \log Z[\eta] &= \text{Var}_\eta [F[\tau]]
\end{aligned}
```
When $\eta = 0$, this gives the observable expectation and variance under the original action. In RL, we can use the observable as the source term and further introduce penalty terms, such as length penalties and KL penalties:
```math
F[\tau; \lambda_G, \lambda_N, \lambda_{KL}] = \lambda_G G[\tau] - \lambda_N N[\tau] - \lambda_{KL} K[\tau]
```
For the KL penalty, we can choose
- strict KL divergence, using the full distribution:
```math
K[\tau] = \sum_{t=0}^T \sum_{i=L_{t}}^{L_{t+1}-1} \int d\tilde{a}_i ~ \pi(\tilde{a}_i|h_{i,t}) \log \frac{\pi(\tilde{a}_i|h_{i,t})}{\pi_\mathrm{ref}(\tilde{a}_i|h_{i,t})}
```
- sampled KL divergence, using only sampled actions:
```math
K[\tau] = \sum_{t=0}^T \sum_{i=L_{t}}^{L_{t+1}-1} \log \frac{\pi(a_i|h_{i,t})}{\pi_\mathrm{ref}(a_i|h_{i,t})}
```

The probability distribution of trajectory $\tau$ under $\eta$ is
```math
q_{\eta}(\tau) = \frac{1}{Z[\eta]} \exp\left( - S_0[\tau] + \eta F[\tau] \right)
```

Using the Hamiltonian definition also gives the effective Hamiltonian $H_{\eta}[\tau] = H_0[\tau] - \frac{\eta}{\beta}F[\tau]$, so
```math
q_\eta[\tau] \propto \exp\left( - \beta H_0[\tau] + \eta F[\tau] \right)
```

This distribution corresponds to the Gibbs distribution in statistical mechanics. Penalty terms can be viewed as equivalent potentials such as chemical potentials. The Gibbs distribution is a biased distribution: the source term biases the path distribution toward high return and low penalty, which matches the RL objective. To return to the original distribution, use reweighting:
```math
\begin{aligned}
& \mathbb{E}_{\eta=0} \left[F[\tau] \right] = \frac{\int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} e^{- \eta F[\tau]} F[\tau]}{\int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} e^{- \eta F[\tau]}}  \\
=~& \frac{\mathbb{E}_{\eta} \left[F[\tau]e^{- \eta F[\tau]} \right]}{\mathbb{E}_{\eta} \left[e^{- \eta F[\tau]} \right]} = \frac{\sum_{b}F[\tau_b]e^{-\eta F[\tau_b]}}{\sum_b e^{-\eta F[\tau_b]}}
\end{aligned}
```
Alternatively, use extrapolation as $\eta \rightarrow 0$.

### 4.3. MCMC Path Sampling

Given the current path $\tau$, the MCMC candidate path should not be understood as an arbitrary perturbation of a complete generated path. Because a history-based RL path has causal structure, later observations and returns depend on earlier histories and actions. If we directly modify an action but keep later observations and rewards, the second half may no longer be legal feedback under the new action. Therefore a legal proposal must generate the candidate path $\tau'$.

The proposal can perturb only the action prefix, or it can perturb actions, allowed observations, and soft returns together. Perturbing only observations or rewards can also work, but if these fields matter little for the model's later decisions, the model may keep generating the same actions and the path branch will not change much. A more general candidate-path proposal can therefore be written as a joint form:
```math
\begin{aligned}
& (c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}}) \sim q_{\mathrm{prop}}(c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}}\mid \tau) \\
& \tau' = \mathrm{Rollout}(c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}})
\end{aligned}
```

Here $`c'_{a}`$ denotes the perturbed action prefix. It can be a token prefix, hidden prefix, tool-call prefix, or agent-step prefix. $`o'_{\mathrm{allow}}`$ denotes observation fields that **allow perturbation**. $`r'_{\mathrm{soft}}`$ denotes perturbable returns or scores, such as soft scores, verifier scores, or reward-model scores. If we perturb only actions, then fix $`o'_{\mathrm{allow}}=o_{\mathrm{allow}}`$ and $`r'_{\mathrm{soft}}=r_{\mathrm{soft}}`$. If we do joint perturbation, sample all three together.

In the LLM autoregressive case, the action prefix can be written as $`c_{a,i,t} = (h_t,a_{L_t},a_{L_t+1},\cdots,a_i)`$, and the perturbed prefix is $`c'_{a,i,t} \sim q_a(c'_{a,i,t}\mid c_{a,i,t})`$. Allowed observations and rewards can be written in mask form:
```math
\begin{aligned}
o'_{t+1} &= \mathcal E_o(o_{t+1};m_o,\xi_o) \\
r'_t &= \mathcal E_r(r_t;m_r,\xi_r)
\end{aligned}
```
Here $\mathcal{E}_o ,~ \mathcal{E}_r$ are random perturbation operators constrained by masks. They can be Gaussian perturbations on certain numeric values or field-level random edits. The key point is that they act only on allowed observation fields or soft scores. The masks $m_o,m_r$ open only fields that allow perturbation. Numeric return values, confidence, soft scores, metadata, and auxiliary fields that do not affect factual semantics can all serve as part of the proposal. Hard observations, such as core tool-call return values, code execution results, judge results, and database query results, cannot be perturbed at will; otherwise the candidate path breaks the causal structure.

After obtaining the candidate path, compare the current path and candidate path with the action including the source term. The Metropolis-Hastings acceptance rate is
```math
\begin{aligned}
A_k(\tau\rightarrow\tau') = \min\left( 1, \exp\left[-(S_\eta[\tau']-S_\eta[\tau])\right] \frac{q_{\mathrm{prop}}(\tau\mid\tau')} {q_{\mathrm{prop}}(\tau'\mid\tau)} \right)
\end{aligned}
```

If the candidate path has lower effective action, it is more likely to be accepted. Even if the candidate path is temporarily worse, a nonzero acceptance rate still gives it a chance to remain. This mechanism lets path sampling jump out of local low points instead of staying near the current path.

### 4.4. Langevin Path Sampling

Langevin requires gradients of the action, so it fits continuous degrees of freedom. For LLMs, discrete tokens cannot receive direct Langevin updates, but hidden states can serve as continuous action variables. For example:
```math
\begin{aligned}
\tau_z &= [o_0,z_0,r_0,o_1,\cdots,o_T,z_T,r_T,o_{T+1}] \\
z_t &= [z_{L_t},\cdots,z_{L_{t+1}-1}]
\end{aligned}
```

Here again, we should not arbitrarily change the full hidden path. A more reasonable method selects a hidden prefix, applies Langevin updates to that prefix, and then continues decoding or rollout from the updated prefix to obtain the later path again. Let $c_z^{(\ell)} = [h_t,z_{L_t}^{(\ell)},\cdots,z_i^{(\ell)}]$. The Langevin update is
```math
z_{L_t:i}^{(\ell+1)} = z_{L_t:i}^{(\ell)} - \epsilon\nabla_{z_{L_t:i}^{(\ell)}}S_\eta[ \mathrm{Rollout}(c_z^{(\ell)},o_{\mathrm{allow}}^{(\ell)},r_{\mathrm{soft}}^{(\ell)}) ] + \sqrt{2\epsilon}\xi_\ell, \quad \xi_\ell \sim \mathcal N(0,I)
```
where $\ell$ denotes the Langevin update step. If we also perturb allowed observations and soft returns, write
```math
o_{\mathrm{allow}}^{(\ell+1)} = \mathcal E_o(o_{\mathrm{allow}}^{(\ell)};m_o,\xi_{o,k}),\quad r_{\mathrm{soft}}^{(\ell+1)} = \mathcal E_r(r_{\mathrm{soft}}^{(\ell)};m_r,\xi_{r,k})
```

After the update, the new candidate prefix and feedback are
```math
c_z^{(\ell+1)} = [h_t,z_{L_t}^{(\ell+1)},\cdots,z_i^{(\ell+1)}]
```

Then continue rollout:
```math
\tau_z^{(\ell+1)} = \mathrm{Rollout} (c_z^{(\ell+1)},o_{\mathrm{allow}}^{(\ell+1)},r_{\mathrm{soft}}^{(\ell+1)})
```

In Langevin, the gradient term pulls the path toward lower action, and the random term preserves thermal fluctuations. The goal is not to let the path drift away from the task, but to perturb the action prefix and allowed feedback while still letting later rollout return to a task-completing, low-penalty, low-action region. A stable good path is not an isolated point, but a path basin; after a small perturbation, later generation can still return to a task-completing trajectory.


### 4.5. Related Work

- Path-integral RL
  - Research direction: Path Integral Control / ${PI}^2$
  - Directly related:
    - [[PMLR 2010] Learning Policy Improvements with Path Integrals](https://proceedings.mlr.press/v9/theodorou10a.html)
    - [[JMLR 2010] A Generalized Path Integral Control Approach to Reinforcement Learning](https://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf)
  - Same direction:
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)

- KL penalty / control cost
  - Research direction: KL control / linearly-solvable MDP
  - Directly related:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)
  - Same direction:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

- RL as inference
  - Research direction: maximum entropy RL / control as inference / SAC
  - Directly related:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  - Same direction:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)

- $e^{-\beta H}$ sampling
  - Research direction: EBM / Gibbs / MCMC / Langevin
  - Directly related:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - Same direction:
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

- Langevin on hidden latent variables
  - Research direction: latent EBM / energy-based text generation / continuous relaxation
  - Directly related:
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
    - [[ICML 2021] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://proceedings.mlr.press/v139/pang21a.html)
  - Same direction:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2511.07124] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought](https://arxiv.org/abs/2511.07124)

- LLM reasoning sampling
  - Research direction: MCMC-inspired reasoning / constrained sampling
  - Directly related:
    - [[arXiv:2506.05754] Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - Same direction:
    - [[arXiv:2510.14901] Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)

---

## 5. Relationship Between Sections 3 and 4

---

# Appendix A: Renormalization Along the Time Direction

Because this is a one-dimensional time path integral, renormalization mainly proceeds along the time direction. In the non-autoregressive version, let $\ell$ denote the macro time-block index, and merge every $b$ microscopic actions into one macro block:

```math
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1})
```

After multilayer compression:

```math
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}
```

At the path-integral level, the mapping from microscopic paths to macroscopic paths is $\bar\tau=\mathcal C(\tau)$. If we discuss the original path distribution, the macroscopic base action comes from integrating out microscopic degrees of freedom:

```math
\exp(-S_0[\bar\tau]) = \int_{\mathcal C(\tau)=\bar\tau} \exp(-S_0[\tau]) \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
```


The LLM autoregressive version has two time structures. The outer layer is the environment / agent step $t$, and the inner layer is the token position $i$. The token segment of the external step $t$ is
```math
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
```

So the most basic LLM time coarse-graining has already integrated token micro-steps into the segment-level action $a_t$:
```math
\prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i = \prod_{t=0}^{T}da_t
```

If we still want to compress token blocks inside one external step, let the $\ell$-th token block inside the external step $t$ be $B_{t,\ell}=C_\phi(a_{L_t+\ell b},a_{L_t+\ell b+1},\ldots,a_{L_t+(\ell+1)b-1})$. Then the coarse-graining of the LLM autoregressive base action is
```math
\exp(-S_0^{\mathrm{AR}}[\bar\tau]) = \int_{\mathcal C(\tau)=\bar\tau} \exp(-S_0^{\mathrm{AR}}[\tau]) \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i\,do_{t+1}\,dr_t
```

where
```math
S_0^{\mathrm{AR}}[\tau] =- \sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t}) -\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
```

If the effective singular-value spectrum inside the compressed block decays fast, truncation is safe. If the spectrum is nearly flat, hard truncation loses a large amount of information:
```math
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ truncation causes major information loss}
```

**Note**: this "renormalization" only discusses token-level compression. It does not describe true "semantic-information renormalization." True renormalization means that after coarse-graining information, some semantic invariant appears at a certain scale. The author believes current token-compression techniques choose and compress information through trainable weights. This is not true renormalization, because such compression cannot guarantee, and has not shown evidence for, any invariant or RG fixed point.
