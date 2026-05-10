# From history-based RL to resampling and the physical picture of RL

## 0. Objective

What is discussed here is a **physical-picture interpretation**, not a strict rewriting of RL as field theory.

The intent of this note is to discuss physical methods for expanding sampled paths. There are two main lines:

- **Resampling**: Under the original cumulative-return definition of RL, resample / re-estimate the observation $o_{t+1}$ and reward $r_t$, in order to improve rollouts, return estimation, advantage estimation, and PPO / GRPO / GSPO updates.
- **Path integral**: Under the path-integral picture, introduce an effective Hamiltonian, Boltzmann weights, Gibbs sampling, MCMC / Langevin, and inverse-temperature annealing, in order to directly expand sampling in path space and search for low-Hamiltonian paths.

Their common basis is:
- history-based RL can be written as a path integral along a one-dimensional time direction
- the return $G[\tau]$ in the original expectation is an observable insertion on the path
- in the thermodynamic analogy, the action $S$ is dimensionless, the Hamiltonian $H$ has the dimension of energy, and they satisfy $S=\beta H$

---

## 1. Original trajectory integral for history-based RL

Start from the most primitive form. The interaction history is:

\[
\begin{equation}
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\ldots,a_{t-1},r_{t-1},o_t).
\tag{1}
\end{equation}
\]

The policy is $\pi(a_t\mid h_t)$, and the environmental conditional density is $\mu(o_{t+1},r_t\mid a_t,h_t)$. Over a finite time horizon $T$, the expected return of the full trajectory can be written as:

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right).
\tag{2}
\end{equation}
\]

The whole path is written as:

\[
\begin{equation}
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T).
\tag{3}
\end{equation}
\]

The path return is written as:

\[
\begin{equation}
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s.
\tag{4}
\end{equation}
\]

Thus, the original RL objective is a weighted integral over all possible paths. The weight of each path is jointly given by the policy and the environment, and the value of each path is given by the discounted return $G[\tau]$.

### 1.1. Dirac-delta degeneration in a deterministic environment

If the environment is deterministic, then after $a_t,h_t$ are given, the next observation and reward are given by deterministic functions:

\[
\begin{equation}
o_{t+1}=O(a_t,h_t),\qquad r_t=R(o_{t+1},a_t,h_t).
\tag{5}
\end{equation}
\]

The environmental conditional density degenerates into Dirac deltas:

\[
\begin{equation}
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t)).
\tag{6}
\end{equation}
\]

Substituting this back into the original trajectory integral gives:

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right).
\tag{7}
\end{equation}
\]

Using the basic integral property of the Dirac delta:

\[
\begin{equation}
\int \delta(x-x_0)f(x)\,dx=f(x_0).
\tag{8}
\end{equation}
\]

After the environmental part collapses, only action sampling remains:

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right].
\tag{9}
\end{equation}
\]

This means:
- a deterministic environment no longer provides path branching; path branching comes only from policy sampling
- if the policy is also deterministic, the entire path collapses into a single path

### 1.2. Finite horizon and finite reward through reward clipping

If the maximum path length is restricted by $T\le T_{\max}$, the path integral is performed only over a finite time interval:

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right).
\tag{10}
\end{equation}
\]

If rewards are clipped:

\[
\begin{equation}
\bar r_t=\operatorname{clip}(r_t,-r_{\max},r_{\max}),\qquad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t.
\tag{11}
\end{equation}
\]

The meaning is:
- a finite horizon is a time cutoff, and finite reward is the boundedness of the return observable

### 1.3. Path sampling and hidden actions in LLM RL

For LLMs, the action $a_t$ can be a token, a full answer segment, a tool call, a code patch, or an agent step. If the action is a token, the model forward pass gives logits, and the token is then obtained through sampling:

\[
\begin{equation}
z_t^{\mathrm{logit}}=f_\theta(h_t),\qquad \pi_{\theta,T_{\mathrm{dec}}}(a_t\mid h_t)=\frac{\exp(z^{\mathrm{logit}}_{t,a_t}/T_{\mathrm{dec}})}{\sum_{a'}\exp(z^{\mathrm{logit}}_{t,a'}/T_{\mathrm{dec}})}.
\tag{12}
\end{equation}
\]

If one wants to perform path sampling in a continuous space, the hidden state obtained after prefill can be used as a continuous action representation:

\[
\begin{equation}
a_t=z_t,\qquad z_t=\operatorname{hidden}_\theta(\operatorname{prefill}(a_{\le t})).
\tag{13}
\end{equation}
\]

Therefore, the continuous-action path can still be written using the same path symbol:

\[
\begin{equation}
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T).
\tag{14}
\end{equation}
\]

No new path variable is introduced here; the discrete token action $a_t$ is merely replaced by the continuous hidden action $z_t$. When hidden actions are not emphasized, we still write $a_t$.

The $i$-th sample is:

\[
\begin{equation}
\tau_i=(a_{i,0},o_{i,1},r_{i,0},a_{i,1},o_{i,2},r_{i,1},\ldots,a_{i,T_i},o_{i,T_i+1},r_{i,T_i}).
\tag{15}
\end{equation}
\]

If hidden states are used as actions, then:

\[
\begin{equation}
a_{i,t}=z_{i,t},\qquad z_{i,t}=\operatorname{hidden}_\theta(\operatorname{prefill}(a_{i,\le t})).
\tag{16}
\end{equation}
\]

Here, $i$ is the sample index, and $t$ is the time step or token position on the path.

---

## 2. Physical-picture interpretation of original history-based RL

### 2.1. From trajectory integral to path integral

Write the product of original densities as a path density:

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{17}
\end{equation}
\]

Then the original objective can be written in path-integral form:

\[
\begin{equation}
J(\pi)=\int P_{\pi,\mu}[\tau]G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{18}
\end{equation}
\]

The physical picture is:
- $\tau$ is a worldline, $P_{\pi,\mu}[\tau]$ is the path weight, $G[\tau]$ is the path-return functional, and $J(\pi)$ is the weighted average over all paths

### 2.2. Base action, base Hamiltonian, and Boltzmann weight

Starting from the path density:

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{19}
\end{equation}
\]

Taking the negative logarithm of the path density gives the base action:

\[
\begin{equation}
S_{\pi,\mu}[\tau]=-\log P_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{20}
\end{equation}
\]

In the thermodynamic analogy, the Boltzmann weight is written as $\exp(-\beta H)$. Therefore, the relation between the base action and the base Hamiltonian is:

\[
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau].
\tag{21}
\end{equation}
\]

Thus, the original path density can be written as:

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\exp(-S_{\pi,\mu}[\tau])=\exp(-\beta H_0[\tau]).
\tag{22}
\end{equation}
\]

The original RL objective becomes:

\[
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{23}
\end{equation}
\]

Here, $G[\tau]$ is an observable insertion, not part of the Hamiltonian.

### 2.3. A single-body complex system along a one-dimensional time direction

The path variable can be written as:

\[
\begin{equation}
x_t=(a_t,o_{t+1},r_t),\qquad \tau=(x_0,x_1,\ldots,x_T).
\tag{24}
\end{equation}
\]

It is a path system along a one-dimensional time direction. The complexity comes from history coupling, because the policy and environment at every step both depend on the full history:

\[
\begin{equation}
\pi(a_t\mid h_t),\qquad \mu(o_{t+1},r_t\mid a_t,h_t).
\tag{25}
\end{equation}
\]

The corresponding base action terms are:

\[
\begin{equation}
S_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{26}
\end{equation}
\]

The physical picture is:
- the RL path integral is a complex single-body system along a one-dimensional time direction; its internal degrees of freedom are action, observation, and reward, and its complexity comes from the long-range temporal coupling of these degrees of freedom through the history $h_t$

### 2.4. Discount factor and Laplace regularization

If the continuous-time return is written as:

\[
\begin{equation}
G[\tau]=\int_0^\infty r(t)\,dt,
\tag{27}
\end{equation}
\]

it may diverge. After adding exponential decay:

\[
\begin{equation}
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt.
\tag{28}
\end{equation}
\]

In discrete time:

\[
\begin{equation}
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t,
\tag{29}
\end{equation}
\]

Let the time-step length be $\Delta t$. The correspondence is:

\[
\begin{equation}
\gamma=e^{-\lambda\Delta t},\qquad \gamma^t=e^{-\lambda t\Delta t}.
\tag{30}
\end{equation}
\]

If the reward is bounded:

\[
\begin{equation}
|r_t|\le r_{\max},
\tag{31}
\end{equation}
\]

then the discounted return is bounded:

\[
\begin{equation}
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}.
\tag{32}
\end{equation}
\]

The physical picture is:
- the discount factor is Laplace damping along the time direction; it compresses the infinite future into a finite effective contribution

---

## 3. Observation and reward resampling under the original RL definition

Starting from the original RL definition of cumulative return, the goal is to expand the local sampling space of the observation $o_{t+1}$ and reward $r_t$ without changing the policy updater, so that the model can better explore paths. More complex methods can use repeated sampling from the real environment, world models, reward models, verifiers, SMC, or CEM proposals. However, this section does not discuss these complex repeated-sampling methods, and uses only the most basic Gaussian-noise proposal.

### 3.1. Gaussian-noise resampling / re-estimation of $o$ and $r$

Let $m=1,\ldots,M$ denote the $m$-th resampling under the same condition $a_t,h_t,o_{t+1},r_t$. If simulated annealing is used, let $k=0,1,\ldots,K_{\mathrm{ann}}$ denote the external annealing iteration step, not the internal path time step $t$. The $k$-th round uses inverse temperature $\beta_k>0$. Let $\sigma_o>0$ be the base noise scale of the observation, and let $\sigma_r>0$ be the base noise scale of the reward. $\sigma_o,\sigma_r$ are manually specified proposal noise intensities, not variances obtained from within-group statistics.

Start with a one-dimensional scalar variable. Given the current value $x_0$ and the $k$-th round noise width $s_k>0$, the most basic one-dimensional Gaussian proposal is defined as:

\[
\begin{equation}
q_k(x'\mid x_0)=\frac{1}{\sqrt{2\pi}s_k}\exp\left(-\frac{(x'-x_0)^2}{2s_k^2}\right).
\tag{33}
\end{equation}
\]

To let the inverse temperature control the width of the proposal, define:

\[
\begin{equation}
s_k=\frac{\sigma}{\sqrt{\beta_k}},
\tag{34}
\end{equation}
\]

where $\sigma>0$ is the base noise scale. Substituting equation (34) into equation (33) gives:

\[
\begin{equation}
q_k(x'\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x'-x_0)^2}{2\sigma^2}\right).
\tag{35}
\end{equation}
\]

The density of the standard Gaussian noise $\xi^{(m)}$ is:

\[
\begin{equation}
p(\xi^{(m)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi^{(m)})^2}{2}\right).
\tag{36}
\end{equation}
\]

Let the $m$-th resampled value be:

\[
\begin{equation}
x^{(m,k)}=x_0+\frac{\sigma}{\sqrt{\beta_k}}\xi^{(m)}.
\tag{37}
\end{equation}
\]

From equation (37), we get:

\[
\begin{equation}
\xi^{(m)}=\frac{\sqrt{\beta_k}}{\sigma}(x^{(m,k)}-x_0).
\tag{38}
\end{equation}
\]

Therefore, the density of $x^{(m,k)}$ is exactly the Gaussian proposal centered at $x_0$ with width $\sigma/\sqrt{\beta_k}$:

\[
\begin{equation}
q_k(x^{(m,k)}\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x^{(m,k)}-x_0)^2}{2\sigma^2}\right).
\tag{39}
\end{equation}
\]

Thus, a small $\beta_k$ corresponds to a wider proposal, while a large $\beta_k$ corresponds to a narrower proposal. This is the simulated-annealing control method in this section.

If $o_{t+1}$ is a one-dimensional continuous observation, take $x_0=o_{t+1}$ and $\sigma=\sigma_o$. This gives the observation resampling equation:

\[
\begin{equation}
o_{t+1}^{(m,k)}=o_{t+1}+\frac{\sigma_o}{\sqrt{\beta_k}}\xi_o^{(m,t)}.
\tag{40}
\end{equation}
\]

Here, $\xi_o^{(m,t)}$ is the standard Gaussian noise at the $m$-th resampling and path time step $t$:

\[
\begin{equation}
p(\xi_o^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_o^{(m,t)})^2}{2}\right).
\tag{41}
\end{equation}
\]

The corresponding observation proposal is:

\[
\begin{equation}
q_k(o'_{t+1}\mid o_{t+1})=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_o}\exp\left(-\frac{\beta_k(o'_{t+1}-o_{t+1})^2}{2\sigma_o^2}\right).
\tag{42}
\end{equation}
\]

If $o_{t+1}$ is a multidimensional vector or a text embedding, the same one-dimensional Gaussian perturbation can be used for each coordinate. No correlated noise matrix is introduced here; if correlated noise is needed in the future, the meaning of the matrix must first be defined before writing it down.

If Gaussian perturbation is applied directly to the scalar reward, take $x_0=r_t$ and $\sigma=\sigma_r$. This gives:

\[
\begin{equation}
r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\xi_r^{(m,t)}.
\tag{43}
\end{equation}
\]

where:

\[
\begin{equation}
p(\xi_r^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_r^{(m,t)})^2}{2}\right).
\tag{44}
\end{equation}
\]

The corresponding reward proposal is:

\[
\begin{equation}
q_k(r'_t\mid r_t)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_r}\exp\left(-\frac{\beta_k(r'_t-r_t)^2}{2\sigma_r^2}\right).
\tag{45}
\end{equation}
\]

However, if one merely adds zero-centered Gaussian noise to $r_t$ and then takes a sample average, the average will return to a value near the original $r_t$. According to the original definition of the sample average:

\[
\begin{equation}
\bar r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\left(\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\right).
\tag{46}
\end{equation}
\]

When positive and negative Gaussian noises approximately cancel:

\[
\begin{equation}
\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\approx 0,
\tag{47}
\end{equation}
\]

then:

\[
\begin{equation}
\bar r_t^{(k)}\approx r_t.
\tag{48}
\end{equation}
\]

Therefore, directly adding noise to the reward is mainly a robustness perturbation. A more meaningful approach is to first Gaussian-resample the observation or the observation hidden representation, and then recompute the reward using the perturbed observation:

\[
\begin{equation}
r_t^{(m,k)}=R(o_{t+1}^{(m,k)},a_t,h_t).
\tag{49}
\end{equation}
\]

The corresponding local reward estimate is:

\[
\begin{equation}
\widehat r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}R(o_{t+1}^{(m,k)},a_t,h_t).
\tag{50}
\end{equation}
\]

For the $i$-th action path, the $m$-th resampled path is:

\[
\begin{equation}
\tau_i^{(m,k)}=(a_{i,0},o_{i,1}^{(m,k)},r_{i,0}^{(m,k)},\ldots,a_{i,T_i},o_{i,T_i+1}^{(m,k)},r_{i,T_i}^{(m,k)}).
\tag{51}
\end{equation}
\]

The return of the $m$-th resampled path is:

\[
\begin{equation}
G_i^{(m,k)}=\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{52}
\end{equation}
\]

Using the original definition of the sample average gives the resampled return estimate in the $k$-th round:

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}G_i^{(m,k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{53}
\end{equation}
\]

This is the core of this section: under the original cumulative-return definition of RL, use Gaussian-noise proposals to expand $o_{t+1}$, $r_t$, or their continuous representations, and then use sample averaging to estimate more stable returns or advantages.

### 3.2. Connection to PPO / GRPO / GSPO

GRPO / GSPO can be taken as refinements of this section. They naturally have within-group sample structures, and are therefore suitable for performing within-group statistics on the resampled returns. However, this method can also serve PPO, because PPO also only needs rollouts, rewards, advantages, and policy-update ratios.

Let $K_s$ denote the number of within-group samples in the $k$-th round. Let $i,j$ denote within-group sample indices, and let $t$ denote an internal path time step or token position. For the same input $x$, sample $K_s$ paths in the $k$-th round:

\[
\begin{equation}
\tau_1^{(k)},\tau_2^{(k)},\ldots,\tau_{K_s}^{(k)}\sim q_k(\tau\mid x).
\tag{54}
\end{equation}
\]

Each path uses Gaussian-noise resampling to obtain a return estimate:

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{55}
\end{equation}
\]

The within-group mean and variance are:

\[
\begin{equation}
\bar G^{(k)}=\frac{1}{K_s}\sum_{i=1}^{K_s}\widehat G_i^{(k)},\qquad (\sigma_G^{(k)})^2=\frac{1}{K_s}\sum_{i=1}^{K_s}(\widehat G_i^{(k)}-\bar G^{(k)})^2.
\tag{56}
\end{equation}
\]

The normalized advantage is:

\[
\begin{equation}
A_i^{(k)}=\frac{\widehat G_i^{(k)}-\bar G^{(k)}}{\sigma_G^{(k)}+\epsilon}.
\tag{57}
\end{equation}
\]

If one wants to perform local statistics on the observations and rewards themselves, one can write:

\[
\begin{equation}
\bar o_{t+1}^{(k)}=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}o_{i,t+1}^{(m,k)},\qquad \bar r_t^{(k)}=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}r_{i,t}^{(m,k)}.
\tag{58}
\end{equation}
\]

The corresponding variances are:

\[
\begin{equation}
(s_o^{(k)})^2=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}(o_{i,t+1}^{(m,k)}-\bar o_{t+1}^{(k)})^2,
\qquad
(s_r^{(k)})^2=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}(r_{i,t}^{(m,k)}-\bar r_t^{(k)})^2.
\tag{59}
\end{equation}
\]

Here, $s_o^{(k)}$ and $s_r^{(k)}$ are feedback widths obtained from within-group statistics, not the base noise scales $\sigma_o$ and $\sigma_r$ in the proposal. A smaller variance can be interpreted as the feedback estimate becoming more stable, or as the current policy entering a more stable local region.

GSPO can be viewed as taking the entire generated sequence as the sampling unit:

\[
\begin{equation}
\tau_i=(a_{i,0},o_{i,1},r_{i,0},\ldots,a_{i,T_i},o_{i,T_i+1},r_{i,T_i}).
\tag{60}
\end{equation}
\]

If model actions, tool returns, environmental observations, and reward text are all mixed in the same sequence, sequence-level sampling approximately samples $a$ and $o$ together in a mixed way. A more reasonable causal decomposition is:

\[
\begin{equation}
q(\tau)=q_a(a_{0:T})q_o(o_{1:T+1}\mid a_{0:T},h_{0:T})q_r(r_{0:T}\mid a_{0:T},o_{1:T+1},h_{0:T}).
\tag{61}
\end{equation}
\]

If hidden states are used as actions:

\[
\begin{equation}
q(\tau)=q_z(z_{0:T})q_o(o_{1:T+1}\mid z_{0:T},h_{0:T})q_r(r_{0:T}\mid z_{0:T},o_{1:T+1},h_{0:T}).
\tag{62}
\end{equation}
\]

The meaning is:
- expand the feedback space of observations and rewards
- the within-group statistics of GRPO / GSPO can more naturally estimate $G$, advantages, and the stability of $o$ and $r$
- PPO can also use the same $o,r$ resampling data; only the subsequent policy updater is different

### 3.3. Simulated annealing

In this section, simulated annealing is used to schedule the Gaussian-noise proposal. The inverse temperature $\beta_k$ in the $k$-th round controls the Gaussian-noise width through equations (40) and (43):

\[
\begin{equation}
\frac{\sigma_o}{\sqrt{\beta_k}},\qquad \frac{\sigma_r}{\sqrt{\beta_k}}.
\tag{63}
\end{equation}
\]

The inverse temperature increases:

\[
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}.
\tag{64}
\end{equation}
\]

A small $\beta_k$ corresponds to a wider Gaussian-noise proposal in the early stage, while a large $\beta_k$ corresponds to a narrower and more conservative feedback estimate in the later stage. Here, $\beta_k$ is a simulated-annealing scheduling parameter used to explain sampling width and feedback fluctuations; it is not the decoder temperature.

### 3.4. Statistical-mechanics interpretation

From the physical picture in Section 2, original history-based RL can already be viewed as a one-dimensional temporal path system. The policy and environment induce the original path distribution:

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{65}
\end{equation}
\]

In this section, the return is treated as a statistical observable on the path:

\[
\begin{equation}
G[\tau]=\sum_{t=0}^{T}\gamma^t r_t.
\tag{66}
\end{equation}
\]

A group of paths sampled for the same input can be viewed as a local ensemble under the original path distribution or its rollout augmentation:

\[
\begin{equation}
\tau_i\sim P_{\pi,\mu}[\tau],\qquad i=1,\ldots,K_s.
\tag{67}
\end{equation}
\]

The within-group return mean and variance are:

\[
\begin{equation}
\bar G=\frac{1}{K_s}\sum_{i=1}^{K_s}G[\tau_i],\qquad \sigma_G^2=\frac{1}{K_s}\sum_{i=1}^{K_s}(G[\tau_i]-\bar G)^2.
\tag{68}
\end{equation}
\]

Here, $\bar G$ describes the average return of this local ensemble, and $\sigma_G^2$ describes return fluctuations. Gaussian-noise resampling of $o_{t+1}$ and $r_t$ can be viewed as expanding this local ensemble, so that the advantage or return estimate does not depend only on the accidental result of a single rollout.

In this section, simulated annealing is first of all a sampling / estimation scheduling mechanism. In the early stage, a smaller $\beta_k$ is used, making the Gaussian proposal wider and allowing larger feedback fluctuations. Later, $\beta_k$ is gradually increased, causing the proposal to contract toward a more stable feedback estimate. In statistical-mechanics language, this is similar to a cooling process from high-temperature exploration to low-temperature stability. Here, the return is still only a statistical observable used to estimate advantages, rank samples, and update PPO / GRPO / GSPO.

### 3.5. Related research

- $(o,r)$ resampling
  - Corresponding literature direction: Gaussian noise proposal / noisy environment augmentation / observation noise / reward noise
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
  - Corresponding literature direction: GRPO / GSPO
  - Directly related:
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
  - Same direction:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- Proposal expansion
  - Corresponding literature direction: SMC policy optimization / CEM / iCEM
  - Directly related:
    - [[arXiv:2402.07963] SPO: Sequential Monte Carlo Policy Optimisation](https://arxiv.org/abs/2402.07963)
    - [[arXiv:2505.16732] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](https://arxiv.org/abs/2505.16732)
    - [[arXiv:2008.06389] Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/abs/2008.06389)
    - [[arXiv:2112.07746] CEM-GD: Cross-Entropy Method with Gradient Descent Planner for Model-Based Reinforcement Learning](https://arxiv.org/abs/2112.07746)
  - Same direction:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)

- Connecting the original RL estimator to PPO / GRPO / GSPO
  - Corresponding literature direction: PPO-family + noisy rollout augmentation / group-level RL
  - Directly related:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
  - Same direction:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)


---

## 4. Path integral / effective Hamiltonian / path sampling

### 4.1. External-field Hamiltonian: physical representation of RL rewards and penalty terms

This section starts from the base-Hamiltonian representation in Section 2. The base Hamiltonian $H_0[\tau]$ comes from the original path weight induced by the policy and environment. The external-field Hamiltonian is used to represent objective terms in RL, such as rewards, KL penalties, and path-length costs. The external-field Hamiltonian here is an objective structure, not a sampling method.

After introducing the return external field, the KL generalized chemical potential, and the path-length chemical potential, the external-field Hamiltonian is written as:

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_G G[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_N N[\tau].
\tag{69}
\end{equation}
\]

Here, $\lambda_G$ is the external-field strength that controls preference for high-return paths, $\lambda_{\mathrm{KL}}$ is the generalized chemical potential or KL pullback strength, and $\lambda_N$ is the chemical-potential cost of path length or number of interactions. If a clipped return was used earlier, $\bar G[\tau]$ can also be used here in place of $G[\tau]$.

The total Hamiltonian is:

\[
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]+H_{\mathrm{field}}[\tau].
\tag{70}
\end{equation}
\]

The corresponding effective action is:

\[
\begin{equation}
S_{\mathrm{eff}}[\tau]=\beta H_{\mathrm{eff}}[\tau]=\beta H_0[\tau]+\beta H_{\mathrm{field}}[\tau].
\tag{71}
\end{equation}
\]

After expansion:

\[
\begin{equation}
S_{\mathrm{eff}}[\tau]=S_{\pi,\mu}[\tau]-\beta \lambda_GG[\tau]+\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\beta\lambda_NN[\tau].
\tag{72}
\end{equation}
\]

The path KL term is:

\[
\begin{equation}
D_{\mathrm{KL}}[\tau]=\sum_{t=0}^{T}D_{\mathrm{KL}}\left(\pi(\cdot\mid h_t)\Vert \pi_{\mathrm{ref}}(\cdot\mid h_t)\right).
\tag{73}
\end{equation}
\]

The narrow-sense particle number or resource number can be written as $N[\tau]$, such as token count, step count, tool-call count, or interaction count. $\lambda_NN[\tau]$ corresponds to a resource chemical-potential cost.

If one directly uses:

\[
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]-\lambda_GG[\tau]+\lambda_NN[\tau],
\tag{74}
\end{equation}
\]

then when $\lambda_N$ is too large, low-Hamiltonian paths will tend to be overly short paths, which may prevent task completion. A safer way is to treat path length as a constraint interval:

\[
\begin{equation}
C_N[\tau]=\mu_+\max(0,N[\tau]-N_{\max})^2+\mu_-\max(0,N_{\min}-N[\tau])^2.
\tag{75}
\end{equation}
\]

The corresponding external-field Hamiltonian is:

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+C_N[\tau].
\tag{76}
\end{equation}
\]

The physical picture is:
- $N[\tau]$ is not something that should simply be made as small as possible; it is a resource constraint. The true low-Hamiltonian path should be successful and short, not short but failed

### 4.2. Boltzmann weight: obtaining path weights from the Hamiltonian

Given the effective Hamiltonian, the Boltzmann weight is:

\[
\begin{equation}
W_\beta[\tau]=\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{77}
\end{equation}
\]

Expanded into the form of the original path density:

\[
\begin{equation}
W_\beta[\tau]=P_{\pi,\mu}[\tau]\exp(\beta \lambda_GG[\tau]-\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]-\beta\lambda_NN[\tau]).
\tag{78}
\end{equation}
\]

Therefore, $\beta\lambda_G$ is the actual exponential strength of the reward tilt, $\beta\lambda_{\mathrm{KL}}$ is the actual exponential strength of the KL pullback, and $\beta\lambda_N$ is the actual exponential strength of the path-length penalty. $\beta$ is the inverse temperature in the thermodynamic analogy. In implementation, it can be treated as an annealing parameter, but it is not the decoder temperature.

### 4.3. Gibbs sampling: a sampling method based on Boltzmann weights

If one does not construct a new Gibbs sampling distribution, and only wants to write the original observable insertion in pure exponential form, one can first use a positive-valued return $\widetilde G[\tau]>0$. If $G[\tau]$ is allowed to take negative values, formally one can also write the negative sign as a complex phase, turning the original path integral into a complex-weight path integral and further considering complex Langevin methods. However, this would introduce unnecessary phase/sign problems and the complexity of complex stochastic processes. This note does not take that route; instead, it uses a positive-valued return $\widetilde G[\tau]>0$ to keep the weights real.

\[
\begin{equation}
\widetilde G[\tau]=G[\tau]+c,\qquad \widetilde G[\tau]>0.
\tag{79}
\end{equation}
\]

At this point:

\[
\begin{equation}
\exp(-\beta H_0[\tau])\widetilde G[\tau]=\exp(-\beta H_0[\tau]+\log\widetilde G[\tau]).
\tag{80}
\end{equation}
\]

Define the observable-equivalent action:

\[
\begin{equation}
S_{\mathrm{obs}}[\tau]=\beta H_0[\tau]-\log\widetilde G[\tau].
\tag{81}
\end{equation}
\]

Then:

\[
\begin{equation}
\widetilde J(\pi)=\int \exp(-S_{\mathrm{obs}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{82}
\end{equation}
\]

If the base path density is normalized, then the original expectation and the positive-valued expectation satisfy:

\[
\begin{equation}
J(\pi)=\widetilde J(\pi)-c.
\tag{83}
\end{equation}
\]

This route is only an exponentiation of the original observable insertion; it is not Gibbs sampling.

Gibbs sampling is a different matter: it constructs a new sampling distribution according to the Boltzmann weight:

\[
\begin{equation}
q_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}(\tau\mid\pi,\mu)=\frac{1}{Z_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}}\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{84}
\end{equation}
\]

The partition function is:

\[
\begin{equation}
Z_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}=\int \exp(-\beta H_{\mathrm{eff}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{85}
\end{equation}
\]

The original rollout distribution is:

\[
\begin{equation}
p_0(\tau\mid\pi,\mu)=P_{\pi,\mu}[\tau]=\exp(-\beta H_0[\tau]).
\tag{86}
\end{equation}
\]

The Gibbs-tilted distribution is:

\[
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}p_0(\tau\mid\pi,\mu)\exp(\beta \lambda_GG[\tau]-\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]-\beta\lambda_NN[\tau]).
\tag{87}
\end{equation}
\]

Only when the external fields are turned off does the Gibbs distribution degenerate back into the original sampling distribution:

\[
\begin{equation}
\lambda_G=0,\qquad \lambda_{\mathrm{KL}}=0,\qquad \lambda_N=0\quad\Longrightarrow\quad q(\tau\mid\pi,\mu)=p_0(\tau\mid\pi,\mu).
\tag{88}
\end{equation}
\]

Therefore, Gibbs sampling is not the original sampling. It is a reward-tilted ensemble introduced for path search and training-sample construction.

If we sample from the Gibbs distribution:

\[
\begin{equation}
\tau_i\sim q(\tau\mid\pi,\mu),\qquad i=1,\ldots,K,
\tag{89}
\end{equation}
\]

then the sample mean estimates the expectation under the Gibbs ensemble:

\[
\begin{equation}
\widehat{\mathbb E}_{q}[G]=\frac{1}{K}\sum_{i=1}^{K}G[\tau_i].
\tag{90}
\end{equation}
\]

This is not $J(\pi)$ under the original rollout distribution. The original expectation is:

\[
\begin{equation}
J(\pi)=\mathbb E_{p_0}[G[\tau]].
\tag{91}
\end{equation}
\]

The expectation under Gibbs sampling is:

\[
\begin{equation}
\mathbb E_q[G[\tau]]=\frac{\partial \log Z}{\partial(\beta \lambda_G)}.
\tag{92}
\end{equation}
\]

If one wants to infer the original $J(\pi)$ from Gibbs samples, importance reweighting is needed:

\[
\begin{equation}
w_i=\exp(-\beta \lambda_GG[\tau_i]+\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau_i]+\beta\lambda_NN[\tau_i]).
\tag{93}
\end{equation}
\]

Thus:

\[
\begin{equation}
\widehat J(\pi)=\frac{\sum_{i=1}^{K}w_iG[\tau_i]}{\sum_{i=1}^{K}w_i}.
\tag{94}
\end{equation}
\]

When the external fields are strong, the variance of the reverse weights will become large. Therefore, if the goal is to estimate the original $J(\pi)$, one should sample directly from $p_0$, or use a very small $\beta \lambda_G$ and perform extrapolation. If the goal is to expand exploration and generate high-value training samples, then the Gibbs sampling distribution $q$ can be used directly.

When $\beta \lambda_G$ is very small:

\[
\begin{equation}
\exp(\beta \lambda_GG[\tau])=1+\beta \lambda_GG[\tau]+O((\beta \lambda_G)^2).
\tag{95}
\end{equation}
\]

At this point, the Gibbs distribution is close to the original rollout distribution:

\[
\begin{equation}
q(\tau\mid\pi,\mu)\approx p_0(\tau\mid\pi,\mu).
\tag{96}
\end{equation}
\]

If sampling is performed under multiple small external-field strengths:

\[
\begin{equation}
g_j=\frac{1}{K_j}\sum_{i=1}^{K_j}G[\tau_i^{(j)}],\qquad \tau_i^{(j)}\sim q_{\beta \lambda_G^{(j)}}(\tau\mid\pi,\mu),
\tag{97}
\end{equation}
\]

then $g_j$ can be used to extrapolate to $\beta \lambda_G\to 0$, thereby approximating the original $J(\pi)$. The cost is increased resource expenditure, because sampling is required for every external-field strength.

### 4.4. MCMC path sampling

If simulated annealing is used, let $k=0,1,\ldots,K_{\mathrm{ann}}$ denote the annealing iteration step of the external sampler, and let $\beta_k$ be the inverse temperature of the $k$-th round. The target path distribution is:

\[
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{98}
\end{equation}
\]

Generate a candidate path from the current path:

\[
\begin{equation}
\tau'\sim q_{\mathrm{prop}}(\tau'\mid\tau).
\tag{99}
\end{equation}
\]

The Metropolis-Hastings acceptance rate is:

\[
\begin{equation}
A_k(\tau\rightarrow\tau')=\min\left(1,\exp\left(-\beta_k(H_{\mathrm{eff}}[\tau']-H_{\mathrm{eff}}[\tau])\right)\frac{q_{\mathrm{prop}}(\tau\mid\tau')}{q_{\mathrm{prop}}(\tau'\mid\tau)}\right).
\tag{100}
\end{equation}
\]

Expanding the Hamiltonian difference:

\[
\begin{align}
H_{\mathrm{eff}}[\tau']-H_{\mathrm{eff}}[\tau]
&=H_0[\tau']-H_0[\tau]-\lambda_G(G[\tau']-G[\tau]) \nonumber \\
&\quad +\lambda_{\mathrm{KL}}(D_{\mathrm{KL}}[\tau']-D_{\mathrm{KL}}[\tau])+\lambda_N(N[\tau']-N[\tau]).
\tag{101}
\end{align}
\]

Therefore, high return, low KL, and short paths reduce $H_{\mathrm{eff}}$ and are more likely to be accepted. MCMC does not merely add randomness; it uses the Boltzmann acceptance rate to push paths toward low-Hamiltonian regions. When $\beta_k$ is small, the acceptance rate is more lenient, which helps exploration. When $\beta_k$ is large, the acceptance rate is more biased toward low-Hamiltonian paths, which helps convergence.

### 4.5. Langevin path sampling

Langevin is more suitable for continuous degrees of freedom, especially hidden actions $a_t=z_t$. The dimensionless action in the $k$-th annealing iteration is:

\[
\begin{equation}
S_{\mathrm{eff}}^{(k)}[\tau]=\beta_k H_{\mathrm{eff}}[\tau].
\tag{102}
\end{equation}
\]

Perform a Langevin update on the local segment $a_{u:v}$:

\[
\begin{equation}
a_{u:v}^{(k+1)}=a_{u:v}^{(k)}-\epsilon\nabla_{a_{u:v}}S_{\mathrm{eff}}^{(k)}[\tau^{(k)}]+\sqrt{2\epsilon}\,\xi_k=a_{u:v}^{(k)}-\epsilon\beta_k\nabla_{a_{u:v}}H_{\mathrm{eff}}[\tau^{(k)}]+\sqrt{2\epsilon}\,\xi_k.
\tag{103}
\end{equation}
\]

where:

\[
\begin{equation}
\xi_k\sim \mathcal N(0,I).
\tag{104}
\end{equation}
\]

The physical picture is:
- Langevin = action-gradient descent + random perturbation
- $\beta_k$ controls the strength of the Hamiltonian gradient in sampling through $S_{\mathrm{eff}}^{(k)}=\beta_k H_{\mathrm{eff}}$
- when $\beta_k$ is small, the gradient selection pressure is weaker, which is more favorable for exploration
- when $\beta_k$ is large, the gradient selection pressure is stronger, making paths more biased toward low-Hamiltonian paths

### 4.6. Inverse-temperature annealing and the cooling picture

Let $k=0,1,\ldots,K_{\mathrm{ann}}$ denote the external annealing iteration step. The Gibbs sampling distribution in the $k$-th round is:

\[
\begin{equation}
q_k(\tau)\propto \exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{105}
\end{equation}
\]

Simulated annealing is implemented by increasing the inverse temperature:

\[
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}.
\tag{106}
\end{equation}
\]

A small $\beta_k$ corresponds to high-temperature exploration, where more high-Hamiltonian paths can still be retained. A large $\beta_k$ corresponds to low-temperature convergence, where the path distribution gradually concentrates in low-Hamiltonian regions.

Section 3 can also use simulated annealing, but there it schedules observation and reward proposals such as $q_o^{(k)}$ and $q_r^{(k)}$, while in this section simulated annealing schedules the inverse temperature $\beta_k$ in the Gibbs / Boltzmann distribution. Both can prevent premature trapping in local optima, but the mathematical objects are different.

### 4.7 Related research

- Path-integral RL
  - Corresponding literature direction: Path Integral Control / PI${}^2$
  - Directly related:
    - [[PMLR 2010] Learning Policy Improvements with Path Integrals](https://proceedings.mlr.press/v9/theodorou10a.html)
    - [[JMLR 2010] A Generalized Path Integral Control Approach to Reinforcement Learning](https://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf)
  - Same direction:
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)

- KL penalty / control cost
  - Corresponding literature direction: KL control / linearly-solvable MDP
  - Directly related:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)
  - Same direction:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

- RL as inference
  - Corresponding literature direction: maximum entropy RL / control as inference / SAC
  - Directly related:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  - Same direction:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)

- $e^{-\beta H}$ sampling
  - Corresponding literature direction: EBM / Gibbs / MCMC / Langevin
  - Directly related:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - Same direction:
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

- Langevin on hidden latents
  - Corresponding literature direction: latent EBM / energy-based text generation / continuous relaxation
  - Directly related:
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
    - [[ICML 2021] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://proceedings.mlr.press/v139/pang21a.html)
  - Same direction:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2511.07124] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought](https://arxiv.org/abs/2511.07124)

- LLM inference sampling
  - Corresponding literature direction: MCMC-inspired reasoning / constrained sampling
  - Directly related:
    - [[arXiv:2506.05754] Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - Same direction:
    - [[arXiv:2510.14901] Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)

---

## 5. Relationship between the two branches

Both routes are used to expand sampled paths, but their mathematical objects are different.

Section 3 is RL estimator / rollout augmentation. Starting from the original cumulative-return definition, it resamples $o_{t+1}$ and $r_t$:

\[
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t).
\tag{107}
\end{equation}
\]

Then it estimates returns or advantages:

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{108}
\end{equation}
\]

Section 4 is path-space Gibbs sampling. It constructs a new path distribution:

\[
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{109}
\end{equation}
\]

Therefore, Section 3 can be connected to PPO / GRPO / GSPO, with the focus on improving rollout data and return estimation. Section 4 uses MCMC / Langevin / annealing to directly sample low-Hamiltonian paths. The statistical-mechanics picture can help explain path selection and annealing behavior in both cases.

---

## 6. Final overall formulas

The base action and base Hamiltonian satisfy:

\[
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{110}
\end{equation}
\]

The original RL objective is:

\[
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{111}
\end{equation}
\]

Section 3 uses the original cumulative-return definition of RL to resample observations and rewards:

\[
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t).
\tag{112}
\end{equation}
\]

The return estimate obtained in Section 3 can be passed to PPO / GRPO / GSPO:

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{113}
\end{equation}
\]

Section 4 introduces the external-field Hamiltonian:

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_NN[\tau].
\tag{114}
\end{equation}
\]

The effective Hamiltonian is:

\[
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]+H_{\mathrm{field}}[\tau].
\tag{115}
\end{equation}
\]

The effective action is:

\[
\begin{equation}
S_{\mathrm{eff}}[\tau]=\beta H_{\mathrm{eff}}[\tau].
\tag{116}
\end{equation}
\]

The Gibbs sampling distribution in Section 4 is:

\[
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}\exp(-S_{\mathrm{eff}}[\tau])=\frac{1}{Z}\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{117}
\end{equation}
\]

The path sampling flow can be summarized as:

\[
\begin{align*}
&~ \text{rollout / proposal} \\
\rightarrow &~ \text{branch 1: resample }o,r\text{ and estimate }G \\
\rightarrow &~ \text{or branch 2: evaluate }H_{\mathrm{eff}}[\tau]\text{ and sample by Boltzmann weight} \\
\rightarrow &~ \text{PPO / GRPO / GSPO update or MCMC / Langevin search} \\
\rightarrow &~ \text{distill back to }\pi_\theta.
\end{align*}
\]

Physical picture:
- history-based RL is a one-dimensional temporal path integral
- $a,o,r$ are internal degrees of freedom on the path
- hidden $z$ can be used as a continuous action $a$
- the original expectation is $\int e^{-\beta H_0}G$
- Section 3 expands the $o,r$ feedback space on the original RL estimator
- Section 4 introduces the external-field Hamiltonian $H_{\mathrm{field}}$, and through the Boltzmann weight $e^{-\beta(H_0+H_{\mathrm{field}})}$, produces path selection biased toward high return, low KL, and controlled path length

---

# Appendix A: Optional observation confidence and reward confidence

If the world model or reward model itself needs confidence penalties, these terms can be added as optional external fields, rather than written into the main formulas. The observation-confidence term can be written as:

\[
\begin{equation}
C_o[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_o(o_{t+1}\mid a_t,h_t).
\tag{A1}
\end{equation}
\]

The reward-confidence term can be written as:

\[
\begin{equation}
C_r[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_r(r_t\mid o_{t+1},a_t,h_t).
\tag{A2}
\end{equation}
\]

If these optional terms are used, the external-field Hamiltonian is extended to:

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_NN[\tau]+\rho_o C_o[\tau]+\rho_r C_r[\tau].
\tag{A3}
\end{equation}
\]

The corresponding Gibbs sampling distribution is still:

\[
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}\exp(-\beta(H_0[\tau]+H_{\mathrm{field}}[\tau])).
\tag{A4}
\end{equation}
\]

---

# Appendix B: Renormalization along the time direction

Because this is a one-dimensional temporal path integral, renormalization is mainly performed along the time direction. Let $\ell$ denote the macro time-block index, and merge every $b$ microscopic actions into one macro block:

\[
\begin{equation}
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1}).
\tag{B1}
\end{equation}
\]

After multi-layer compression:

\[
\begin{equation}
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}.
\tag{B2}
\end{equation}
\]

At the path-integral level, the mapping from microscopic paths to macroscopic paths is $\bar\tau=\mathcal C(\tau)$, and the macroscopic effective action is obtained by integrating out microscopic degrees of freedom:

\[
\begin{equation}
\exp(-S_{\mathrm{eff}}[\bar\tau])=\int_{\mathcal C(\tau)=\bar\tau}\exp(-S_{\mathrm{eff}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{B3}
\end{equation}
\]

If the effective singular-value spectrum inside the compressed block decays rapidly, truncation is safe; if the spectrum is approximately flat, hard truncation will lose a large amount of information:

\[
\begin{equation}
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ truncation causes strong information loss}.
\tag{B4}
\end{equation}
\]

The physical picture is:
- temporal RG can compress long-time paths into macroscopic paths, but the model must support compressed tokens / hidden macro-actions; otherwise the compression is only a summary, not an effective degree of freedom
