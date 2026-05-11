# 从 history-based RL 到重采样以及 RL 的物理图像

## 0. 目标

这里讨论的是一种**物理图像解释**，不是把 RL 严格重写成场论。

文章意图是讨论扩大采样路径的物理方法，主线有两条：

- **再采样** : 在 RL 原始累积回报定义下，对观测量 $o_{t+1}$ 和奖励 $r_t$ 做再采样 / 重估计，用于改进 rollout、回报估计、advantage 估计和 PPO / GRPO / GSPO 更新。
- **路径积分** : 在路径积分图像下，引入有效哈密顿量、Boltzmann 权重、Gibbs 采样、MCMC / Langevin 和逆温度退火，用于直接在路径空间中扩大采样并寻找低哈密顿量路径。

共同基础是 : 
- history-based RL 可以写成一维时间方向上的路径积分
- 原始期望中的回报 $G[\tau]$ 是路径上的 observable insertion
- 在热力学类比中，作用量 $S$ 是无量纲量，哈密顿量 $H$ 具有能量量纲，并满足 $S=\beta H$

---

## 1. 原始 history-based RL 轨迹积分

从最原始的形式开始。交互历史为 : 

```math
\begin{equation}
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\ldots,a_{t-1},r_{t-1},o_t).
\tag{1}
\end{equation}
```

策略为 $\pi(a_t\mid h_t)$，环境条件密度为 $\mu(o_{t+1},r_t\mid a_t,h_t)$。有限时间 $T$ 内，整条轨迹的期望回报可以写成 : 

```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right).
\tag{2}
\end{equation}
```

整条路径写成 : 

```math
\begin{equation}
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T).
\tag{3}
\end{equation}
```

路径回报写成 : 

```math
\begin{equation}
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s.
\tag{4}
\end{equation}
```

于是原始 RL 目标就是对所有可能路径做加权积分，每条路径的权重由策略和环境共同给出，每条路径的值由折扣回报 $G[\tau]$ 给出。

### 1.1. 决定论环境下的 Dirac delta 退化

如果环境是决定论的，则给定 $a_t,h_t$ 后，下一步观测和奖励由确定函数给出 : 

```math
\begin{equation}
o_{t+1}=O(a_t,h_t),\quad r_t=R(o_{t+1},a_t,h_t).
\tag{5}
\end{equation}
```

环境条件密度退化为 Dirac delta : 

```math
\begin{equation}
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t)).
\tag{6}
\end{equation}
```

代回原始轨迹积分 : 

```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right).
\tag{7}
\end{equation}
```

利用 Dirac delta 的基本积分性质 : 

```math
\begin{equation}
\int \delta(x-x_0)f(x)\,dx=f(x_0).
\tag{8}
\end{equation}
```

环境部分坍缩后，只剩动作采样 : 

```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right].
\tag{9}
\end{equation}
```

这意味着 : 
- 决定论环境不再提供路径分支，路径分支只来自策略采样
- 如果策略也决定论，整条路径坍缩成单条路径

### 1.2. 有限时域 (finite horizon)、有限奖励 (reward clipping)

如果限制最大路径长度 $T\le T_{\max}$，路径积分只在有限时间区间上进行 : 

```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right).
\tag{10}
\end{equation}
```

如果对奖励进行裁剪 : 

```math
\begin{equation}
\bar r_t=\operatorname{clip}(r_t,-r_{\max},r_{\max}),\quad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t.
\tag{11}
\end{equation}
```

意义 : 
- 有限时域是时间截断，有限奖励是回报观测量的有界化

### 1.3. LLM RL 中的路径采样与 hidden 动作

对 LLM 来说，动作 $a_t$ 可以是 token、完整回答片段、tool call、代码 patch 或 agent step。如果动作是 token，则模型前向给出 logits，再经过采样得到 token : 

```math
\begin{equation}
z_t^{\mathrm{logit}}=f_\theta(h_t),\quad \pi_{\theta,T_{\mathrm{dec}}}(a_t\mid h_t)=\frac{\exp(z^{\mathrm{logit}}_{t,a_t}/T_{\mathrm{dec}})}{\sum_{a'}\exp(z^{\mathrm{logit}}_{t,a'}/T_{\mathrm{dec}})}.
\tag{12}
\end{equation}
```

如果要在连续空间做路径采样，可以用 prefill 后得到的 hidden 作为连续动作表示 : 

```math
\begin{equation}
a_t=z_t,\quad z_t=\operatorname{hidden}_\theta(\operatorname{prefill}(a_{\le t})).
\tag{13}
\end{equation}
```

因此连续动作路径仍然写成同一个路径符号 : 

```math
\begin{equation}
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T).
\tag{14}
\end{equation}
```

这里没有引入新的路径变量，只是把离散 token action $a_t$ 替换为连续 hidden action $z_t$。在不强调 hidden 时仍写 $a_t$。

第 $i$ 条样本为 : 

```math
\begin{equation}
\tau_i=(a_{i,0},o_{i,1},r_{i,0},a_{i,1},o_{i,2},r_{i,1},\ldots,a_{i,T_i},o_{i,T_i+1},r_{i,T_i}).
\tag{15}
\end{equation}
```

如果用 hidden 作为动作，则 : 

```math
\begin{equation}
a_{i,t}=z_{i,t},\quad z_{i,t}=\operatorname{hidden}_\theta(\operatorname{prefill}(a_{i,\le t})).
\tag{16}
\end{equation}
```

这里 $i$ 是样本编号，$t$ 是路径上的时间步或 token 位置。

---

## 2. 原始 history-based RL 的物理图像解释

### 2.1. 从轨迹积分到路径积分

把原始密度连乘写成路径密度 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{17}
\end{equation}
```

于是原始目标可以写成路径积分形式 : 

```math
\begin{equation}
J(\pi)=\int P_{\pi,\mu}[\tau]G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{18}
\end{equation}
```

物理图像是 : 
- $\tau$ 是一条 worldline，$P_{\pi,\mu}[\tau]$ 是路径权重，$G[\tau]$ 是路径收益泛函，$J(\pi)$ 是所有路径的加权平均

### 2.2. 基础作用量、基础哈密顿量与玻尔兹曼权重

从路径密度出发 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{19}
\end{equation}
```

对路径密度取负对数，得到基础作用量 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=-\log P_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{20}
\end{equation}
```

热力学类比中，玻尔兹曼权重写成 $\exp(-\beta H)$。因此基础作用量与基础哈密顿量的关系是 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau].
\tag{21}
\end{equation}
```

于是原始路径密度可以写成 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\exp(-S_{\pi,\mu}[\tau])=\exp(-\beta H_0[\tau]).
\tag{22}
\end{equation}
```

原始 RL 目标变成 : 

```math
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{23}
\end{equation}
```

这里 $G[\tau]$ 是 observable insertion，不是哈密顿量的一部分。

### 2.3. 一维时间方向的单体复杂系统

路径变量可以写成 : 

```math
\begin{equation}
x_t=(a_t,o_{t+1},r_t),\quad \tau=(x_0,x_1,\ldots,x_T).
\tag{24}
\end{equation}
```

它是一维时间方向上的路径系统。复杂性来自 history coupling，因为每一步的策略和环境都依赖完整历史 : 

```math
\begin{equation}
\pi(a_t\mid h_t),\quad \mu(o_{t+1},r_t\mid a_t,h_t).
\tag{25}
\end{equation}
```

对应基础作用量项为 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{26}
\end{equation}
```

物理图像是 : 
- RL path integral 是一维时间方向上的复杂单体系统，内部自由度是 action、observation、reward，复杂性来自这些自由度通过历史 $h_t$ 发生长程时间耦合

### 2.4. 折扣因子与 Laplace 正则化

连续时间回报如果写成 : 

```math
\begin{equation}
G[\tau]=\int_0^\infty r(t)\,dt,
\tag{27}
\end{equation}
```

可能发散。加入指数衰减后 : 

```math
\begin{equation}
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt.
\tag{28}
\end{equation}
```

离散时间中 : 

```math
\begin{equation}
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t,
\tag{29}
\end{equation}
```

令时间步长为 $\Delta t$，对应关系为 : 

```math
\begin{equation}
\gamma=e^{-\lambda\Delta t},\quad \gamma^t=e^{-\lambda t\Delta t}.
\tag{30}
\end{equation}
```

如果奖励有界 : 

```math
\begin{equation}
|r_t|\le r_{\max},
\tag{31}
\end{equation}
```

则折扣回报有界 : 

```math
\begin{equation}
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}.
\tag{32}
\end{equation}
```

物理图像是 : 
- discount factor 是时间方向上的 Laplace damping，它把无限未来压成有限有效贡献

---

## 3. RL 原定义下的观测量与奖励再采样

从原始 RL 累积回报定义出发，目标是在不改变策略更新器的前提下，扩展观测量 $o_{t+1}$ 和奖励 $r_t$ 的局部采样空间，使模型更好地探索路径。更复杂的做法可以使用真实环境重复采样、world model、reward model、verifier、SMC 或 CEM proposal等，但是本节不讨论这些复杂的重复采样法，只使用最基本的高斯噪声 proposal。

### 3.1. $o$ 和 $r$ 的高斯噪声再采样 / 重估计

设 $m=1,\ldots,M$ 表示在同一个 $a_t,h_t,o_{t+1},r_t$ 条件下的第 $m$ 次重采样。若使用模拟退火，设 $k=0,1,\ldots,K_{\mathrm{ann}}$ 表示外部退火迭代步，不是路径内部时间步 $t$。第 $k$ 轮使用逆温度 $\beta_k>0$。设 $\sigma_o>0$ 是观测量的基础噪声尺度，$\sigma_r>0$ 是奖励的基础噪声尺度。$\sigma_o,\sigma_r$ 是人为指定的 proposal 噪声强度，不是组内统计得到的方差。

先从一维标量变量开始。给定当前值 $x_0$ 和第 $k$ 轮噪声宽度 $s_k>0$，最基本的一维高斯 proposal 定义为 : 

```math
\begin{equation}
q_k(x'\mid x_0)=\frac{1}{\sqrt{2\pi}s_k}\exp\left(-\frac{(x'-x_0)^2}{2s_k^2}\right).
\tag{33}
\end{equation}
```

为了让逆温度控制 proposal 的宽度，定义 : 

```math
\begin{equation}
s_k=\frac{\sigma}{\sqrt{\beta_k}},
\tag{34}
\end{equation}
```

其中 $\sigma>0$ 是基础噪声尺度。把式 (34) 代入式 (33)，得到 : 

```math
\begin{equation}
q_k(x'\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x'-x_0)^2}{2\sigma^2}\right).
\tag{35}
\end{equation}
```

标准高斯噪声 $\xi^{(m)}$ 的密度为 : 

```math
\begin{equation}
p(\xi^{(m)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi^{(m)})^2}{2}\right).
\tag{36}
\end{equation}
```

令第 $m$ 次重采样值为 : 

```math
\begin{equation}
x^{(m,k)}=x_0+\frac{\sigma}{\sqrt{\beta_k}}\xi^{(m)}.
\tag{37}
\end{equation}
```

由式 (37) 可得 : 

```math
\begin{equation}
\xi^{(m)}=\frac{\sqrt{\beta_k}}{\sigma}(x^{(m,k)}-x_0).
\tag{38}
\end{equation}
```

因此 $x^{(m,k)}$ 的密度正是以 $x_0$ 为中心、宽度为 $\sigma/\sqrt{\beta_k}$ 的高斯 proposal : 

```math
\begin{equation}
q_k(x^{(m,k)}\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x^{(m,k)}-x_0)^2}{2\sigma^2}\right).
\tag{39}
\end{equation}
```

所以小 $\beta_k$ 对应更宽的 proposal，大 $\beta_k$ 对应更窄的 proposal。这就是本节里的模拟退火控制方式。

如果 $o_{t+1}$ 是一维连续观测量，取 $x_0=o_{t+1}$，取 $\sigma=\sigma_o$，得到观测量重采样式 : 

```math
\begin{equation}
o_{t+1}^{(m,k)}=o_{t+1}+\frac{\sigma_o}{\sqrt{\beta_k}}\xi_o^{(m,t)}.
\tag{40}
\end{equation}
```

其中 $\xi_o^{(m,t)}$ 是第 $m$ 次重采样、路径时间步 $t$ 上的标准高斯噪声 : 

```math
\begin{equation}
p(\xi_o^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_o^{(m,t)})^2}{2}\right).
\tag{41}
\end{equation}
```

对应的观测量 proposal 为 : 

```math
\begin{equation}
q_k(o'_{t+1}\mid o_{t+1})=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_o}\exp\left(-\frac{\beta_k(o'_{t+1}-o_{t+1})^2}{2\sigma_o^2}\right).
\tag{42}
\end{equation}
```

如果 $o_{t+1}$ 是多维向量或文本 embedding，可以对每个坐标使用同样的一维高斯扰动。这里不引入相关噪声矩阵；如果以后需要相关噪声，必须先定义矩阵含义再写。

如果直接对标量奖励做高斯扰动，取 $x_0=r_t$，取 $\sigma=\sigma_r$，得到 : 

```math
\begin{equation}
r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\xi_r^{(m,t)}.
\tag{43}
\end{equation}
```

其中 : 

```math
\begin{equation}
p(\xi_r^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_r^{(m,t)})^2}{2}\right).
\tag{44}
\end{equation}
```

对应的奖励 proposal 为 : 

```math
\begin{equation}
q_k(r'_t\mid r_t)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_r}\exp\left(-\frac{\beta_k(r'_t-r_t)^2}{2\sigma_r^2}\right).
\tag{45}
\end{equation}
```

但是，如果只是对 $r_t$ 加零中心高斯噪声，然后做样本平均，平均值会回到原来的 $r_t$ 附近。按样本平均的原定义 : 

```math
\begin{equation}
\bar r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\left(\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\right).
\tag{46}
\end{equation}
```

当高斯噪声正负大致抵消时 : 

```math
\begin{equation}
\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\approx 0,
\tag{47}
\end{equation}
```

于是 : 

```math
\begin{equation}
\bar r_t^{(k)}\approx r_t.
\tag{48}
\end{equation}
```

因此，直接对奖励加噪声主要是鲁棒性扰动。更有意义的做法是先对观测量或观测 hidden 做高斯重采样，再用扰动后的观测重新计算奖励 : 

```math
\begin{equation}
r_t^{(m,k)}=R(o_{t+1}^{(m,k)},a_t,h_t).
\tag{49}
\end{equation}
```

对应的局部奖励估计为 : 

```math
\begin{equation}
\widehat r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}R(o_{t+1}^{(m,k)},a_t,h_t).
\tag{50}
\end{equation}
```

对第 $i$ 条动作路径，第 $m$ 次重采样路径为 : 

```math
\begin{equation}
\tau_i^{(m,k)}=(a_{i,0},o_{i,1}^{(m,k)},r_{i,0}^{(m,k)},\ldots,a_{i,T_i},o_{i,T_i+1}^{(m,k)},r_{i,T_i}^{(m,k)}).
\tag{51}
\end{equation}
```

第 $m$ 次重采样路径的回报为 : 

```math
\begin{equation}
G_i^{(m,k)}=\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{52}
\end{equation}
```

用样本平均的原定义得到第 $k$ 轮重采样回报估计 : 

```math
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}G_i^{(m,k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{53}
\end{equation}
```

这就是本节的核心 : 在原始 RL 累积回报定义下，用高斯噪声 proposal 扩展 $o_{t+1}$、$r_t$ 或它们的连续表示，再用样本平均估计更稳定的回报或 advantage。

### 3.2. PPO / GRPO / GSPO 接入

GRPO / GSPO 可以作为本节的细化方向。它们天然有组内样本结构，因此适合对重采样后的回报做组内统计，但这种方法同样可以服务于 PPO，因为 PPO 也只需要 rollout、reward、advantage 和策略更新比率。

设 $K_s$ 表示第 $k$ 轮的组内样本数，$i,j$ 表示组内样本编号，$t$ 表示路径内部时间步或 token 位置。对同一个输入 $x$，第 $k$ 轮采样 $K_s$ 条路径 : 

```math
\begin{equation}
\tau_1^{(k)},\tau_2^{(k)},\ldots,\tau_{K_s}^{(k)}\sim q_k(\tau\mid x).
\tag{54}
\end{equation}
```

每条路径使用高斯噪声重采样得到回报估计 : 

```math
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{55}
\end{equation}
```

组内平均与方差为 : 

```math
\begin{equation}
\bar G^{(k)}=\frac{1}{K_s}\sum_{i=1}^{K_s}\widehat G_i^{(k)},\quad (\sigma_G^{(k)})^2=\frac{1}{K_s}\sum_{i=1}^{K_s}(\widehat G_i^{(k)}-\bar G^{(k)})^2.
\tag{56}
\end{equation}
```

标准化 advantage 为 : 

```math
\begin{equation}
A_i^{(k)}=\frac{\widehat G_i^{(k)}-\bar G^{(k)}}{\sigma_G^{(k)}+\epsilon}.
\tag{57}
\end{equation}
```

如果要对观测量和奖励本身做局部统计，可以写成 : 

```math
\begin{equation}
\bar o_{t+1}^{(k)}=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}o_{i,t+1}^{(m,k)},\quad \bar r_t^{(k)}=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}r_{i,t}^{(m,k)}.
\tag{58}
\end{equation}
```

对应方差为 : 

```math
\begin{equation}
(s_o^{(k)})^2=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}(o_{i,t+1}^{(m,k)}-\bar o_{t+1}^{(k)})^2,
\quad
(s_r^{(k)})^2=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}(r_{i,t}^{(m,k)}-\bar r_t^{(k)})^2.
\tag{59}
\end{equation}
```

这里的 $s_o^{(k)}$ 和 $s_r^{(k)}$ 是组内统计得到的反馈宽度，不是 proposal 里的基础噪声尺度 $\sigma_o$ 和 $\sigma_r$。方差变小可以解释为反馈估计趋于稳定，或者当前策略进入某个更稳定的局部区域。

GSPO 可以看成把整条生成序列作为采样单位 : 

```math
\begin{equation}
\tau_i=(a_{i,0},o_{i,1},r_{i,0},\ldots,a_{i,T_i},o_{i,T_i+1},r_{i,T_i}).
\tag{60}
\end{equation}
```

如果模型动作、工具返回、环境观测和奖励文本都混在同一个 sequence 中，sequence-level 采样会近似把 $a$ 和 $o$ 混在一起采样。更合理的因果分解是 : 

```math
\begin{equation}
q(\tau)=q_a(a_{0:T})q_o(o_{1:T+1}\mid a_{0:T},h_{0:T})q_r(r_{0:T}\mid a_{0:T},o_{1:T+1},h_{0:T}).
\tag{61}
\end{equation}
```

如果用 hidden 作为动作 : 

```math
\begin{equation}
q(\tau)=q_z(z_{0:T})q_o(o_{1:T+1}\mid z_{0:T},h_{0:T})q_r(r_{0:T}\mid z_{0:T},o_{1:T+1},h_{0:T}).
\tag{62}
\end{equation}
```

意义 : 
- 扩展观测量和奖励反馈空间
- GRPO / GSPO 的组内统计可以更自然地估计 $G$、advantage、$o$ 和 $r$ 的稳定性
- PPO 也可以使用同样的 $o,r$ 再采样数据，只是后续策略更新器不同

### 3.3. 模拟退火

模拟退火在本节中用于调度高斯噪声 proposal。第 $k$ 轮的逆温度 $\beta_k$ 通过式 (40) 和式 (43) 控制高斯噪声宽度 : 

```math
\begin{equation}
\frac{\sigma_o}{\sqrt{\beta_k}},\quad \frac{\sigma_r}{\sqrt{\beta_k}}.
\tag{63}
\end{equation}
```

其中逆温度递增 : 

```math
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}.
\tag{64}
\end{equation}
```

小 $\beta_k$ 对应早期更宽的高斯噪声 proposal，大 $\beta_k$ 对应后期更窄、更保守的反馈估计。这里的 $\beta_k$ 是模拟退火调度参数，用于解释采样宽度和反馈涨落，不是 decoder temperature。

### 3.4. 统计力学解释

从第二节的物理图像看，原始 history-based RL 已经可以被看成一维时间路径系统。策略和环境诱导出原始路径分布 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{65}
\end{equation}
```

在本节中，回报作为路径上的统计观测量 : 

```math
\begin{equation}
G[\tau]=\sum_{t=0}^{T}\gamma^t r_t.
\tag{66}
\end{equation}
```

对同一输入采样得到的一组路径可以看成原始路径分布或其 rollout augmentation 下的局部 ensemble : 

```math
\begin{equation}
\tau_i\sim P_{\pi,\mu}[\tau],\quad i=1,\ldots,K_s.
\tag{67}
\end{equation}
```

组内回报均值和方差为 : 

```math
\begin{equation}
\bar G=\frac{1}{K_s}\sum_{i=1}^{K_s}G[\tau_i],\quad \sigma_G^2=\frac{1}{K_s}\sum_{i=1}^{K_s}(G[\tau_i]-\bar G)^2.
\tag{68}
\end{equation}
```

这里 $\bar G$ 描述该局部 ensemble 的平均回报，$\sigma_G^2$ 描述回报涨落。对 $o_{t+1}$ 和 $r_t$ 的高斯噪声再采样，可以看成扩大这个局部 ensemble，使 advantage 或回报估计不只依赖单次 rollout 的偶然结果。

模拟退火在本节中首先是采样 / 估计调度机制。早期使用更小的 $\beta_k$，使高斯 proposal 更宽，允许更大的反馈涨落；后期逐渐增大 $\beta_k$，使 proposal 收缩到更稳定的反馈估计。用统计力学语言看，这类似从高温探索到低温稳定的冷却过程。这里的回报仍然只是统计观测量，用于估计 advantage、排序样本、更新 PPO / GRPO / GSPO。

### 3.5. 相关研究

- $(o,r)$ 再采样
  - 对应文献方向: Gaussian noise proposal / noisy environment augmentation / observation noise / reward noise
  - 直接相关:
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
    - [[arXiv:2106.11420] Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/abs/2106.11420)
    - [[arXiv:1810.01032] Reinforcement Learning with Perturbed Rewards](https://arxiv.org/abs/1810.01032)
    - [[PMLR 2020] Deep Reinforcement Learning with Robust and Smooth Policy](https://proceedings.mlr.press/v119/shen20b.html)
  - 同方向:
    - [[arXiv:2310.00344] HarmonyDream: Task Harmonization Inside World Models](https://arxiv.org/abs/2310.00344)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)

- 组内统计
  - 对应文献方向: GRPO / GSPO
  - 直接相关:
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
  - 同方向:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- proposal 扩展
  - 对应文献方向: SMC policy optimization / CEM / iCEM
  - 直接相关:
    - [[arXiv:2402.07963] SPO: Sequential Monte Carlo Policy Optimisation](https://arxiv.org/abs/2402.07963)
    - [[arXiv:2505.16732] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](https://arxiv.org/abs/2505.16732)
    - [[arXiv:2008.06389] Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/abs/2008.06389)
    - [[arXiv:2112.07746] CEM-GD: Cross-Entropy Method with Gradient Descent Planner for Model-Based Reinforcement Learning](https://arxiv.org/abs/2112.07746)
  - 同方向:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)

- 用原 RL 估计器接 PPO / GRPO / GSPO
  - 对应文献方向: PPO-family + noisy rollout augmentation / group-level RL
  - 直接相关:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
  - 同方向:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)


---

## 4. 路径积分 / 有效哈密顿量 / 路径采样

### 4.1. 外场哈密顿量：RL 奖励与惩罚项的物理表示

本节从第二节的基础哈密顿量表示出发。基础哈密顿量 $H_0[\tau]$ 来自策略和环境的原始路径权重，外场哈密顿量用于表示 RL 中的奖励、KL 惩罚、路径长度成本等目标项。这里的外场哈密顿量是目标结构，不是采样方法。

引入回报外场、KL 广义化学势和路径长度化学势后，外场哈密顿量写成 : 

```math
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_G G[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_N N[\tau].
\tag{69}
\end{equation}
```

其中 $\lambda_G$ 是控制高回报路径偏好的外场强度，$\lambda_{\mathrm{KL}}$ 是广义化学势或 KL 回拉强度，$\lambda_N$ 是路径长度或交互次数的化学势成本。若前文使用了 clipped return，也可以在这里用 $\bar G[\tau]$ 替代 $G[\tau]$。

总哈密顿量为 : 

```math
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]+H_{\mathrm{field}}[\tau].
\tag{70}
\end{equation}
```

对应的有效作用量为 : 

```math
\begin{equation}
S_{\mathrm{eff}}[\tau]=\beta H_{\mathrm{eff}}[\tau]=\beta H_0[\tau]+\beta H_{\mathrm{field}}[\tau].
\tag{71}
\end{equation}
```

展开后 : 

```math
\begin{equation}
S_{\mathrm{eff}}[\tau]=S_{\pi,\mu}[\tau]-\beta \lambda_GG[\tau]+\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\beta\lambda_NN[\tau].
\tag{72}
\end{equation}
```

路径 KL 项为 : 

```math
\begin{equation}
D_{\mathrm{KL}}[\tau]=\sum_{t=0}^{T}D_{\mathrm{KL}}\left(\pi(\cdot\mid h_t)\Vert \pi_{\mathrm{ref}}(\cdot\mid h_t)\right).
\tag{73}
\end{equation}
```

狭义粒子数或资源数可写成 $N[\tau]$，例如 token 数、step 数、tool-call 数或 interaction 数。$\lambda_NN[\tau]$ 对应资源化学势成本。

如果直接使用 : 

```math
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]-\lambda_GG[\tau]+\lambda_NN[\tau],
\tag{74}
\end{equation}
```

当 $\lambda_N$ 过大时，低哈密顿量路径会偏向过短路径，可能导致任务无法完成。更安全的方式是把路径长度当作约束区间 : 

```math
\begin{equation}
C_N[\tau]=\mu_+\max(0,N[\tau]-N_{\max})^2+\mu_-\max(0,N_{\min}-N[\tau])^2.
\tag{75}
\end{equation}
```

对应外场哈密顿量为 : 

```math
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+C_N[\tau].
\tag{76}
\end{equation}
```

物理图像是 :
- $N[\tau]$ 不是越小越好，它是资源约束。真正的低哈密顿量路径应该是成功且短，不是短但失败

### 4.2. Boltzmann 权重：由哈密顿量得到路径权重

给定等效哈密顿量后，Boltzmann 权重为 : 

```math
\begin{equation}
W_\beta[\tau]=\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{77}
\end{equation}
```

展开成原始路径密度的形式 : 

```math
\begin{equation}
W_\beta[\tau]=P_{\pi,\mu}[\tau]\exp(\beta \lambda_GG[\tau]-\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]-\beta\lambda_NN[\tau]).
\tag{78}
\end{equation}
```

因此 $\beta\lambda_G$ 是 reward tilt 的实际指数强度，$\beta\lambda_{\mathrm{KL}}$ 是 KL 回拉的实际指数强度，$\beta\lambda_N$ 是路径长度惩罚的实际指数强度。$\beta$ 是热力学类比中的逆温度；实现中可以把它当成 annealing parameter，但它不是 decoder temperature。

### 4.3. Gibbs 采样：一种基于 Boltzmann 权重的采样方法

如果不构造新的 Gibbs 采样分布，而只是想把原始 observable insertion 写成纯指数形式，可以先取正值化回报 $\widetilde G[\tau]>0$。如果 $G[\tau]$ 允许取负，形式上也可以把负号写成复相位，使原始路径积分变成复权重路径积分，并进一步考虑复 Langevin 方法；但这会引入不必要的 phase/sign problem 和复随机过程复杂度。本文不走这条路线，而是使用正值化回报 $\widetilde G[\tau]>0$ 来保持实数权重。

```math
\begin{equation}
\widetilde G[\tau]=G[\tau]+c,\quad \widetilde G[\tau]>0.
\tag{79}
\end{equation}
```

此时 : 

```math
\begin{equation}
\exp(-\beta H_0[\tau])\widetilde G[\tau]=\exp(-\beta H_0[\tau]+\log\widetilde G[\tau]).
\tag{80}
\end{equation}
```

定义 observable 等效作用量 : 

```math
\begin{equation}
S_{\mathrm{obs}}[\tau]=\beta H_0[\tau]-\log\widetilde G[\tau].
\tag{81}
\end{equation}
```

于是 : 

```math
\begin{equation}
\widetilde J(\pi)=\int \exp(-S_{\mathrm{obs}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{82}
\end{equation}
```

如果基础路径密度归一化，则原始期望与正值化期望之间满足 : 

```math
\begin{equation}
J(\pi)=\widetilde J(\pi)-c.
\tag{83}
\end{equation}
```

这个路线只是原始 observable insertion 的指数化，不是 Gibbs 采样。

Gibbs 采样则是另一件事：它根据 Boltzmann 权重构造新的采样分布 : 

```math
\begin{equation}
q_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}(\tau\mid\pi,\mu)=\frac{1}{Z_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}}\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{84}
\end{equation}
```

配分函数为 : 

```math
\begin{equation}
Z_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}=\int \exp(-\beta H_{\mathrm{eff}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{85}
\end{equation}
```

原始 rollout 分布为 : 

```math
\begin{equation}
p_0(\tau\mid\pi,\mu)=P_{\pi,\mu}[\tau]=\exp(-\beta H_0[\tau]).
\tag{86}
\end{equation}
```

Gibbs 倾斜分布为 : 

```math
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}p_0(\tau\mid\pi,\mu)\exp(\beta \lambda_GG[\tau]-\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]-\beta\lambda_NN[\tau]).
\tag{87}
\end{equation}
```

只有在外场关闭时，Gibbs 分布才退化回原采样 : 

```math
\begin{equation}
\lambda_G=0,\quad \lambda_{\mathrm{KL}}=0,\quad \lambda_N=0\quad\Longrightarrow\quad q(\tau\mid\pi,\mu)=p_0(\tau\mid\pi,\mu).
\tag{88}
\end{equation}
```

因此，Gibbs 采样不是原采样。它是为了路径搜索和训练样本构造而引入的 reward-tilted ensemble。

如果从 Gibbs 分布中采样 : 

```math
\begin{equation}
\tau_i\sim q(\tau\mid\pi,\mu),\quad i=1,\ldots,K,
\tag{89}
\end{equation}
```

则样本均值估计的是 Gibbs ensemble 下的期望 : 

```math
\begin{equation}
\widehat{\mathbb E}_{q}[G]=\frac{1}{K}\sum_{i=1}^{K}G[\tau_i].
\tag{90}
\end{equation}
```

这不是原始 rollout 分布下的 $J(\pi)$。原始期望是 : 

```math
\begin{equation}
J(\pi)=\mathbb E_{p_0}[G[\tau]].
\tag{91}
\end{equation}
```

Gibbs 采样下的期望是 : 

```math
\begin{equation}
\mathbb E_q[G[\tau]]=\frac{\partial \log Z}{\partial(\beta \lambda_G)}.
\tag{92}
\end{equation}
```

如果要从 Gibbs 样本反推原始 $J(\pi)$，需要 importance reweighting : 

```math
\begin{equation}
w_i=\exp(-\beta \lambda_GG[\tau_i]+\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau_i]+\beta\lambda_NN[\tau_i]).
\tag{93}
\end{equation}
```

于是 : 

```math
\begin{equation}
\widehat J(\pi)=\frac{\sum_{i=1}^{K}w_iG[\tau_i]}{\sum_{i=1}^{K}w_i}.
\tag{94}
\end{equation}
```

当外场很强时，反向权重方差会变大。因此，如果目标是估计原始 $J(\pi)$，应从 $p_0$ 直接采样，或者使用很小的 $\beta \lambda_G$ 并做外插；如果目标是扩大探索并生成高价值训练样本，则可以直接使用 Gibbs 采样分布 $q$。

当 $\beta \lambda_G$ 很小时 : 

```math
\begin{equation}
\exp(\beta \lambda_GG[\tau])=1+\beta \lambda_GG[\tau]+O((\beta \lambda_G)^2).
\tag{95}
\end{equation}
```

此时 Gibbs 分布接近原始 rollout 分布 : 

```math
\begin{equation}
q(\tau\mid\pi,\mu)\approx p_0(\tau\mid\pi,\mu).
\tag{96}
\end{equation}
```

如果在多个小外场强度下采样 : 

```math
\begin{equation}
g_j=\frac{1}{K_j}\sum_{i=1}^{K_j}G[\tau_i^{(j)}],\quad \tau_i^{(j)}\sim q_{\beta \lambda_G^{(j)}}(\tau\mid\pi,\mu),
\tag{97}
\end{equation}
```

则可以用 $g_j$ 对 $\beta \lambda_G\to 0$ 做外插，从而近似原始 $J(\pi)$。代价是资源开销增加，因为每个外场强度都需要采样。

### 4.4. MCMC 路径采样

若使用模拟退火，设 $k=0,1,\ldots,K_{\mathrm{ann}}$ 表示外部采样器的退火迭代步，$\beta_k$ 是第 $k$ 轮的逆温度。目标路径分布为 : 

```math
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{98}
\end{equation}
```

从当前路径生成候选路径 : 

```math
\begin{equation}
\tau'\sim q_{\mathrm{prop}}(\tau'\mid\tau).
\tag{99}
\end{equation}
```

Metropolis-Hastings 接受率为 : 

```math
\begin{equation}
A_k(\tau\rightarrow\tau')=\min\left(1,\exp\left(-\beta_k(H_{\mathrm{eff}}[\tau']-H_{\mathrm{eff}}[\tau])\right)\frac{q_{\mathrm{prop}}(\tau\mid\tau')}{q_{\mathrm{prop}}(\tau'\mid\tau)}\right).
\tag{100}
\end{equation}
```

展开哈密顿量差 : 

```math
\begin{align}
H_{\mathrm{eff}}[\tau']-H_{\mathrm{eff}}[\tau]
&=H_0[\tau']-H_0[\tau]-\lambda_G(G[\tau']-G[\tau]) \nonumber \\
&\quad +\lambda_{\mathrm{KL}}(D_{\mathrm{KL}}[\tau']-D_{\mathrm{KL}}[\tau])+\lambda_N(N[\tau']-N[\tau]).
\tag{101}
\end{align}
```

因此，高回报、低 KL、短路径会降低 $H_{\mathrm{eff}}$，更容易被接受。MCMC 不只是增加随机性，而是用玻尔兹曼接受率把路径往低哈密顿量区域推。小 $\beta_k$ 时接受率更宽松，更利于探索；大 $\beta_k$ 时接受率更偏向低哈密顿量路径，更利于收敛。

### 4.5. Langevin 路径采样

Langevin 更适合在连续自由度上做，尤其是 hidden action $a_t=z_t$。第 $k$ 轮退火迭代的无量纲作用量为 : 

```math
\begin{equation}
S_{\mathrm{eff}}^{(k)}[\tau]=\beta_k H_{\mathrm{eff}}[\tau].
\tag{102}
\end{equation}
```

对局部片段 $a_{u:v}$ 做 Langevin 更新 : 

```math
\begin{equation}
a_{u:v}^{(k+1)}=a_{u:v}^{(k)}-\epsilon\nabla_{a_{u:v}}S_{\mathrm{eff}}^{(k)}[\tau^{(k)}]+\sqrt{2\epsilon}\,\xi_k=a_{u:v}^{(k)}-\epsilon\beta_k\nabla_{a_{u:v}}H_{\mathrm{eff}}[\tau^{(k)}]+\sqrt{2\epsilon}\,\xi_k.
\tag{103}
\end{equation}
```

其中 : 

```math
\begin{equation}
\xi_k\sim \mathcal N(0,I).
\tag{104}
\end{equation}
```

物理图像是 : 
- Langevin = 作用量梯度下降 + 随机扰动
- $\beta_k$ 通过 $S_{\mathrm{eff}}^{(k)}=\beta_k H_{\mathrm{eff}}$ 控制哈密顿量梯度在采样中的强度
- 小 $\beta_k$ 时梯度选择压力较弱，更利于探索
- 大 $\beta_k$ 时梯度选择压力增强，更偏向低哈密顿量路径

### 4.6. 逆温度退火与冷却图像

设 $k=0,1,\ldots,K_{\mathrm{ann}}$ 表示外部退火迭代步。第 $k$ 轮 Gibbs 采样分布为 : 

```math
\begin{equation}
q_k(\tau)\propto \exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{105}
\end{equation}
```

模拟退火通过递增逆温度实现 : 

```math
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}.
\tag{106}
\end{equation}
```

小 $\beta_k$ 对应高温探索，较多高哈密顿量路径仍能保留。大 $\beta_k$ 对应低温收敛，路径分布逐渐集中到低哈密顿量区域。

第三节也可以使用模拟退火，但它调度的是 $q_o^{(k)}$ 和 $q_r^{(k)}$ 这类观测量与奖励 proposal，而本节的模拟退火调度的是 Gibbs / Boltzmann 分布中的逆温度 $\beta_k$。两者都可以防止过早陷入局部最优，但数学对象不同。

### 4.7 相关研究

- 路径积分 RL
  - 对应文献方向: Path Integral Control / PI${}^2$
  - 直接相关:
    - [[PMLR 2010] Learning Policy Improvements with Path Integrals](https://proceedings.mlr.press/v9/theodorou10a.html)
    - [[JMLR 2010] A Generalized Path Integral Control Approach to Reinforcement Learning](https://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf)
  - 同方向:
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)

- KL 惩罚 / 控制成本
  - 对应文献方向: KL control / linearly-solvable MDP
  - 直接相关:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)
  - 同方向:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

- RL as inference
  - 对应文献方向: maximum entropy RL / control as inference / SAC
  - 直接相关:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  - 同方向:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)

- $e^{-\beta H}$ 采样
  - 对应文献方向: EBM / Gibbs / MCMC / Langevin
  - 直接相关:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - 同方向:
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

- hidden latent 上 Langevin
  - 对应文献方向: latent EBM / energy-based text generation / continuous relaxation
  - 直接相关:
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
    - [[ICML 2021] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://proceedings.mlr.press/v139/pang21a.html)
  - 同方向:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2511.07124] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought](https://arxiv.org/abs/2511.07124)

- LLM 推理采样
  - 对应文献方向: MCMC-inspired reasoning / constrained sampling
  - 直接相关:
    - [[arXiv:2506.05754] Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - 同方向:
    - [[arXiv:2510.14901] Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)

---

## 5. 两个分支的关系

两条路线都用于扩大采样路径，但数学对象不同。

第三节是 RL estimator / rollout augmentation。它从原始累积回报定义出发，对 $o_{t+1}$ 和 $r_t$ 做再采样 : 

```math
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t).
\tag{107}
\end{equation}
```

然后估计回报或 advantage : 

```math
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{108}
\end{equation}
```

第四节是路径空间 Gibbs 采样。它构造新的路径分布 : 

```math
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{109}
\end{equation}
```

因此，第三节可以接入 PPO / GRPO / GSPO，重点是改进 rollout 数据和回报估计，第四节使用 MCMC / Langevin / 退火直接采样低哈密顿量路径。统计力学图像可以帮助解释两者中的路径选择和退火行为。

---

## 6. 最终总公式

基础作用量和基础哈密顿量满足 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{110}
\end{equation}
```

原始 RL 目标为 : 

```math
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{111}
\end{equation}
```

第三节使用原始 RL 累积回报定义，对观测量和奖励进行再采样 : 

```math
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t).
\tag{112}
\end{equation}
```

第三节得到的回报估计可以交给 PPO / GRPO / GSPO : 

```math
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{113}
\end{equation}
```

第四节引入外场哈密顿量 : 

```math
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_NN[\tau].
\tag{114}
\end{equation}
```

等效哈密顿量为 : 

```math
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]+H_{\mathrm{field}}[\tau].
\tag{115}
\end{equation}
```

等效作用量为 : 

```math
\begin{equation}
S_{\mathrm{eff}}[\tau]=\beta H_{\mathrm{eff}}[\tau].
\tag{116}
\end{equation}
```

第四节的 Gibbs 采样分布为 : 

```math
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}\exp(-S_{\mathrm{eff}}[\tau])=\frac{1}{Z}\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{117}
\end{equation}
```

路径采样流程可以概括为 : 

```math
\begin{align*}
&~ \text{rollout / proposal} \\
\rightarrow &~ \text{branch 1: resample }o,r\text{ and estimate }G \\
\rightarrow &~ \text{or branch 2: evaluate }H_{\mathrm{eff}}[\tau]\text{ and sample by Boltzmann weight} \\
\rightarrow &~ \text{PPO / GRPO / GSPO update or MCMC / Langevin search} \\
\rightarrow &~ \text{distill back to }\pi_\theta.
\end{align*}
```

物理图像 : 
- history-based RL 是一维时间路径积分
- $a,o,r$ 是路径内部自由度
- hidden $z$ 可以作为连续动作 $a$
- 原始期望是 $\int e^{-\beta H_0}G$
- 第三节在原始 RL 估计器上扩展 $o,r$ 反馈空间
- 第四节引入外场哈密顿量 $H_{\mathrm{field}}$，通过玻尔兹曼权重 $e^{-\beta(H_0+H_{\mathrm{field}})}$ 产生偏向高回报、低 KL、受控路径长度的路径选择

---

# Appendix A : 可选观测可信度与奖励可信度

如果世界模型或奖励模型本身需要可信度惩罚，可以把这些项作为可选外场加入，而不是写进主体公式。观测可信度项可以写成 : 

```math
\begin{equation}
C_o[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_o(o_{t+1}\mid a_t,h_t).
\tag{A1}
\end{equation}
```

奖励可信度项可以写成 : 

```math
\begin{equation}
C_r[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_r(r_t\mid o_{t+1},a_t,h_t).
\tag{A2}
\end{equation}
```

如果使用这些可选项，外场哈密顿量扩展为 : 

```math
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_NN[\tau]+\rho_o C_o[\tau]+\rho_r C_r[\tau].
\tag{A3}
\end{equation}
```

对应的 Gibbs 采样分布仍然是 : 

```math
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}\exp(-\beta(H_0[\tau]+H_{\mathrm{field}}[\tau])).
\tag{A4}
\end{equation}
```

---

# Appendix B : 时间方向重整化

因为这是一个一维时间路径积分，所以重整化主要沿时间方向做。设 $\ell$ 表示宏观时间块编号，把每 $b$ 个微观 action 合并成一个宏观块 : 

```math
\begin{equation}
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1}).
\tag{B1}
\end{equation}
```

多层压缩后 : 

```math
\begin{equation}
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}.
\tag{B2}
\end{equation}
```

在路径积分层面，微观路径到宏观路径的映射为 $\bar\tau=\mathcal C(\tau)$，宏观有效作用量由积分掉微观自由度得到 : 

```math
\begin{equation}
\exp(-S_{\mathrm{eff}}[\bar\tau])=\int_{\mathcal C(\tau)=\bar\tau}\exp(-S_{\mathrm{eff}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{B3}
\end{equation}
```

如果压缩块内部的有效奇异值谱快速衰减，则截断安全；如果谱近似平直，则硬截断会损失大量信息 : 

```math
\begin{equation}
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ 的截断会造成强信息损失}.
\tag{B4}
\end{equation}
```

物理图像是 : 
- temporal RG 可以把长时间路径压缩成宏观路径，但模型必须支持 compressed token / hidden macro-action，否则压缩只是摘要，不是有效自由度
