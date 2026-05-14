# 从 history-based RL 到重采样以及 RL 的物理图像

## 0. 目标

这里讨论的是一种**物理图像解释**，不是把 RL 严格重写成场论。

文章意图是讨论扩大采样路径的物理方法，主线有两条 : 

- **再采样** : 在 RL 原始累积回报定义下，对观测量 $o_{t+1}$ 和奖励 $r_t$ 做再采样 / 重估计，用于改进 rollout、回报估计、advantage 估计和 PPO / GRPO / GSPO 更新
- **路径积分** : 在路径积分图像下，引入有效哈密顿量、Boltzmann 权重、Gibbs 采样、MCMC / Langevin 和逆温度退火，用于直接在路径空间中扩大采样并寻找低哈密顿量路径

共同基础是 : 
- history-based RL 可以写成一维时间方向上的路径积分
- 原始期望中的回报 $G[\tau]$ 是路径上的 observable insertion
- 在热力学类比中，作用量 $S$ 是无量纲量，哈密顿量 $H$ 具有能量量纲，并满足 $S=\beta H$

---

## 1. 原始 history-based RL 轨迹积分

从最原始的形式开始。交互历史为 : 
```math
\begin{equation}
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\ldots,a_{t-1},r_{t-1},o_t)
\tag{1}
\end{equation}
```

策略为 $\pi(a_t\mid h_t)$，环境条件密度为 $\mu(o_{t+1},r_t\mid a_t,h_t)$。有限时间 $T$ 内，整条轨迹的期望回报可以写成 : 
```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
\tag{2}
\end{equation}
```

整条路径写成 : 
```math
\begin{equation}
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
\tag{3}
\end{equation}
```

路径回报写成 : 
```math
\begin{equation}
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s
\tag{4}
\end{equation}
```

于是原始 RL 目标就是对所有可能路径做加权积分，每条路径的权重由策略和环境共同给出，每条路径的值由折扣回报 $G[\tau]$ 给出。

### 1.1. 决定论环境下的 Dirac delta 退化

如果环境是决定论的，则给定 $a_t,h_t$ 后，下一步观测和奖励由确定函数给出 : 
```math
\begin{equation}
o_{t+1}=O(a_t,h_t),\quad r_t=R(o_{t+1},a_t,h_t)
\tag{5}
\end{equation}
```

环境条件密度退化为 Dirac delta : 
```math
\begin{equation}
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))
\tag{6}
\end{equation}
```

代回原始轨迹积分 : 
```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
\tag{7}
\end{equation}
```

利用 Dirac delta 的基本积分性质 : 
```math
\begin{equation}
\int \delta(x-x_0)f(x)\,dx=f(x_0)
\tag{8}
\end{equation}
```

环境部分坍缩后，只剩动作采样 : 
```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right]
\tag{9}
\end{equation}
```

这意味着 : 
- 决定论环境不再提供路径分支，路径分支只来自策略采样
- 如果策略也决定论，整条路径坍缩成单条路径

### 1.2. 有限时域 (finite horizon)、有限奖励 (reward clipping)
为了防止奖励发散，除了把折扣设为 $\gamma<0$ 之外，还可以
- 限制最大路径长度 $T\le T_{\max}$，路径积分只在有限时间区间上进行 (截断) : 
```math
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right)
\tag{10}
\end{equation}
```

- 对奖励进行裁剪 : 
```math
\begin{equation}
\bar r_t=\operatorname{clip}(r_t,-r_{\max},r_{\max}),\quad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t
\tag{11}
\end{equation}
```

### 1.3. LLM RL 中的路径采样与 hidden 动作

对 LLM RL 来说，外部轨迹步和自回归 token 步需要区分。本文后续约定 :
- $t=0,\ldots,T$ 表示外部轨迹步，例如一次问答、一次 tool call、一次环境交互或一次 agent step
- $i$ 表示 LLM 在外部第 $t$ 步内部自回归生成的 token 位置

设第 $t$ 个外部步对应的 token 区间为 :
```math
\begin{equation}
L_t\le i\le L_{t+1}-1
\tag{12}
\end{equation}
```

于是原来粗粒度的动作 $a_t$ 在 LLM 自回归情形下应理解为一整段生成 :
```math
\begin{equation}
a_t\equiv [a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}],\quad
da_t\equiv \prod_{i=L_t}^{L_{t+1}-1}da_i
\tag{13}
\end{equation}
```

因此自回归结构的完整动作测度为 :
```math
\begin{equation}
\prod_{t=0}^{T}da_t
\equiv
\prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i
\tag{14}
\end{equation}
```

第 $t$ 个外部步开始前的历史仍记为 $h_t$。块内第 $i$ 个 token 的 prefix history 写成 :
```math
\begin{equation}
h_{i,t}\equiv (h_t,a_{L_t},a_{L_t+1},\ldots,a_{i-1}),\quad L_t\le i\le L_{t+1}-1
\tag{15}
\end{equation}
```

这里 $h_t$ 是外部步边界处的历史，$h_{i,t}$ 是同一外部步内部的 token-prefix 历史。

模型前向在每个 token prefix 上给出 logits :
```math
\begin{equation}
z_i^{\mathrm{logit}}=f_\theta(h_{i,t}),\quad
\pi_{\theta,T_{\mathrm{dec}}}(a_i\mid h_{i,t})=
\frac{\exp(z^{\mathrm{logit}}_{i,a_i}/T_{\mathrm{dec}})}
{\sum_{a'}\exp(z^{\mathrm{logit}}_{i,a'}/T_{\mathrm{dec}})}
\tag{16}
\end{equation}
```

于是外部第 $t$ 步的段级策略就是块内 token 策略的自回归乘积 :
```math
\begin{equation}
\pi_\theta(a_t\mid h_t)
\equiv
\prod_{i=L_t}^{L_{t+1}-1}\pi_\theta(a_i\mid h_{i,t})
\tag{16a}
\end{equation}
```

在这个约定下，LLM 轨迹仍然可以保持原来的粗粒度形式 :
```math
\begin{equation}
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
\tag{16b}
\end{equation}
```

但其中每个 $a_t$ 都是一段自回归生成，而不是单个 token。如果要在连续空间做路径采样，可以把 token hidden 当作连续自由度。块内 hidden 写成 :
```math
\begin{equation}
z_i=\operatorname{hidden}_\theta(\operatorname{prefill}(h_{i,t},a_i)),\quad
z_t\equiv[z_{L_t},z_{L_t+1},\ldots,z_{L_{t+1}-1}]
\tag{16c}
\end{equation}
```

连续动作路径可以写成 :
```math
\begin{equation}
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T)
\tag{16d}
\end{equation}
```

其中 $z_t$ 同样表示第 $t$ 个外部步内部的一整段 hidden 变量。若用 $b$ 表示第 $b$ 条样本，则样本轨迹写成 :
```math
\begin{equation}
\tau_b=(a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
\tag{16e}
\end{equation}
```

此时奖励可以写成如下形式 :
```math
\begin{equation}
G[\tau] = \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \Gamma_{i,t} r_{i,t}, \quad \Gamma_{i,t} = \begin{cases} 
\omega_i &, L_t \leq i < L_{t+1}-1 \\
\gamma^t &, i = L_{t+1}
\end{cases}, \quad r_{i,t} = R(a_{i},h_{i,t}, o_{t+1}) + \phi(a_{i},h_{t}, o_{t+1})
\tag{16f}
\end{equation}
```

```math
\begin{equation}
\Gamma_{i,t} = \begin{cases} 
\omega_i &, L_t \leq i < L_{t+1}-1 \\
\gamma^t &, i = L_{t+1}
\end{cases}, \quad r_{i,t} = R(a_{i},h_{i,t}, o_{t+1}) + \phi(a_{i},h_{t}, o_{t+1})
\tag{16f}
\end{equation}
```

---

## 2. 原始 history-based RL 的物理图像解释

### 2.1. 从轨迹积分到路径积分

把原始密度连乘写成路径密度 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)
\tag{17}
\end{equation}
```

于是原始目标可以写成路径积分形式 : 

```math
\begin{equation}
J(\pi)=\int P_{\pi,\mu}[\tau]G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{18}
\end{equation}
```

物理图像是 : 
- $\tau$ 是一条 worldline，$P_{\pi,\mu}[\tau]$ 是路径权重，$G[\tau]$ 是路径收益泛函，$J(\pi)$ 是所有路径的加权平均。

对 LLM 自回归生成，式 (22) 中的 $a_t$ 是一段生成。根据第 1.3 节的约定 :
```math
\begin{equation}
da_t\equiv\prod_{i=L_t}^{L_{t+1}-1}da_i,\quad
\pi(a_t\mid h_t)\equiv\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
\tag{18a}
\end{equation}
```

因此 LLM 版本的路径密度是 :
```math
\begin{equation}
P_{\pi,\mu}^{\mathrm{AR}}[\tau]=
\prod_{t=0}^{T}
\left[
\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
\right]
\mu(o_{t+1},r_t\mid h_t,a_t)
\tag{18b}
\end{equation}
```

对应测度为 :
```math
\begin{equation}
\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\equiv
\prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right]
do_{t+1}\,dr_t
\tag{18c}
\end{equation}
```

也就是说，LLM 自回归结构只是把外部一步 $a_t$ 展开成块内 token 积分。积分完以后，外层路径变量仍然是段级动作 $a_t$。

### 2.2. 基础作用量、基础哈密顿量与玻尔兹曼权重

从路径密度出发 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)
\tag{19}
\end{equation}
```

对路径密度取负对数，得到基础作用量 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=-\log P_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t)
\tag{20}
\end{equation}
```

热力学类比中，玻尔兹曼权重写成 $\exp(-\beta H)$。因此基础作用量与基础哈密顿量的关系是 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau]
\tag{21}
\end{equation}
```

于是原始路径密度可以写成 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\exp(-S_{\pi,\mu}[\tau])=\exp(-\beta H_0[\tau])
\tag{22}
\end{equation}
```

原始 RL 目标变成 : 

```math
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{23}
\end{equation}
```

这里 $G[\tau]$ 是 observable insertion，不是哈密顿量的一部分。

在 LLM 自回归形式下，基础作用量中的策略项展开为 :
```math
\begin{equation}
S_{\pi,\mu}^{\mathrm{AR}}[\tau]=-
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t})
-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
\tag{23a}
\end{equation}
```

并且 :
```math
\begin{equation}
S_{\pi,\mu}^{\mathrm{AR}}[\tau]=\beta H_0^{\mathrm{AR}}[\tau],\quad
P_{\pi,\mu}^{\mathrm{AR}}[\tau]=\exp(-\beta H_0^{\mathrm{AR}}[\tau])
\tag{23b}
\end{equation}
```

这说明 LLM 的 token log-prob 连乘只是 $H_0$ 的策略部分在自回归微步上的展开。

### 2.3. 一维时间方向的单体复杂系统

路径变量可以写成 : 

```math
\begin{equation}
x_t=(a_t,o_{t+1},r_t),\quad \tau=(x_0,x_1,\ldots,x_T)
\tag{24}
\end{equation}
```

它是一维时间方向上的路径系统。复杂性来自 history coupling，因为每一步的策略和环境都依赖完整历史 : 

```math
\begin{equation}
\pi(a_t\mid h_t),\quad \mu(o_{t+1},r_t\mid a_t,h_t)
\tag{25}
\end{equation}
```

对应基础作用量项为 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t)
\tag{26}
\end{equation}
```

物理图像是 : 
- RL path integral 是一维时间方向上的复杂单体系统，内部自由度是 action、observation、reward，复杂性来自这些自由度通过历史 $h_t$ 发生长程时间耦合。

对 LLM 来说，这个一维时间方向具有嵌套结构。外层时间 $t$ 是环境 / 工具 / agent 交互步，内层 token 位置 $i$ 是同一外部步内部的自回归微步。于是自由度可写成 :
```math
\begin{equation}
x_t^{\mathrm{AR}}=(a_{L_t},\ldots,a_{L_{t+1}-1},o_{t+1},r_t)
\tag{26a}
\end{equation}
```

其中 :
```math
\begin{equation}
h_{i,t}=(h_t,a_{L_t},\ldots,a_{i-1})
\tag{26b}
\end{equation}
```

所以 LLM path integral 仍然是一维时间路径积分，只是每个外部格点 $t$ 内部又有一段 token 微结构。token 微结构积分完以后，外层仍然回到段级变量 $a_t$。

### 2.4. 折扣因子与 Laplace 正则化

连续时间回报如果写成 : 

```math
\begin{equation}
G[\tau]=\int_0^\infty r(t)\,dt
\tag{27}
\end{equation}
```

可能发散。加入指数衰减后 : 

```math
\begin{equation}
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt
\tag{28}
\end{equation}
```

离散时间中 : 

```math
\begin{equation}
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t
\tag{29}
\end{equation}
```

令时间步长为 $\Delta t$，对应关系为 : 

```math
\begin{equation}
\gamma=e^{-\lambda\Delta t},\quad \gamma^t=e^{-\lambda t\Delta t}
\tag{30}
\end{equation}
```

如果奖励有界 : 

```math
\begin{equation}
|r_t|\le r_{\max}
\tag{31}
\end{equation}
```

则折扣回报有界 : 

```math
\begin{equation}
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}
\tag{32}
\end{equation}
```

物理图像是 : 
- discount factor 是时间方向上的 Laplace damping，它把无限未来压成有限有效贡献

---

## 3. RL 原定义下的观测量与奖励再采样

从原始 RL 累积回报定义出发，目标是在不改变策略更新器的前提下，扩展观测量 $o_{t+1}$ 和奖励 $r_t$ 的局部采样空间，使模型更好地探索路径。更复杂的做法可以使用真实环境重复采样、world model、reward model、verifier、SMC 或 CEM proposal 等，但是本节不讨论这些复杂的重复采样法，只使用最基本的高斯噪声 proposal。

### 3.1. $o$ 和 $r$ 的高斯噪声再采样 / 重估计

设 $m=1,\ldots,M$ 表示在同一个 $a_t,h_t,o_{t+1},r_t$ 条件下的第 $m$ 次重采样。若使用模拟退火，设 $k=0,1,\ldots,K_{\mathrm{ann}}$ 表示外部退火迭代步，不是路径内部时间步 $t$。第 $k$ 轮使用逆温度 $\beta_k>0$。

先从一维标量变量开始。给定当前值 $x_0$ 和第 $k$ 轮噪声宽度 $s_k>0$，最基本的一维高斯 proposal 定义为 : 

```math
\begin{equation}
q_k(x'\mid x_0)=\frac{1}{\sqrt{2\pi}s_k}\exp\left(-\frac{(x'-x_0)^2}{2s_k^2}\right)
\tag{33}
\end{equation}
```

为了让逆温度控制 proposal 的宽度，定义 : 

```math
\begin{equation}
s_k=\frac{\sigma}{\sqrt{\beta_k}}
\tag{34}
\end{equation}
```

其中 $\sigma>0$ 是人为指定的基础噪声尺度。代入上式后，一维高斯 proposal 变成 : 

```math
\begin{equation}
q_k(x'\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x'-x_0)^2}{2\sigma^2}\right)
\tag{35}
\end{equation}
```

标准高斯噪声 $\xi^{(m)}$ 的密度为 : 

```math
\begin{equation}
p(\xi^{(m)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi^{(m)})^2}{2}\right)
\tag{36}
\end{equation}
```

令第 $m$ 次重采样值为 : 

```math
\begin{equation}
x^{(m,k)}=x_0+\frac{\sigma}{\sqrt{\beta_k}}\xi^{(m)}
\tag{37}
\end{equation}
```

由上式可得 : 

```math
\begin{equation}
\xi^{(m)}=\frac{\sqrt{\beta_k}}{\sigma}(x^{(m,k)}-x_0)
\tag{38}
\end{equation}
```

因此 $x^{(m,k)}$ 的密度正是以 $x_0$ 为中心、宽度为 $\sigma/\sqrt{\beta_k}$ 的高斯 proposal : 

```math
\begin{equation}
q_k(x^{(m,k)}\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x^{(m,k)}-x_0)^2}{2\sigma^2}\right)
\tag{39}
\end{equation}
```

所以小 $\beta_k$ 对应更宽的 proposal，大 $\beta_k$ 对应更窄的 proposal。这就是本节里的模拟退火控制方式。

接下来把一维定义推广到多维观测量。设 $o_{t+1}$ 是 $d_o$ 维观测向量或文本 embedding : 

```math
\begin{equation}
o_{t+1}=(o_{t+1,1},o_{t+1,2},\ldots,o_{t+1,d_o})
\tag{40}
\end{equation}
```

重采样后的观测量为 : 

```math
\begin{equation}
o'_{t+1}=(o'_{t+1,1},o'_{t+1,2},\ldots,o'_{t+1,d_o})
\tag{41}
\end{equation}
```

引入观测噪声协方差矩阵 $\Sigma_o\in\mathbb R^{d_o\times d_o}$。其中 $\Sigma_{o,j\ell}$ 描述第 $j$ 个维度和第 $\ell$ 个维度之间的噪声相关结构；对角项控制单个维度的噪声强度，非对角项控制不同维度之间的相关扰动。多维高斯 proposal 定义为 : 

```math
\begin{equation}
q_k(o'_{t+1}\mid o_{t+1})=
\frac{\beta_k^{d_o/2}}{(2\pi)^{d_o/2}|\Sigma_o|^{1/2}}
\exp\left(-\frac{\beta_k}{2}(o'_{t+1}-o_{t+1})^\top\Sigma_o^{-1}(o'_{t+1}-o_{t+1})\right)
\tag{42}
\end{equation}
```

如果使用矩阵 $L_o$ 生成相关高斯噪声，则先定义 : 

```math
\begin{equation}
\Sigma_o=L_oL_o^\top
\tag{43}
\end{equation}
```

令标准高斯噪声向量为 : 

```math
\begin{equation}
\xi_o^{(m,t)}=(\xi_{o,1}^{(m,t)},\xi_{o,2}^{(m,t)},\ldots,\xi_{o,d_o}^{(m,t)})
\tag{44}
\end{equation}
```

其中每个维度的标准高斯密度为 : 

```math
\begin{equation}
p(\xi_{o,j}^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_{o,j}^{(m,t)})^2}{2}\right),\quad j=1,\ldots,d_o
\tag{45}
\end{equation}
```

于是观测量重采样式为 : 

```math
\begin{equation}
o_{t+1,j}^{(m,k)}=o_{t+1,j}+\frac{1}{\sqrt{\beta_k}}\sum_{\ell=1}^{d_o}L_{o,j\ell}\xi_{o,\ell}^{(m,t)},\quad j=1,\ldots,d_o
\tag{46}
\end{equation}
```

如果不考虑不同维度之间的噪声相关性，可以取 $\Sigma_o=\sigma_o^2I$。这时 $L_o=\sigma_o I$，上式退化成每个维度独立同尺度扰动 : 

```math
\begin{equation}
o_{t+1,j}^{(m,k)}=o_{t+1,j}+\frac{\sigma_o}{\sqrt{\beta_k}}\xi_{o,j}^{(m,t)},\quad j=1,\ldots,d_o
\tag{47}
\end{equation}
```

奖励 $r_t$ 通常是标量。如果直接对标量奖励做高斯扰动，设 $\sigma_r>0$ 是奖励的基础噪声尺度，则有 : 

```math
\begin{equation}
r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\xi_r^{(m,t)}
\tag{48}
\end{equation}
```

其中 : 

```math
\begin{equation}
p(\xi_r^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_r^{(m,t)})^2}{2}\right)
\tag{49}
\end{equation}
```

对应的奖励 proposal 为 : 

```math
\begin{equation}
q_k(r'_t\mid r_t)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_r}\exp\left(-\frac{\beta_k(r'_t-r_t)^2}{2\sigma_r^2}\right)
\tag{50}
\end{equation}
```

但是，如果只是对 $r_t$ 加零中心高斯噪声，然后做样本平均，平均值会回到原来的 $r_t$ 附近。按样本平均的原定义 : 

```math
\begin{equation}
\bar r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\left(\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\right)
\tag{51}
\end{equation}
```

当高斯噪声正负大致抵消时 : 

```math
\begin{equation}
\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\approx 0
\tag{52}
\end{equation}
```

于是 : 

```math
\begin{equation}
\bar r_t^{(k)}\approx r_t
\tag{53}
\end{equation}
```

因此，直接对奖励加噪声主要是鲁棒性扰动。更有意义的做法是先对观测量或观测 hidden 做高斯重采样，再用扰动后的观测重新计算奖励 : 

```math
\begin{equation}
r_t^{(m,k)}=R(o_{t+1}^{(m,k)},a_t,h_t)
\tag{54}
\end{equation}
```

对应的局部奖励估计为 : 

```math
\begin{equation}
\widehat r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}R(o_{t+1}^{(m,k)},a_t,h_t)
\tag{55}
\end{equation}
```

对第 $b$ 条动作路径，第 $m$ 次重采样路径为 : 

```math
\begin{equation}
\tau_b^{(m,k)}=(a_{b,0},o_{b,1}^{(m,k)},r_{b,0}^{(m,k)},\ldots,a_{b,T_b},o_{b,T_b+1}^{(m,k)},r_{b,T_b}^{(m,k)})
\tag{56}
\end{equation}
```

第 $m$ 次重采样路径的回报为 : 

```math
\begin{equation}
G_b^{(m,k)}=\sum_{t=0}^{T_b}\gamma^t r_{b,t}^{(m,k)}
\tag{57}
\end{equation}
```

用样本平均的原定义得到第 $k$ 轮重采样回报估计 : 

```math
\begin{equation}
\widehat G_b^{(k)}=\frac{1}{M}\sum_{m=1}^{M}G_b^{(m,k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_b}\gamma^t r_{b,t}^{(m,k)}
\tag{58}
\end{equation}
```

这就是本节的核心 : 在原始 RL 累积回报定义下，用高斯噪声 proposal 扩展 $o_{t+1}$、$r_t$ 或它们的连续表示，再用样本平均估计更稳定的回报或 advantage。

### 3.2. PPO / GRPO / GSPO 接入

GRPO / GSPO 可以作为本节的细化方向。它们天然有组内样本结构，因此适合对重采样后的回报做组内统计，但这种方法同样可以服务于 PPO，因为 PPO 也只需要 rollout、reward、advantage 和策略更新比率。

设 $K_s$ 表示第 $k$ 轮的组内样本数，$b,c$ 表示组内样本编号，$t$ 表示外部轨迹步。对同一个输入 $x$，第 $k$ 轮采样 $K_s$ 条路径 : 

```math
\begin{equation}
\tau_1^{(k)},\tau_2^{(k)},\ldots,\tau_{K_s}^{(k)}\sim q_k(\tau\mid x)
\tag{59}
\end{equation}
```

每条路径使用高斯噪声重采样得到回报估计 : 

```math
\begin{equation}
\widehat G_b^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_b}\gamma^t r_{b,t}^{(m,k)}
\tag{60}
\end{equation}
```

组内平均与方差为 : 

```math
\begin{equation}
\bar G^{(k)}=\frac{1}{K_s}\sum_{b=1}^{K_s}\widehat G_b^{(k)},\quad (\sigma_G^{(k)})^2=\frac{1}{K_s}\sum_{b=1}^{K_s}(\widehat G_b^{(k)}-\bar G^{(k)})^2
\tag{61}
\end{equation}
```

标准化 advantage 为 : 

```math
\begin{equation}
A_b^{(k)}=\frac{\widehat G_b^{(k)}-\bar G^{(k)}}{\sigma_G^{(k)}+\epsilon}
\tag{62}
\end{equation}
```

如果要对多维观测量和奖励本身做局部统计，可以写成 : 

```math
\begin{equation}
\bar o_{t+1}^{(k)}=\frac{1}{K_sM}\sum_{b=1}^{K_s}\sum_{m=1}^{M}o_{b,t+1}^{(m,k)},\quad \bar r_t^{(k)}=\frac{1}{K_sM}\sum_{b=1}^{K_s}\sum_{m=1}^{M}r_{b,t}^{(m,k)}
\tag{63}
\end{equation}
```

对应的观测量散布矩阵与奖励方差为 : 

```math
\begin{equation}
S_{o,t+1}^{(k)}=\frac{1}{K_sM}\sum_{b=1}^{K_s}\sum_{m=1}^{M}(o_{b,t+1}^{(m,k)}-\bar o_{t+1}^{(k)})(o_{b,t+1}^{(m,k)}-\bar o_{t+1}^{(k)})^\top,
\quad
(s_r^{(k)})^2=\frac{1}{K_sM}\sum_{b=1}^{K_s}\sum_{m=1}^{M}(r_{b,t}^{(m,k)}-\bar r_t^{(k)})^2
\tag{64}
\end{equation}
```

这里的 $S_{o,t+1}^{(k)}$ 和 $s_r^{(k)}$ 是组内统计得到的反馈宽度，不是 proposal 里的噪声矩阵 $\Sigma_o$ 和基础奖励噪声尺度 $\sigma_r$。方差或散布矩阵变小可以解释为反馈估计趋于稳定，或者当前策略进入某个更稳定的局部区域。

GSPO 可以看成把整条生成序列作为采样单位 : 

```math
\begin{equation}
\tau_b=(a_{b,0},o_{b,1},r_{b,0},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
\tag{65}
\end{equation}
```

如果模型动作、工具返回、环境观测和奖励文本都混在同一个 sequence 中，sequence-level 采样会近似把 $a$ 和 $o$ 混在一起采样。更合理的因果分解是 : 

```math
\begin{equation}
q(\tau)=q_a(a_{0:T})q_o(o_{1:T+1}\mid a_{0:T},h_{0:T})q_r(r_{0:T}\mid a_{0:T},o_{1:T+1},h_{0:T})
\tag{66}
\end{equation}
```

如果用 hidden 作为动作 : 

```math
\begin{equation}
q(\tau)=q_z(z_{0:T})q_o(o_{1:T+1}\mid z_{0:T},h_{0:T})q_r(r_{0:T}\mid z_{0:T},o_{1:T+1},h_{0:T})
\tag{67}
\end{equation}
```

意义 : 
- 扩展观测量和奖励反馈空间
- GRPO / GSPO 的组内统计可以更自然地估计 $G$、advantage、$o$ 和 $r$ 的稳定性
- PPO 也可以使用同样的 $o,r$ 再采样数据，只是后续策略更新器不同

### 3.3. 模拟退火

模拟退火在本节中用于调度高斯噪声 proposal。第 $k$ 轮的逆温度 $\beta_k$ 通过第 3.1 节中的高斯 proposal 控制噪声宽度。观测量 proposal 的有效协方差和奖励 proposal 的有效方差为 : 

```math
\begin{equation}
\Sigma_{o,k}=\frac{1}{\beta_k}\Sigma_o,\quad s_{r,k}^2=\frac{\sigma_r^2}{\beta_k}
\tag{68}
\end{equation}
```

其中逆温度递增 : 

```math
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}
\tag{69}
\end{equation}
```

小 $\beta_k$ 对应早期更宽的高斯噪声 proposal，大 $\beta_k$ 对应后期更窄、更保守的反馈估计。这里的 $\beta_k$ 是模拟退火调度参数，用于解释采样宽度和反馈涨落，不是 decoder temperature。

### 3.4. 统计力学解释

从第二节的物理图像看，原始 history-based RL 的路径权重已经可以写成 Boltzmann 权重 : 

```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)=\exp(-\beta H_0[\tau])
\tag{70}
\end{equation}
```

其中 $H_0[\tau]$ 是由原始策略和环境诱导出的基础哈密顿量，$\beta$ 是第二节中 $S=\beta H$ 的共同逆温度。

在 LLM 自回归形式下，式 (70) 的策略部分展开为 :
```math
\begin{equation}
P_{\pi,\mu}^{\mathrm{AR}}[\tau]=
\prod_{t=0}^{T}
\left[
\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
\right]
\mu(o_{t+1},r_t\mid h_t,a_t)
=\exp(-\beta H_0^{\mathrm{AR}}[\tau])
\tag{71}
\end{equation}
```

第 3.1--3.3 节讨论的是 $o_{t+1}$ 和 $r_t$ 的再采样，不改变 token 生成测度本身。因此在 LLM 情形下，$a_t$ 只需要理解成自回归 token 段 :
```math
\begin{equation}
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad
da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
\tag{72}
\end{equation}
```

回报仍然作为路径上的统计观测量 : 

```math
\begin{equation}
G[\tau]=\sum_{t=0}^{T}\gamma^t r_t
\tag{73}
\end{equation}
```

对同一输入采样得到的一组路径可以看成原始路径分布或其 rollout augmentation 下的局部 ensemble : 

```math
\begin{equation}
\tau_b\sim P_{\pi,\mu}[\tau],\quad b=1,\ldots,K_s
\tag{74}
\end{equation}
```

这里用 $b$ 表示样本编号，避免和 token 位置 $i$ 冲突。

组内回报均值和方差为 : 

```math
\begin{equation}
\bar G=\frac{1}{K_s}\sum_{b=1}^{K_s}G[\tau_b],\quad \sigma_G^2=\frac{1}{K_s}\sum_{b=1}^{K_s}(G[\tau_b]-\bar G)^2
\tag{75}
\end{equation}
```

这里 $\bar G$ 是该局部 ensemble 的平均观测量，$\sigma_G^2$ 是回报观测量的涨落强度。

在 3.1--3.3 中，$\beta_k$ 被直接用作第 $k$ 轮模拟退火的逆温度参数。为了和第二节的热力学图像统一，可以把 $\beta_k$ 看成共同逆温度 $\beta$ 与第 $k$ 轮局部重采样强度 $\alpha_{\,\mathrm{res}}^{(k)}$ 的乘积 : 

```math
\begin{equation}
\beta_k=\beta\alpha_{\mathrm{res}}^{(k)}
\tag{76}
\end{equation}
```

其中 $\alpha_{\mathrm{res}}^{(k)}$ 是局部观测量 / 奖励扰动势的强度。后文为简洁起见仍统一写作 $\beta_k$。

给定一条原始 rollout 路径 $\tau$，第 $m$ 次高斯重采样得到的路径写成 : 

```math
\begin{equation}
\tau^{(m,k)}=(a_0,o_1^{(m,k)},r_0^{(m,k)},\ldots,a_T,o_{T+1}^{(m,k)},r_T^{(m,k)})
\tag{77}
\end{equation}
```

如果是 LLM 自回归路径，则每个 $a_t$ 仍然是同一个 token 段，重采样只作用在 $o_{t+1}$、$r_t$ 或它们的连续表示上。

观测量高斯重采样对应一个局部二次哈密顿量。若采用独立同尺度高斯扰动，可以写成 : 

```math
\begin{equation}
H_o(\tau^{(m,k)}\mid\tau)=\sum_{t=0}^{T}\frac{\lVert o_{t+1}^{(m,k)}-o_{t+1}\rVert^2}{2\sigma_o^2}
\tag{78}
\end{equation}
```

如果使用观测噪声协方差矩阵 $\Sigma_o$，则对应写成 : 

```math
\begin{equation}
H_o(\tau^{(m,k)}\mid\tau)=\sum_{t=0}^{T}\frac{1}{2}(o_{t+1}^{(m,k)}-o_{t+1})^\top\Sigma_o^{-1}(o_{t+1}^{(m,k)}-o_{t+1})
\tag{79}
\end{equation}
```

若直接对奖励也做高斯扰动，可以增加奖励扰动哈密顿量 : 

```math
\begin{equation}
H_r(\tau^{(m,k)}\mid\tau)=\sum_{t=0}^{T}\frac{(r_t^{(m,k)}-r_t)^2}{2\sigma_r^2}
\tag{80}
\end{equation}
```

因此，局部重采样哈密顿量可以写成 : 

```math
\begin{equation}
H_{\mathrm{res}}(\tau^{(m,k)}\mid\tau)=H_o(\tau^{(m,k)}\mid\tau)+H_r(\tau^{(m,k)}\mid\tau)
\tag{81}
\end{equation}
```

如果奖励不是直接加噪声，而是由扰动后的观测量重新计算，则可以只保留 $H_o$，并令 : 

```math
\begin{equation}
r_t^{(m,k)}=R(o_{t+1}^{(m,k)},a_t,h_t)
\tag{82}
\end{equation}
```

在 LLM 自回归情形下，$R(o_{t+1}^{(m,k)},a_t,h_t)$ 中的 $a_t$ 是整个 token 段 $[a_{L_t},\ldots,a_{L_{t+1}-1}]$。

这样，原始路径与重采样路径的联合权重可以写成 : 

```math
\begin{equation}
P_k(\tau,\tau^{(m,k)})\propto \exp\left(-\beta H_0[\tau]-\beta_k H_{\mathrm{res}}(\tau^{(m,k)}\mid\tau)\right)
\tag{83}
\end{equation}
```

等价地，增强后的无量纲作用量为 : 

```math
\begin{equation}
S_{\mathrm{aug}}^{(k)}[\tau,\tau^{(m,k)}]=\beta H_0[\tau]+\beta_k H_{\mathrm{res}}(\tau^{(m,k)}\mid\tau)
\tag{84}
\end{equation}
```

这就是第 3 节的统计力学解释 : 原始路径由基础哈密顿量 $H_0[\tau]$ 加权，在这条路径附近的观测量和奖励重采样由局部扰动哈密顿量 $H_{\mathrm{res}}$ 产生热涨落。$\beta_k$ 控制局部涨落强度；小 $\beta_k$ 对应较宽的重采样，较大的 $\beta_k$ 对应更窄、更稳定的反馈估计。

重采样路径上的回报仍然是统计观测量 : 

```math
\begin{equation}
G[\tau^{(m,k)}]=\sum_{t=0}^{T}\gamma^t r_t^{(m,k)}
\tag{85}
\end{equation}
```

第 $k$ 轮的回报估计为 : 

```math
\begin{equation}
\widehat G^{(k)}[\tau]=\frac{1}{M}\sum_{m=1}^{M}G[\tau^{(m,k)}]
\tag{86}
\end{equation}
```

如果 GSPO / GRPO / PPO 更新需要把 step-level score 折算到 token，可以在训练损失中引入 token credit :
```math
\begin{equation}
r_i^{(m,k)}=\omega_i r_t^{(m,k)},\quad L_t\le i\le L_{t+1}-1
\tag{87}
\end{equation}
```

并得到 token 展开的训练观测量。这里的 $\Gamma_i$ 沿用第 1.3 节定义，默认是 $\Gamma_i=\gamma^{t(i)}$ :
```math
\begin{equation}
G_{\mathrm{tok}}[\tau^{(m,k)}]=
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\Gamma_i r_i^{(m,k)}
\tag{88}
\end{equation}
```

但这只是把已有的 step-level 反馈分配到 token log-prob 上，不意味着第 3.1--3.3 节正在对 token 本身做再采样。

模拟退火对应局部重采样强度递增，也就是有效逆温度递增 : 

```math
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}
\tag{89}
\end{equation}
```

从统计力学角度看，早期小 $\beta_k$ 相当于局部高温，重采样路径在原始路径附近有更大的热涨落；后期大 $\beta_k$ 相当于局部低温，重采样路径逐渐收缩到原始观测和奖励附近。这里的回报仍然作为观测量，用于估计 advantage、排序样本、更新 PPO / GRPO / GSPO。

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

本节主线是 LLM RL。为了保持层级清楚，先用非自回归的 LLM RL / agent step 形式说明，再把每个 $a_t$ 展开成 LLM 自回归 token 段。

本节的核心原则是 :
- $H_0[\tau]$ 只来自原始路径权重 $P_{\pi,\mu}[\tau]$。
- $G[\tau]$、KL 惩罚、长度成本、可信度成本等都是目标函数中的 observable / penalty。
- 只有在 Gibbs tilt 采样阶段，整体目标观测量 $F[\tau]$ 才进入指数权重。

### 4.1. 非自回归 LLM RL : 目标观测量与惩罚量

先假设一个外部 step 的动作 $a_t$ 是完整回答片段、tool call、代码 patch 或 agent step，不展开其 token 内部结构。原始路径权重仍然是 :
```math
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)=\exp(-\beta H_0[\tau])
\tag{90}
\end{equation}
```

正则化 RL 目标中的各项先作为 observable / penalty 定义，而不是作为哈密顿量的一部分。

回报 observable 为 :
```math
\begin{equation}
G[\tau]=\sum_{t=0}^{T}\gamma^t r_t
\tag{91}
\end{equation}
```

给定 history $h_t$，exact KL 是两个动作分布之间的 divergence :
```math
\begin{equation}
D_t(h_t)=\int d\tilde a\,\pi(\tilde a\mid h_t)
\log\frac{\pi(\tilde a\mid h_t)}{\pi_{\mathrm{ref}}(\tilde a\mid h_t)}
\tag{92}
\end{equation}
```

这里的积分变量是 $\tilde a$，不是路径里已经采样出来的 $a_t$。如果沿当前路径访问到的 histories 取和，可以定义 :
```math
\begin{equation}
K_{\mathrm{exact\ path}}[\tau]=\sum_{t=0}^{T}D_t(h_t)
\tag{93}
\end{equation}
```

它的目标函数贡献是 $\mathbb E_{\tau\sim P_{\pi,\mu}}[K_{\mathrm{exact\ path}}[\tau]]$。

如果使用 sampled log-ratio，则路径级 KL observable 为 :
```math
\begin{equation}
K_{\mathrm{sample}}[\tau]=\sum_{t=0}^{T}
\log\frac{\pi(a_t\mid h_t)}{\pi_{\mathrm{ref}}(a_t\mid h_t)}
\tag{94}
\end{equation}
```

资源或长度成本只有在能区分路径时才有意义，例如 :
```math
\begin{equation}
N[\tau]=\sum_{t=0}^{T}c_N(a_t,h_t)
\tag{95}
\end{equation}
```

如果 horizon 固定且 $N[\tau]=T+1$ 是常数，则它不会改变路径选择。

使用 sampled log-ratio 时，可以把整体目标观测量写成 :
```math
\begin{equation}
F[\tau]=G[\tau]-\lambda_{\mathrm{KL}}K_{\mathrm{sample}}[\tau]-\lambda_NN[\tau]
\tag{96}
\end{equation}
```

如果使用 exact KL，则把 $K_{\mathrm{sample}}[\tau]$ 替换成 $K_{\mathrm{exact\ path}}[\tau]$，并明确它最终要对 $P_{\pi,\mu}$ 诱导的 history occupancy 取期望。

正则目标是 :
```math
\begin{equation}
J_{\mathrm{reg}}(\pi)=\mathbb E_{\tau\sim P_{\pi,\mu}}[F[\tau]]
\tag{97}
\end{equation}
```

这仍然是 RL 原定义下的目标函数，不是哈密顿量重写。

### 4.2. 基础 Boltzmann 权重

基础 Boltzmann 权重只回顾原始路径分布 :
```math
\begin{equation}
P_{\pi,\mu}[\tau]=\exp(-\beta H_0[\tau])
\tag{98}
\end{equation}
```

对应配分函数为 :
```math
\begin{equation}
Z_0(\beta)=\int \exp(-\beta H_0[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{99}
\end{equation}
```

在原始 rollout 分布已经归一化时，$Z_0(\beta)=1$ 可以看作形式写法。关键是 : $G$、KL、$N$ 都不出现在基础路径权重里。

### 4.3. Gibbs 采样 : 对整体目标观测量做 tilt

如果只是估计原始期望或正则目标，不需要把 observable 放进指数，直接从 $P_{\pi,\mu}$ 采样并求平均即可 :
```math
\begin{equation}
\widehat J_{\mathrm{reg}}(\pi)=\frac{1}{K}\sum_{b=1}^{K}F[\tau_b],\quad \tau_b\sim P_{\pi,\mu}
\tag{100}
\end{equation}
```

Gibbs 采样用于另一个目的 : 构造一个偏向高目标值路径的采样分布。引入 tilt 强度 $\eta$，定义 :
```math
\begin{equation}
q_\eta(\tau\mid\pi,\mu)=
\frac{1}{Z(\beta,\eta,\lambda_{\mathrm{KL}},\lambda_N)}
\exp\left(-\beta H_0[\tau]+\eta F[\tau]\right)
\tag{101}
\end{equation}
```

配分函数为 :
```math
\begin{equation}
Z(\beta,\eta,\lambda_{\mathrm{KL}},\lambda_N)=
\int\exp\left(-\beta H_0[\tau]+\eta F[\tau]\right)
\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{102}
\end{equation}
```

这时可以等价定义 Gibbs 采样分布的有效哈密顿量 :
```math
\begin{equation}
H_{\mathrm{Gibbs}}[\tau]=H_0[\tau]-\frac{\eta}{\beta}F[\tau]
\tag{103}
\end{equation}
```

但 $H_{\mathrm{Gibbs}}$ 只是 tilted sampling distribution 的等价哈密顿量，不是原始 RL 路径哈密顿量。关闭 tilt 时 :
```math
\begin{equation}
\eta=0\quad\Longrightarrow\quad q_\eta(\tau\mid\pi,\mu)=P_{\pi,\mu}[\tau]
\tag{104}
\end{equation}
```

如果从 Gibbs 分布中采样 :
```math
\begin{equation}
\tau_b\sim q_\eta(\tau\mid\pi,\mu),\quad b=1,\ldots,K
\tag{105}
\end{equation}
```

则样本均值估计的是 Gibbs ensemble 下的期望 :
```math
\begin{equation}
\widehat{\mathbb E}_{q_\eta}[F]=\frac{1}{K}\sum_{b=1}^{K}F[\tau_b]
\tag{106}
\end{equation}
```

并且 :
```math
\begin{equation}
\mathbb E_{q_\eta}[F]=\frac{\partial\log Z}{\partial\eta}
\tag{107}
\end{equation}
```

如果要从 Gibbs 样本反推原始路径分布下的期望，需要 importance reweighting :
```math
\begin{equation}
w_b=\exp(-\eta F[\tau_b])
\tag{108}
\end{equation}
```

于是对任意路径 observable $O[\tau]$ 有 :
```math
\begin{equation}
\widehat{\mathbb E}_{P}[O]=\frac{\sum_{b=1}^{K}w_bO[\tau_b]}{\sum_{b=1}^{K}w_b}
\tag{109}
\end{equation}
```

### 4.4. LLM 自回归形式 : token 测度、KL 和目标观测量

现在把非自回归的 $a_t$ 展开成 LLM 自回归 token 段。根据第 1.3 节 :
```math
\begin{equation}
da_t\equiv\prod_{i=L_t}^{L_{t+1}-1}da_i,\quad
a_t\equiv[a_{L_t},\ldots,a_{L_{t+1}-1}]
\tag{110}
\end{equation}
```

块内 prefix 为 :
```math
\begin{equation}
h_{i,t}\equiv(h_t,a_{L_t},\ldots,a_{i-1})
\tag{111}
\end{equation}
```

策略分布展开为 :
```math
\begin{equation}
\pi(a_t\mid h_t)\equiv\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
\tag{112}
\end{equation}
```

基础作用量为 :
```math
\begin{equation}
S_{\pi,\mu}^{\mathrm{AR}}[\tau]=\beta H_0^{\mathrm{AR}}[\tau]=-
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t})
-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
\tag{113}
\end{equation}
```

exact token-prefix KL 为 :
```math
\begin{equation}
D_i(h_{i,t})=\int d\tilde a_i\,\pi(\tilde a_i\mid h_{i,t})
\log\frac{\pi(\tilde a_i\mid h_{i,t})}{\pi_{\mathrm{ref}}(\tilde a_i\mid h_{i,t})}
\tag{114}
\end{equation}
```

sampled token log-ratio 为 :
```math
\begin{equation}
k_i=\log\frac{\pi(a_i\mid h_{i,t})}{\pi_{\mathrm{ref}}(a_i\mid h_{i,t})}
\tag{115}
\end{equation}
```

对应整条路径的 sampled KL observable 是 :
```math
\begin{equation}
K_{\mathrm{sample}}^{\mathrm{AR}}[\tau]=
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}
\log\frac{\pi(a_i\mid h_{i,t})}{\pi_{\mathrm{ref}}(a_i\mid h_{i,t})}
\tag{116}
\end{equation}
```

这也等价于段级 sampled log-ratio :
```math
\begin{equation}
K_{\mathrm{sample}}^{\mathrm{AR}}[\tau]
=\sum_{t=0}^{T}\log\frac{\pi(a_t\mid h_t)}{\pi_{\mathrm{ref}}(a_t\mid h_t)}
\tag{117}
\end{equation}
```

但式 (117) 成立的前提是 $\pi(a_t\mid h_t)$ 被理解为式 (112) 的自回归乘积。

资源成本也应按 token 或真实资源写成 :
```math
\begin{equation}
N^{\mathrm{AR}}[\tau]=\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}c_i
\tag{118}
\end{equation}
```

若 step-level 分数或 GSPO / GRPO advantage 需要折算到 token，可以写成 :
```math
\begin{equation}
r_i=\omega_i R_t(h_t,a_t,o_{t+1}),\quad L_t\le i\le L_{t+1}-1
\tag{119}
\end{equation}
```

于是 token 展开的整体目标观测量为。这里 $\Gamma_i$ 沿用第 1.3 节定义，默认是 $\Gamma_i=\gamma^{t(i)}$ :
```math
\begin{equation}
F^{\mathrm{AR}}[\tau]=
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}
\left(\Gamma_i r_i-\lambda_{\mathrm{KL}}k_i-\lambda_Nc_i\right)
\tag{120}
\end{equation}
```

对应 Gibbs tilted 分布为 :
```math
\begin{equation}
q_\eta^{\mathrm{AR}}(\tau)=
\frac{1}{Z_{\mathrm{AR}}}
\exp\left(-\beta H_0^{\mathrm{AR}}[\tau]+\eta F^{\mathrm{AR}}[\tau]\right)
\tag{121}
\end{equation}
```

### 4.5. MCMC 路径采样

若使用模拟退火，设 $k=0,1,\ldots,K_{\mathrm{ann}}$ 表示外部采样器的退火迭代步，$\beta_k$ 是第 $k$ 轮的逆温度，$\eta_k$ 是目标 tilt 强度。目标路径分布为 :
```math
\begin{equation}
q_k(\tau)=\frac{1}{Z_k}\exp\left(-\beta_k H_0[\tau]+\eta_kF[\tau]\right)
\tag{122}
\end{equation}
```

从当前路径生成候选路径 :
```math
\begin{equation}
\tau'\sim q_{\mathrm{prop}}(\tau'\mid\tau)
\tag{123}
\end{equation}
```

Metropolis-Hastings 接受率为 :
```math
\begin{equation}
A_k(\tau\rightarrow\tau')=
\min\left(1,
\exp\left[-\beta_k(H_0[\tau']-H_0[\tau])+\eta_k(F[\tau']-F[\tau])\right]
\frac{q_{\mathrm{prop}}(\tau\mid\tau')}{q_{\mathrm{prop}}(\tau'\mid\tau)}
\right)
\tag{124}
\end{equation}
```

这说明路径选择由两部分共同决定 : 原始路径概率通过 $H_0$ 进入，正则化后的目标值通过 $F$ 的 Gibbs tilt 进入。高 $F$ 路径更容易被接受，但这是 Gibbs 采样分布的选择规则，不是把 $F$ 放进了原始哈密顿量。

在 LLM 自回归版本中，只需把 $H_0$ 和 $F$ 替换为 $H_0^{\mathrm{AR}}$ 和 $F^{\mathrm{AR}}$。

### 4.6. Langevin 路径采样与逆温度退火

Langevin 更适合在连续自由度上做，尤其是 hidden action $z_i$ 或 hidden block $z_t$。第 $k$ 轮退火迭代的无量纲 Gibbs 采样作用量为 :
```math
\begin{equation}
S_{\mathrm{Gibbs}}^{(k)}[\tau]=\beta_k H_0[\tau]-\eta_kF[\tau]
\tag{125}
\end{equation}
```

对连续 hidden 片段 $z_{u:v}$ 做 Langevin 更新 :
```math
\begin{equation}
z_{u:v}^{(k+1)}=z_{u:v}^{(k)}-
\epsilon\nabla_{z_{u:v}}S_{\mathrm{Gibbs}}^{(k)}[\tau^{(k)}]
+\sqrt{2\epsilon}\,\xi_k
\tag{126}
\end{equation}
```

其中 :
```math
\begin{equation}
\xi_k\sim\mathcal N(0,I)
\tag{127}
\end{equation}
```

模拟退火通过递增逆温度实现 :
```math
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}
\tag{128}
\end{equation}
```

小 $\beta_k$ 对应高温探索，较多低原始概率或高采样作用量路径仍能保留。大 $\beta_k$ 对应低温收敛，路径分布逐渐集中到 $-\beta_kH_0+\eta_kF$ 较大的区域。

第三节也可以使用模拟退火，但它调度的是 $q_o^{(k)}$ 和 $q_r^{(k)}$ 这类观测量与奖励 proposal；本节的模拟退火调度的是 Gibbs tilted 路径分布。两者都可以防止过早陷入局部最优，但数学对象不同。

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

## 5. 第三节与第四节的关系

两条路线都用于扩大采样路径，但数学对象不同。

第三节是 RL estimator / rollout augmentation。它从原始累积回报定义出发，对 $o_{t+1}$ 和 $r_t$ 做再采样 : 

```math
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t)
\tag{129}
\end{equation}
```

在 LLM 自回归形式中，$a_t$ 是第 $t$ 个外部步的整段生成 :
```math
\begin{equation}
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}]
\tag{130}
\end{equation}
```

第三节得到的回报估计可以交给 PPO / GRPO / GSPO : 

```math
\begin{equation}
\widehat G_b^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_b}\gamma^t r_{b,t}^{(m,k)}
\tag{131}
\end{equation}
```

如果更新器需要 token-level loss，则把 step-level score 或 advantage 分配到 token 上即可。

第四节是路径空间 Gibbs 采样。它不把惩罚项预先塞进基础哈密顿量，而是先定义整体目标观测量 $F[\tau]$，再做 tilt :
```math
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp\left(-\beta_kH_0[\tau]+\eta_kF[\tau]\right)
\tag{132}
\end{equation}
```

因此，第三节重点是改进 rollout 数据和回报估计；第四节使用 MCMC / Langevin / 退火直接采样 Gibbs tilted 路径。统计力学图像可以帮助解释两者中的路径选择和退火行为，但二者的采样对象不同。

---

## 6. 最终总公式

基础作用量和基础哈密顿量满足 : 

```math
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t)
\tag{133}
\end{equation}
```

原始 RL 目标为 : 

```math
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{134}
\end{equation}
```

第三节使用原始 RL 累积回报定义，对观测量和奖励进行再采样 : 

```math
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t)
\tag{135}
\end{equation}
```

第三节得到的回报估计可以交给 PPO / GRPO / GSPO : 

```math
\begin{equation}
\widehat G_b^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_b}\gamma^t r_{b,t}^{(m,k)}
\tag{136}
\end{equation}
```

第四节的正则化目标先写成整体路径观测量 :
```math
\begin{equation}
F[\tau]=G[\tau]-\lambda_{\mathrm{KL}}K[\tau]-\lambda_NN[\tau]
\tag{137}
\end{equation}
```

其中 $K[\tau]$ 必须明确是 sampled log-ratio 路径 observable，或者是沿路径访问到的 exact history KL 之和。sampled 版本为 :
```math
\begin{equation}
K_{\mathrm{sample}}[\tau]=\sum_{t=0}^{T}\log\frac{\pi(a_t\mid h_t)}{\pi_{\mathrm{ref}}(a_t\mid h_t)}
\tag{138}
\end{equation}
```

exact history KL 版本为 :
```math
\begin{equation}
K_{\mathrm{exact\ path}}[\tau]=\sum_{t=0}^{T}D_t(h_t),\quad
D_t(h_t)=\int d\tilde a\,\pi(\tilde a\mid h_t)
\log\frac{\pi(\tilde a\mid h_t)}{\pi_{\mathrm{ref}}(\tilde a\mid h_t)}
\tag{139}
\end{equation}
```

正则目标是 :
```math
\begin{equation}
J_{\mathrm{reg}}(\pi)=\mathbb E_{\tau\sim P_{\pi,\mu}}[F[\tau]]
\tag{140}
\end{equation}
```

Gibbs tilted 路径采样分布为 :
```math
\begin{equation}
q_\eta(\tau\mid\pi,\mu)=
\frac{1}{Z(\beta,\eta,\lambda_{\mathrm{KL}},\lambda_N)}
\exp\left(-\beta H_0[\tau]+\eta F[\tau]\right)
\tag{141}
\end{equation}
```

对应的 Gibbs 等价哈密顿量为 :
```math
\begin{equation}
H_{\mathrm{Gibbs}}[\tau]=H_0[\tau]-\frac{\eta}{\beta}F[\tau]
\tag{142}
\end{equation}
```

注意 $H_{\mathrm{Gibbs}}$ 只是 tilted sampling distribution 的等价写法，不是原始路径哈密顿量。

LLM 自回归版本在原有公式上增加 :
```math
\begin{equation}
da_t\equiv\prod_{i=L_t}^{L_{t+1}-1}da_i,\quad
a_t\equiv[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad
h_{i,t}\equiv(h_t,a_{L_t},\ldots,a_{i-1})
\tag{143}
\end{equation}
```

```math
\begin{equation}
\pi(a_t\mid h_t)\equiv\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
\tag{144}
\end{equation}
```

于是自回归基础作用量为 :
```math
\begin{equation}
S_{\pi,\mu}^{\mathrm{AR}}[\tau]=\beta H_0^{\mathrm{AR}}[\tau]=-
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t})
-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
\tag{145}
\end{equation}
```

LLM 自回归 sampled KL 为 :
```math
\begin{equation}
K_{\mathrm{sample}}^{\mathrm{AR}}[\tau]=
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}
\log\frac{\pi(a_i\mid h_{i,t})}{\pi_{\mathrm{ref}}(a_i\mid h_{i,t})}
\tag{146}
\end{equation}
```

如果将 reward / score / advantage 折算到 token，定义 :
```math
\begin{equation}
F^{\mathrm{AR}}[\tau]=
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}
\left(\Gamma_i r_i-\lambda_{\mathrm{KL}}k_i-\lambda_Nc_i\right)
\tag{147}
\end{equation}
```

其中 $\Gamma_i$ 沿用第 1.3 节定义，默认是 $\Gamma_i=\gamma^{t(i)}$；若不使用时间折扣则取 $\Gamma_i=1$。$k_i$ 为 sampled token log-ratio :
```math
\begin{equation}
k_i=\log\frac{\pi(a_i\mid h_{i,t})}{\pi_{\mathrm{ref}}(a_i\mid h_{i,t})}
\tag{148}
\end{equation}
```

最终 LLM 自回归 Gibbs tilted 分布为 :
```math
\begin{equation}
q_\eta^{\mathrm{AR}}(\tau)=
\frac{1}{Z_{\mathrm{AR}}(\beta,\eta,\lambda_{\mathrm{KL}},\lambda_N)}
\exp\left(-\beta H_0^{\mathrm{AR}}[\tau]+\eta F^{\mathrm{AR}}[\tau]\right)
\tag{149}
\end{equation}
```

路径采样流程可以概括为 : 

```math
\begin{align*}
&~ \text{rollout / proposal} \\
\rightarrow &~ \text{第 3 节: resample }o,r\text{ and estimate }G\text{ or token credit} \\
\text{or} &~ \text{第 4 节: build }F[\tau]\text{ and sample with }e^{-\beta H_0[\tau]+\eta F[\tau]} \\
\rightarrow &~ \text{PPO / GRPO / GSPO update or MCMC / Langevin search} \\
\rightarrow &~ \text{distill back to }\pi_\theta
\end{align*}
```

物理图像 : 
- history-based RL 是一维时间路径积分。
- $a,o,r$ 是路径内部自由度。
- 在 LLM 中，$a_t$ 是一段自回归 token，$i$ 才是 token 位置。
- hidden $z_i$ 或 $z_t$ 可以作为连续动作自由度。
- 原始期望是 $\int e^{-\beta H_0}G$。
- 第三节在原始 RL 估计器上扩展 $o,r$ 反馈空间。
- 第四节先组织整体目标 observable $F$，再在 Gibbs 采样阶段通过 $e^{\eta F[\tau]}$ 做 tilt。

---

# Appendix A : 可选观测可信度与奖励可信度

如果世界模型或奖励模型本身需要可信度惩罚，这些项也应先作为目标 observable / penalty 定义，而不是直接塞进基础哈密顿量。观测可信度项可以写成 : 

```math
\begin{equation}
C_o[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_o(o_{t+1}\mid a_t,h_t)
\tag{A1}
\end{equation}
```

奖励可信度项可以写成 : 

```math
\begin{equation}
C_r[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_r(r_t\mid o_{t+1},a_t,h_t)
\tag{A2}
\end{equation}
```

若使用这些可选惩罚，应该扩展整体目标观测量 :
```math
\begin{equation}
F_{\mathrm{cred}}[\tau]=
F[\tau]-\rho_oC_o[\tau]-\rho_rC_r[\tau]
\tag{A3}
\end{equation}
```

对应 Gibbs tilted 采样分布为 :
```math
\begin{equation}
q_\eta(\tau)=
\frac{1}{Z_{\mathrm{cred}}}
\exp\left(-\beta H_0[\tau]+\eta F_{\mathrm{cred}}[\tau]\right)
\tag{A4}
\end{equation}
```

这里 $C_o$ 和 $C_r$ 没有进入原始 $H_0$。只有当 $\hat\mu_o$ 或 $\hat\mu_r$ 被明确改造成新的生成 / proposal 路径分布时，它们的负对数才会成为那个新 proposal 的基础作用量的一部分；这和把可信度惩罚作为 RL 目标项是两件事。

在 LLM 自回归形式下，只需使用 :
```math
\begin{equation}
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad
\pi(a_t\mid h_t)=\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
\tag{A5}
\end{equation}
```

因此 $C_o$ 和 $C_r$ 仍然是 step-level 观测 / 奖励可信度惩罚；如果需要 token-level 分配，可以再把 $C_o$、$C_r$ 按权重 $\omega_i$ 分配到 token loss 上。

---

# Appendix B : 时间方向重整化

因为这是一个一维时间路径积分，所以重整化主要沿时间方向做。非自回归版本中，设 $\ell$ 表示宏观时间块编号，把每 $b$ 个微观 action 合并成一个宏观块 : 

```math
\begin{equation}
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1})
\tag{B1}
\end{equation}
```

多层压缩后 : 

```math
\begin{equation}
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}
\tag{B2}
\end{equation}
```

在路径积分层面，微观路径到宏观路径的映射为 $\bar\tau=\mathcal C(\tau)$。如果讨论原始路径分布，则宏观基础作用量由积分掉微观自由度得到 : 

```math
\begin{equation}
\exp(-S_0[\bar\tau])=
\int_{\mathcal C(\tau)=\bar\tau}
\exp(-S_0[\tau])
\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{B3}
\end{equation}
```

其中 $S_0[\tau]=\beta H_0[\tau]$。如果讨论 Gibbs tilted 采样分布，则应该压缩的是 :
```math
\begin{equation}
S_{\mathrm{Gibbs}}[\tau]=\beta H_0[\tau]-\eta F[\tau]
\tag{B4}
\end{equation}
```

对应 :
```math
\begin{equation}
\exp(-\bar S_{\mathrm{Gibbs}}[\bar\tau])=
\int_{\mathcal C(\tau)=\bar\tau}
\exp(-S_{\mathrm{Gibbs}}[\tau])
\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
\tag{B5}
\end{equation}
```

这区分了两种对象 : 原始路径分布的 temporal RG 和 Gibbs tilted 采样分布的 temporal RG。不要把目标惩罚项提前混入 $S_0$。

LLM 自回归版本有两层时间结构。外层是环境 / agent step $t$，内层是 token 位置 $i$。第 $t$ 个外部 step 的 token 段为 :
```math
\begin{equation}
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad
da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
\tag{B6}
\end{equation}
```

因此最基本的 LLM 时间粗粒化已经是把 token 微步积分成段级动作 $a_t$ :
```math
\begin{equation}
\prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i
=\prod_{t=0}^{T}da_t
\tag{B7}
\end{equation}
```

如果还要在一个外部 step 内部继续做 token-block 压缩，令第 $t$ 个外部 step 内部的第 $\ell$ 个 token 块为 :
```math
\begin{equation}
B_{t,\ell}=C_\phi(a_{L_t+\ell b},a_{L_t+\ell b+1},\ldots,a_{L_t+(\ell+1)b-1})
\tag{B8}
\end{equation}
```

则 LLM 自回归基础作用量的粗粒化写成 :
```math
\begin{equation}
\exp(-S_0^{\mathrm{AR}}[\bar\tau])=
\int_{\mathcal C(\tau)=\bar\tau}
\exp(-S_0^{\mathrm{AR}}[\tau])
\prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i\,do_{t+1}\,dr_t
\tag{B9}
\end{equation}
```

其中 :
```math
\begin{equation}
S_0^{\mathrm{AR}}[\tau]=-
\sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t})
-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
\tag{B10}
\end{equation}
```

如果做 LLM Gibbs tilted 路径采样，则需要压缩 :
```math
\begin{equation}
S_{\mathrm{Gibbs}}^{\mathrm{AR}}[\tau]=
\beta H_0^{\mathrm{AR}}[\tau]-\eta F^{\mathrm{AR}}[\tau]
\tag{B11}
\end{equation}
```

而不是只压缩 $F^{\mathrm{AR}}$ 或只压缩 KL 惩罚。

如果压缩块内部的有效奇异值谱快速衰减，则截断安全；如果谱近似平直，则硬截断会损失大量信息 : 

```math
\begin{equation}
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ 的截断会造成强信息损失}
\tag{B12}
\end{equation}
```

物理图像是 : 
- temporal RG 可以把长时间路径压缩成宏观路径，但模型必须支持 compressed token / hidden macro-action，否则压缩只是摘要，不是有效自由度。
- LLM 的第一层粗粒化是 token $i$ 到外部 step $t$ 的段级动作 $a_t$。
- 如果进一步粗粒化，应区分原始路径作用量 $S_0$ 和 Gibbs tilted 作用量 $S_{\mathrm{Gibbs}}$。
