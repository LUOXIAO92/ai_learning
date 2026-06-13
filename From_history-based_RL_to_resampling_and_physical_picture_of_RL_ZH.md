# 从 history-based RL 到重采样以及 RL 的物理图像

## 0. 目标

这里讨论的是一种**物理图像解释**，不是把 RL 严格重写成场论。本文是讨论扩大采样路径的物理方法，主线有两条 : 
- **再采样** : 在 RL 原始累积回报定义下，对观测量 $o_{t+1}$ 和奖励 $r_t$ 做再采样 / 重估计，用于改进 rollout、回报估计、优势(advantage) 估计和 PPO / GRPO / GSPO 更新
- **路径积分** : 在路径积分图像下，引入有效哈密顿量、Boltzmann 权重、Gibbs 采样、MCMC / Langevin 和逆温度退火，用于直接在路径空间中扩大采样并寻找低哈密顿量路径

RL 与物理的共同基础是 : 
- history-based RL 可以写成一维时间方向上的路径积分
- 原始期望中的回报 $G[\tau]$ 是路径上的观测量

---

## 1. 原始 history-based RL 轨迹积分

从最原始的形式开始。交互历史为 : 
```math
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\cdots,a_{t-1},r_{t-1},o_t)
```

策略为 $\pi(a_t\mid h_t)$，环境条件密度为 $\mu(o_{t+1},r_t\mid a_t,h_t)$。有限时间 $T$ 内，整条轨迹的期望回报可以写成 : 
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
```

此时整条路径可写成 : 
```math
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
```

路径回报则为 : 
```math
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s
```

于是原始 RL 目标就是对所有可能路径做加权积分，每条路径的权重由策略和环境共同给出，每条路径的值由折扣回报 $G[\tau]$ 给出。

### 1.1. 决定论环境下的 Dirac delta 退化

如果环境是决定论的，则给定 $a_t,h_t$ 后，下一步观测和奖励由确定函数给出 : 
```math
o_{t+1}=O(a_t,h_t),\quad r_t=R(o_{t+1},a_t,h_t)
```

环境条件密度退化为 Dirac delta : 
```math
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))
```

代回原始轨迹积分 : 
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
```

利用 Dirac delta 的基本积分性质 : 
```math
\int \delta(x-x_0)f(x)\,dx=f(x_0)
```

环境部分坍缩后，只剩动作采样 : 
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right]
```

这意味着 : 
- 决定论环境不再提供路径分支，路径分支只来自策略采样
- 如果策略也决定论，整条路径坍缩成单条路径

### 1.2. 有限时域 (finite horizon)、有限奖励 (reward clipping)
为了防止奖励发散，除了把折扣设为 $0<\gamma\leq 1$ 之外，还可以
- 限制最大路径长度 $T\le T_{\max}$，路径积分只在有限时间区间上进行 (截断) : 
```math
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right)
```

- 对奖励进行裁剪 : 
```math
\bar r_t=\mathrm{clip}(r_t,-r_{\max},r_{\max}),\quad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t
```

### 1.3. LLM RL 中的路径采样与隐状态动作

对 LLM RL 来说，外部轨迹步和自回归 token 步需要区分。本文后续约定 :
- $t=0,\ldots,T$ 表示外部轨迹步，例如一次问答、一次 tool call、一次环境交互或一次 agent step
- $i$ 表示 LLM 在外部第 $t$ 步内部自回归生成的 token 位置

设第 $t$ 个外部步对应的 token 区间为 $L_t\le i\le L_{t+1}-1$ ，并且已有外部步历史 $h_{t} = [h_{t-1}, a_{L_{t-1}}, \cdots a_{L_t-1}, o_{t}, r_{t-1}] $。同一外部步区间的 token 前缀历史记为 $h_{i,t} \equiv \text{concat}(h_t, [a_{L_t},a_{L_t+1},\ldots,a_{i-1}])$ 。模型在这个区间的 token 生成步骤如下:
- 预测第 $i$ 个 token (的logit):
```math
z_i^{\mathrm{logit}} = f_\theta(h_{i,t})
```
- 采样 (在这一步可使用top-p, top-k等采样策略):
```math
\pi_{\theta,T_{\mathrm{dec}}}(a_i\mid h_{i,t}) =  \frac{\exp(z^{\mathrm{logit}}_{i,a_i}/T_{\mathrm{dec}})} {\sum_{a'}\exp(z^{\mathrm{logit}}_{i,a'}/T_{\mathrm{dec}})}
```
- 拼接: 按照上面给的 $h_{i,t}$ 的定义拼接新生成的 token


于是原来粗粒度的动作 $a_t$ 在 LLM 自回归情形下应该理解为一整段生成，token 序列服从以下的联合分布 :
```math
\pi_\theta(a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}|h_t) = \prod_{i=L_t}^{L_{t+1}-1} \pi_\theta(a_i|h_{i,t})
```
因此粗粒度动作 $a_t$ 可表示为对该联合分布做采样 :
```math
a_t \sim \prod_{i=L_{t}}^{L_{t+1}-1} \pi_\theta(\cdot|h_{i,t}), \quad a_t = [a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}]
```

因此积分形式下的自回归结构的完整动作测度变为如下形式 :
```math
\prod_{t=0}^{T} da_t~ \pi_\theta(a_t|h_{t})  \equiv \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1} da_i~ \pi_\theta(a_i|h_{i,t}) 
```

当 LLM 生成一条完整的 token 序列 $a_t$ 并且跟环境产生相互作用得到观测量 $o_{t+1}$ 以及回报 $r_t$ 时，我们仍可以把历史写成 $h_{t+1} = [h_t, (a_t, o_{t+1}, r_t)]$ ，在这个约定下，LLM 轨迹仍然可以保持原来的粗粒度形式 :
```math
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
```

$a_t$ 是一段自回归生成的 token 序列，不是单个 token。LLM 自回归结构只是把外部一步 $a_t$ 展开成块内 token 积分；积分完以后，外层路径变量仍然是阶段级的动作 $a_t$。如果要在连续空间做路径采样，可以把 token 隐状态当作连续自由度。块内隐状态写成 :
```math
z_t\equiv[z_{L_t},z_{L_t+1},\ldots,z_{L_{t+1}-1}],\quad z_i \in \mathbb{R}^{d_\mathrm{model}}
```

连续动作路径可以写成 :
```math
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T)
```

其中 $z_t$ 同样表示第 $t$ 个外部步内部的一整段隐状态变量。若用 $b$ 表示第 $b$ 条样本，则样本轨迹写成 :
```math
\tau_b=(a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
```

此时奖励可以写成如下形式 :
```math
\begin{aligned}
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t}
\end{aligned}
```
$\omega^{i-L_t}$ 和 $\gamma^t$ 分别为 token 级和任务步数级折扣， $R(h_{i,t}, a_{i})$ 为 token 级回报，而 $\phi(h_{i,t}, a_{i}, o_{i+1})$ 则可以作为阶段性回报，当 $t=T$ 时可化为终局回报。

---

## 2. 原始 history-based RL 的物理图像解释

### 2.1. 从轨迹积分到路径积分

首先把原始概率密度连乘写成路径密度 : 
```math
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T} \pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)
```

于是累计回报的期望值 $J$ 可以写成泛函积分形式 : 
```math
J(\pi)=\int \mathcal{D}\tau ~ P_{\pi,\mu}[\tau]G[\tau],\quad \mathcal{D}\tau = \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
```

在这里，我们进一步定义作用量 $S_{\pi,\mu}[\tau]$ :
```math
S_{\pi,\mu}[\tau] = - \log P_{\pi,\mu}[\tau] = - \sum_{t=0}^T \left[ \log \pi(a_t|h_t) + \log \mu(o_{t+1},r_t | h_t, a_t) \right]
```
原始的轨迹积分可以用路径积分来描述 :
```math
J(\pi)=\int \mathcal{D}\tau ~ e^{-S_{\pi,\mu}[\tau]} G[\tau]
```

对 LLM 自回归生成，根据第 1.3 节的约定 :
```math
da_t\equiv\prod_{i=L_t}^{L_{t+1}-1}da_i,\quad
\pi(a_t\mid h_t)\equiv\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
```

因此 LLM 版本的路径密度是 :
```math
P^{AR}_{\pi,\mu}[\tau] = \prod_{t=0}^{T} \left[ \prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t}) \right]
\mu(o_{t+1},r_t\mid h_t,a_t)
```

作用量为 : 
```math
S^{AR}_{\pi,\mu}[\tau] = - \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \left( \log \pi(a_i|h_{i,t}) + \delta_{i,L_{t+1}-1} \log \mu(o_{t+1},r_t | h_t, a_t) \right)
```

时刻 $t$ 的传播子就是 :
```math
\begin{aligned}
f(h_{t+1}, h_t) &= \left[\prod_{i=L_t}^{L_{t+1}-1} \pi(a_i|h_{i,t})\right]\mu(o_{t+1},r_t|h_t,a_t)\\
\Delta h_t &= [a_t,o_{t+1},r_t] ,~ h_{t+1} = \text{concat}(h_t, \Delta h_t)
\end{aligned}
```

对应测度为 :
```math
\mathcal{D}\tau = \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \equiv \prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right] do_{t+1}\,dr_t
```

因此，强化学习可以在路径积分表象下作出以下物理解释
- $a_t,o_t,r_t$: 一维时间路径上场的自由度，这里的场就是标量场 (动作场，观测场以及回报场)
- $\mathcal{D}\tau$: 所有格点上的场的测度
- $G[\tau]$: 物理观测量
- $f(h_{t+1},h_t)$: 传播子，但注意由于history-based RL的定义，这里的变化量为序列 $\Delta h_t = h_{t+1}-h_{t}\equiv [a_t, o_{t+1}, r_t]$，而不是场的微小变化量
- $e^{-S_{\pi,\mu}[\tau]}$: 路径积分的权重，其中 $S_{\pi,\mu}$ 为作用量
- History-based RL 可以看成一个拥有 $a,o,r$ 三个自由度的复杂的一维单粒子系统，其复杂性体现在基于历史轨迹 $h_t$ 的长程耦合。其耦合并不是简单的 $w_{o_0,a_0,o_1,r_0,\cdots,a_t,o_{t+1},r_t}$ 系数，而是由策略 $\pi$ 以及环境测度 $\mu$ 决定的。
- 虽然我们把路径密度的负对数 $- \log P_{\pi,\mu}$ 定义为作用量，但是考虑到我们能通过逆温度 $\beta$ 来控制全体耦合的强度，因此可以对作用量进行再定义
```math
S_{\pi,\mu}[\tau] = \beta H_{\pi,\mu}[\tau],\quad H_{\pi,\mu}[\tau] \equiv - \log P_{\pi,\mu}
```

### 2.2. 折扣因子与 Laplace 正则化

连续时间回报如果写成 : 

```math
G[\tau]=\int_0^\infty r(t)\,dt
```

可能发散。加入指数衰减后 : 

```math
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt
```

离散时间中 : 

```math
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t
```

令时间步长为 $\Delta t$，对应关系为 $\gamma=e^{-\lambda\Delta t}, ~\gamma^t=e^{-\lambda t\Delta t}$，如果奖励有界 $|r_t|\le r_{\max}$，则折扣回报有界 
```math
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}
```

---

## 3. RL 原定义下的观测量与奖励再采样

从原始 RL 累积回报定义出发，我们可以在不改变策略更新器的前提下，使用重采样的方法来扩展观测量 $o_{t+1}$ 和奖励 $r_t$ 的局部采样空间，使模型更好地探索路径。由环境测度的定义，我们可以把隐变量考虑为对测量以及奖励产生了随机的扰动，这里就拥有了一个随机性。但是在 LLM RL 中，通常情况下观测值以及奖励都是确定的，因此路径的探索取决于策略模型，探索能力受到限制，所以我们更需要为观测量和奖励引入随机扰动，来让 LLM 拥有更大的探索空间。
- 注意: 某些任务，特别是工具调用等，我们很难或者无法对其观测量 (也就是返回值) 加入随机扰动

更复杂的做法可以使用真实环境重复采样、world model、reward model、verifier、SMC 或 CEM 提议分布等，但是本节不讨论这些复杂的重复采样法，只使用最基本的高斯噪声提议分布。

### 3.1. 决定论条件下的观测量 $o$ 以及奖励 $r$ 的近似
在第一节，我们讨论了使用 $\delta$ 函数时，随机观测以及奖励可以退化成决定论的取值 (也就是由确定性的物理、数学建模得出)。在这里，我们可以通过 $\delta$ 函数的定义
```math
\delta(x-x_0) = \lim_{\sigma\rightarrow 0}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( -\frac{(x-x_0)^2}{2\sigma^2} \right)
```
先取一个极小的 $\sigma$ 把决定论扩张并且近似成“概率分布”。因此我们可以用极小方差的提议分布近似零方差 Dirac delta 函数，此时 $o_{t+1}, r_t$ 近似地服从以下分布
```math
o_{t+1},r_t \sim \mu_\sigma(\cdot|h_t,a_t) \quad\text{or}\quad o_{t+1}\sim \mu_{\sigma_O}(\cdot|h_t,a_t),~ r_t \sim \mu_{\sigma_R}(\cdot|h_t, a_t, o_{t+1})
```
在这里我们选择高斯分布作为近似，并且使用决定论观测量 $o$ 以及奖励 $r$ 作为期望，此时随机过程 $o\rightarrow o', r\rightarrow r'$ 的状态转移概率为
```math
q(o'|o) = \frac{1}{\sqrt{2\pi\sigma^2_O}}\exp\left( -\frac{(o'-o)^2}{2\sigma^2_O} \right) ,\quad q(r'|r) = \frac{1}{\sqrt{2\pi\sigma^2_R}}\exp\left( -\frac{(r'-r)^2}{2\sigma^2_R} \right)
```

通过这种近似，我们可以很自然地导入模拟退火法。在这里我们先考虑**标量**的情况，令 $\beta$为逆温度，此时可得
```math
q_{\beta} (x'|x) = \sqrt{\frac{\beta}{2\pi\sigma^2}} \exp\left( - \beta \frac{(x'-x)^2}{2\sigma^2} \right)
```
在这里，等效方差为 $\sigma^2_{\text{eff}}(\beta) = \sigma^2 / \beta$，因此在模拟退火中，我们可以通过调节逆温度来决定采样的幅度。我们把原始的近似高斯分布改写成正态分布
```math
q(\xi) = \frac{1}{\sqrt{2\pi}} \exp\left(- \frac{\xi^2}{2} \right), \quad \xi^2 = \beta\frac{(x'-x)^2}{\sigma^2}
```

此时我们可以做高斯采样，新的变量为 $x' = x + \xi / \sqrt{\beta} ,~ \xi \sim \mathcal{N}(0, \sigma^2)$。采样顺序为: 1. 生成一个服从标准正态分布的随机数; 2. 计算新的 $x'$。

下面我们继续把以上讨论扩张为多维高斯分布的情况，此时 $\boldsymbol{x} = (x_1, \cdots, x_d)$， $\Sigma$ 为协方差矩阵并且为正定矩阵
```math
q_{\beta}(\boldsymbol{x}'|\boldsymbol{x}) = \frac{\beta_k^{d/2}} {(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{\beta}{2} (\boldsymbol{x}'-\boldsymbol{x})^T\Sigma^{-1}(\boldsymbol{x}'-\boldsymbol{x}) \right)
```

当 $\Sigma$ 的所有特征值都趋近于0时，上式可在 $\beta=1$ 时退化为多维狄拉克函数 $q_{\beta}(\boldsymbol{x}'|\boldsymbol{x}) \rightarrow \delta^{(d)}(\boldsymbol{x}'-\boldsymbol{x})$。如果我们需要对决定论向量做模拟退火，可以构建一个协方差矩阵 $\Sigma = \sigma^2 I + \epsilon^2 (U U^{T} - \mathrm{diag}{UU^T})$， $I$ 为单位矩阵， $U$ 为随机实矩阵的归一化矩阵， $\epsilon$ 为非对角成分的微小关联，并且 $0 < \epsilon \ll \sigma$，此时可得到新分布 $\boldsymbol{x}' = \boldsymbol{x} + \boldsymbol{\eta} / \sqrt{\beta},~ \boldsymbol{\eta}\sim \mathcal{N}(0, \Sigma)$。

注意以上讨论的是对决定论的矢量形式的观测量以及奖励讨论的，也许会偏离**当前的**真实的训练环境，特别是 LLM 的强化学习，但是依然保留该讨论仅作为参考。


### 3.2. 随机再采样与 PPO / GRPO / GSPO 的协同

上一节把决定论条件下的观测量和奖励近似成了带有小方差的有效环境测度。接下来，把这个近似接回原始 RL 更新流程。原始 PPO / GRPO / GSPO 的更新器并不需要改变；变化发生在 rollout 阶段，也就是原来由环境测度 $\mu$ 给出的反馈，现在改成由小方差近似后的有效环境测度 $\mu_\sigma$ 给出。

原始环境测度下，一条路径由策略和环境共同生成
```math
\tau_b = (a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
```

其中 $b$ 为样本序号。在高斯近似后的有效环境测度下，对应的路径写成
```math
\tau'_b = (a'_{b,0},o'_{b,1},r'_{b,0},a'_{b,1},o'_{b,2},r'_{b,1},\ldots,a'_{b,T_b},o'_{b,T_b+1},r'_{b,T_b})
```

这里的撇号表示这条路径是在扰动后的环境反馈下生成的。也就是说，扰动后的观测量会进入下一步历史 :
```math
h'_{b,t+1}=\mathrm{concat}(h'_{b,t},a'_{b,t},o'_{b,t+1},r'_{b,t})
```

因此下一步动作仍然由策略生成，但条件历史已经变成扰动后的历史 :
```math
a'_{b,t}\sim \pi_\theta(\cdot\mid h'_{b,t})
```

在标量近似下，观测量和奖励的扰动可以写成 :
```math
\begin{aligned}
o'_{b,t+1} &= o_{b,t+1} + \sigma_O\xi_{O,b,t}, \quad \xi_{O,b,t} \sim \mathcal N(0,1) \\
r'_{b,t} &= r_{b,t} + \sigma_R\xi_{R,b,t}, \quad \xi_{R,b,t} \sim \mathcal N(0,1)
\end{aligned}
```

如果奖励由扰动后的观测量重新计算，则写成 :
```math
r'_{b,t} = R(o'_{b,t+1},a'_{b,t},h'_{b,t})
```

这样得到的路径回报为 :
```math
G[\tau'_b] = \sum_{t=0}^{T_b} \gamma^t r'_{b,t}
```

如果使用第 1.3 节中的 LLM 自回归回报形式，则可以写成 :
```math
G[\tau'_b] = \sum_{t=0}^{T_b} \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r'_{b,i,t}
```

其中 $r'_{b,i,t}$ 是扰动反馈下分配到第 $t$ 个外部步内部第 $i$ 个 token 的训练回报。若使用 token 级回报和阶段性回报的分解，则有
```math
r'_{b,i,t} = \omega^{i-L_t}R(h'_{b,i,t},a'_{b,i}) + \delta_{i,L_{t+1}-1}\phi(h'_{b,t},a'_{b,t},o'_{b,t+1})
```

从这里开始，为了表达简便，我们把 $'$ 号省略，从这里往后的被扰动过的量均用 $h, a, r, o$ 等原符号表示。使用 PPO 时，扰动路径 $\tau_b$ 给出新的 return 和 advantage。设扰动路径上的 advantage 为 $A_{b,i,t}$，
```math
\begin{aligned}
A_{b,i,t} &= Q(h_{b,i,t},a_{b,i}) - V(h_{b,i,t}) \\
Q(h_{b,i,t},a_{b,i}) &\simeq \widehat{G}_{b,i,t} = \sum_{j=i}^{L_{t+1}-1} r_{b,j,t} + \sum_{s=t+1}^{T} \gamma^{s-t} \sum_{j=L_s}^{L_{s+1}-1} r_{b,j,s}
\end{aligned}
```


则策略比率仍然按照当前策略和旧策略在同一动作上的概率比来计算
```math
\rho_{b,t}(\theta) = \frac{ \pi_\theta(a_{b,t}\mid h_{b,t}) }{ \pi_{\theta_{\mathrm{old}}}(a_{b,t}\mid h_{b,t}) }
```

于是 PPO 的 clipped objective 可以写成
```math
L_{\mathrm{PPO}} = \mathbb E_{b,t} \left[ \min \left( \rho_{b,t}(\theta)A_{b,t}, \mathrm{clip}(\rho_{b,t}(\theta),1-\epsilon,1+\epsilon)A_{b,t} \right) \right]
```

对 LLM 自回归生成，粗粒度动作 $a_{b,t}$ 是一段 token 序列，因此策略比率需要展开成 token 级概率比的连乘 :
```math
\rho_{b,t}^{\mathrm{AR}}(\theta) = \prod_{i=L_t}^{L_{t+1}-1} \frac{ \pi_\theta(a_{b,i}\mid h_{b,i,t}) }{ \pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t}) }
```

等价地，可以写成 log-ratio 的求和形式 :
```math
\log\rho_{b,t}^{\mathrm{AR}}(\theta)=\sum_{i=L_t}^{L_{t+1}-1}\left[\log\pi_\theta(a_{b,i}\mid h_{b,i,t})-\log\pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t})\right]
```

同理，GRPO / GSPO 也可以沿用这个结构。对同一个输入 $x$，策略生成一组扰动反馈下的路径 $\tau_1,\tau_2,\ldots,\tau_{K_s}$，每条路径都有自己的回报 $G_b = G[\tau_b],~ b=1, \ldots, K_s$，组内平均和方差为 :
```math
\bar G = \frac{1}{K_s} \sum_{b=1}^{K_s}G_b, \quad (\sigma_G)^2 = \frac{1}{K_s} \sum_{b=1}^{K_s} (G_b-\bar G)^2
```

于是组内标准化 advantage 为 :
```math
A_b = \frac{G_b-\bar G}{\sigma_G+\epsilon}
```

GSPO 可以进一步把整条生成序列作为采样单位。此时每条样本的回报 $G_b$ 由完整序列路径 $\tau_b$ 给出，而策略更新仍然通过序列中各 token 的 log-prob 或 log-ratio 作用到模型参数上。若把第 $b$ 条序列内部的 token 展开，则对应的序列级 log-ratio 为
```math
\log\rho_b^{\mathrm{seq}}(\theta) = \sum_{t=0}^{T_b} \sum_{i=L_t}^{L_{t+1}-1} \left[ \log\pi_\theta(a_{b,i}\mid h_{b,i,t}) - \log\pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t}) \right]
```

因此，高斯近似后的环境反馈可以直接接入 PPO / GRPO / GSPO。它改变的是 rollout 路径上的观测量、奖励和由此得到的回报，而策略更新仍然使用原有的 ratio、advantage、clipping 或 group normalization 结构。

### 3.3. 模拟退火

上一节只是把决定论的观测量和奖励扩张成小方差高斯提议分布。接下来引入模拟退火，用温度调度器来控制这个提议分布的扰动幅度。作用在观测量和奖励的高斯扰动上，用来控制 rollout 在反馈空间中的探索宽度。

令第 $k$ 轮的逆温度为
```math
\beta_k=\mathcal B(k),\quad \beta_k>0
```

其中 $B(k)$ 是人为指定的退火调度器。对标量变量 $x$，第 $k$ 轮的高斯提议分布可以写成 :
```math
q_{\beta_k}(x'\mid x) = \sqrt{\frac{\beta_k}{2\pi\sigma^2}} \exp\left( -\frac{\beta_k(x'-x)^2}{2\sigma^2} \right)
```

对应采样形式为 :
```math
x^{(k)} = x+\frac{\sigma}{\sqrt{\beta_k}}\xi_k, \quad \xi_k \sim \mathcal N(0,1) 
```

因此有效方差为 :
```math
\sigma_{\mathrm{eff}}^2(k) = \frac{\sigma^2}{\beta_k}
```

所以 $\beta_k$ 越小，提议分布越宽，rollout 在观测量和奖励空间中的偏离越大； $\beta_k$ 越大，提议分布越窄，路径越接近原始决定论反馈。若使用温度 $T_k$ 而不是逆温度，则有
```math
T_k=\frac{1}{\beta_k}, \quad \sigma_{\mathrm{eff}}^2(k)=\sigma^2T_k
```

对多维变量 $\boldsymbol{x}$，若协方差矩阵为 $\Sigma$，则第 $k$ 轮的退火提议分布为
```math
q_{\beta_k}(\boldsymbol{x}'\mid\boldsymbol{x}) = \frac{\beta_k^{d/2}} {(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{\beta_k}{2} (\boldsymbol{x}'-\boldsymbol{x})^T\Sigma^{-1}(\boldsymbol{x}'-\boldsymbol{x}) \right)
```

对应有效协方差为
```math
\Sigma_{\mathrm{eff}}(k) = \frac{1}{\beta_k}\Sigma
```

因此，模拟退火的作用可以概括为 : 固定基础高斯提议分布的形状，再用调度器 $\mathcal{B(k)}$ 控制其整体尺度。若采用单调冷却，则可以让早期 rollout 有更大的反馈扰动，后期逐渐收缩到原始反馈附近；若采用循环升温和降温，则可以在局部收缩后重新放大扰动，用来模拟淬炼过程并增加逃出局部区域的机会。

### 3.4. 统计力学解释

从第二节的物理图像看，原始 history-based RL 的路径权重已经可以写成 Boltzmann 权重
```math
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)=\exp(-\beta H_0[\tau])
```

其中 $H_0[\tau]$ 是由原始策略和环境诱导出的基础哈密顿量， $\beta$ 是逆温度。在 LLM 自回归形式下，上式的策略部分展开为
```math
P_{\pi,\mu}^{\mathrm{AR}}[\tau]= \prod_{t=0}^{T} \left[ \prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t}) \right] \mu(o_{t+1},r_t\mid h_t,a_t) =\exp(-\beta H_0^{\mathrm{AR}}[\tau])
```

第 3.1 ~ 3.3 节讨论的是 $o_{t+1}$ 和 $r_t$ 的再采样，不改变 token 生成测度本身。因此在 LLM 情形下， $a_t$ 只需要理解成自回归 token 序列
```math
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
```

回报仍然作为路径上的统计观测量
```math
G[\tau_b]=\sum_{t=0}^{T}\gamma^t r_{b,t} \quad \text{or} \quad G[\tau_b] = \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{b,i,t}
```

对同一输入采样得到的一组路径可以看成原始或者 rollout 增强下(包含局部扰动)的路径分布 $\tau_b\sim P_{\pi,\mu}[\tau],~ b=1,\ldots,K_s$， $b$ 为样本编号。组内回报均值和方差为
```math
\bar G=\frac{1}{K_s}\sum_{b=1}^{K_s}G[\tau_b],\quad \sigma_G^2=\frac{1}{K_s}\sum_{b=1}^{K_s}(G[\tau_b]-\bar G)^2
```

这里 $\bar G$ 是组内回报的平均观测量， $\sigma_G^2$ 是回报观测量的涨落强度。

在 3.3 中， $\beta_k$ 被直接用作第 $k$ 轮模拟退火的逆温度参数。为了和第二节的热力学图像统一，可以把 $\beta_k$ 看成逆温度 $\beta$ 与第 $k$ 轮局部重采样强度 $\alpha_{\,\mathrm{res}}^{(k)}$ 的乘积 
```math
\beta_k=\beta\alpha_{\mathrm{res}}^{(k)}
```

其中 $\alpha_{\mathrm{res}}^{(k)}$ 是局部观测量 / 奖励扰动势的强度。后文为简洁起见仍统一写作 $\beta_k$。给定一条原始 rollout 路径 $\tau$，高斯重采样得到的路径为 $\tau^k=(a_0,o_1^{(k)},r_0^{(k)},\ldots,a_T,o_{T+1}^{(k)},r_T^{(k)})$，如果是 LLM 自回归路径，则每个 $a_t$ 仍然是同一个 token 序列，重采样只作用在 $o_{t+1}$、 $r_t$ 或它们的连续表示上。观测量高斯重采样对应为一个局部时刻 $t$ 的二次哈密顿量。若采用独立同尺度高斯扰动，可以写成
```math
H_o(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{(o_{t+1}^{(k)}-o_{t+1})^2}{2\sigma_o^2}
```

如果使用观测噪声协方差矩阵 $\Sigma_o$，则为

```math
H_o(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{1}{2}(o_{t+1}^{(k)}-o_{t+1})^\top\Sigma_o^{-1}(o_{t+1}^{(k)}-o_{t+1})
```

若直接对奖励也做高斯扰动，可以增加奖励扰动哈密顿量
```math
H_r(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{(r_t^{(k)}-r_t)^2}{2\sigma_r^2}
```

因此，重采样哈密顿量为如下形式
```math
H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)=H_o(\tau^{(k)}\mid\tau)+H_r(\tau^{(k)}\mid\tau)
```

如果奖励不是直接加噪声，而是由扰动后的观测量重新计算，则可以只保留 $H_o$，并令  $r_t^{(k)}=R(o_{t+1}^{(k)},a_t,h_t)$。

综上原始路径与重采样路径的联合权重为
```math
P_k(\tau,\tau^{(k)})\propto \exp\left(-\beta H_0[\tau]-\beta_k H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)\right)
```

因此经过采样增强后的作用量为 : 

```math
S_{\mathrm{augmented}}^{(k)}[\tau,\tau^{(k)}]=\beta H_0[\tau]+\beta_k H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)
```

此时我们可以做这样的统计力学解释
- 原始路径由基础哈密顿量 $H_0[\tau]$ 加权，在这条路径附近的观测量和奖励重采样由局部扰动哈密顿量 $H_{\mathrm{resampled}}$ 产生热涨落
- $\beta_k$ 控制局部涨落强度。小 $\beta_k$ 相当于局部高温，重采样路径在原始路径附近有更大的热涨落；后期大 $\beta_k$ 相当于局部低温，重采样路径逐渐收缩到原始观测和奖励附近

### 3.5. 相关研究

- $(o,r)$ 再采样
  - 对应文献方向: 高斯噪声提议分布 / noisy environment augmentation / observation noise / reward noise
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

- 提议分布扩展
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

## 4. 基于路径积分的路径采样

第三节讨论的是在原始 RL 更新器不变的前提下，对观测量 $o_{t+1}$ 和奖励 $r_t$ 做局部高斯扰动，从而扩展 rollout 的反馈空间。这种方法仍然是在原始策略生成的路径附近做增强，主要改变的是环境反馈，而不是直接在整条路径空间中搜索更优路径。

本节换一个角度，从路径积分表象出发，把整条轨迹 $\tau$ 本身看成采样对象。路径的质量不再只由单步奖励或局部扰动决定，而是由整条路径的作用量以及目标观测量共同决定。采样到的路径不仅要完成任务，还要满足 KL、长度、格式、资源等约束；这样的路径应该对应更低的有效作用量，或者说位于更稳定的低能区域。因此需要引入路径积分。

为此，我们在原始作用量 $S_0[\tau]$ 上加入源项，把回报、惩罚和约束组织成目标观测量 $F[\tau]$，从而得到带源项的吉布斯分布。这个分布会把路径采样偏向高回报、低惩罚的区域。接下来使用 MCMC 或 Langevin 这类采样方法，不是为了把 $a,o,r$ 分别当作状态做局部随机游走，而是为了在整条路径空间中搜索低作用量路径。MCMC 可以让陷入局部低点的路径仍有概率跳出；Langevin 则在连续隐状态路径上结合了作用量下降和随机扰动。真正**稳定的高质量路径**，应当是在扰动后仍能回到可完成任务区域的路径盆地，而不是脱离目标越走越远。

### 4.1. 回顾： RL 的路径积分表示

第二节中我们已经得到了 RL 的路径积分表示
```math
J(\pi)=\int \mathcal{D}\tau ~ e^{-S_{\pi,\mu}[\tau]} G[\tau]
```
其有自回归与非自回归两种形式，自回归形式对应 LLM 的 next token prediction，多个新 token 构成一个新动作 $a_t$；非自回归形式对应传统的 next action prediction，直接生成动作 $a_t$。两种形式的作用量、测度以及观测量分别为
- 自回归: 
```math
\begin{aligned}
S^{AR}_{\pi,\mu}[\tau] &= - \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \left( \log \pi(a_i|h_{i,t}) + \delta_{i,L_{t+1}-1} \log \mu(o_{t+1},r_t | h_t, a_t) \right) \\
\mathcal{D}\tau &= \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \equiv \prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right] do_{t+1}\,dr_t\\
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t}
\end{aligned}
```

- 非自回归:
```math
\begin{aligned}
S_{\pi,\mu}[\tau] &= - \sum_{t=0}^T \left[ \log \pi(a_t|h_t) + \log \mu(o_{t+1},r_t | h_t, a_t) \right] \\
\mathcal{D}\tau &= \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \\
G[\tau] &= \sum_{t=0}^T \gamma^t r_{t}
\end{aligned}
```

在路径积分表象下，强化学习可以作出以下物理解释
- $a_t,o_t,r_t$: 一维时间路径上场的自由度，这里的场就是标量场 (动作场，观测场以及回报场)
- $\mathcal{D}\tau$: 所有格点上的场的测度
- $G[\tau]$: 物理观测量
- $e^{-S_{\pi,\mu}[\tau]}$: 路径积分的权重，其中 $S_{\pi,\mu}$ 为作用量
- History-based RL: 拥有 $a,o,r$ 三个自由度的复杂的一维单粒子系统，其复杂性体现在基于历史轨迹 $h_t$ 的长程耦合。其耦合由策略 $\pi$ 以及环境测度 $\mu$ 决定。
- 等效原始哈密顿量: $S_{\pi,\mu}[\tau] = \beta H_{\pi,\mu}[\tau] \equiv S_0[\tau] = \beta H_0[\tau],~ H_{\pi,\mu}[\tau] = - \log P_{\pi,\mu}$


### 4.2 源项与吉布斯分布
这里引入场论中的源项，建立它与 RL 物理图像的对应。带源项的路径积分为
```math
Z[\eta] = \int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]}
```
对源项系数微分可得到
```math
\begin{aligned}
\frac{\partial}{\partial \eta} \log Z[\eta] &= \frac{1}{Z[\eta]} \int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} F[\tau] = \mathbb{E}_\eta [F[\tau]] \\
\frac{\partial^2}{\partial^2 \eta} \log Z[\eta] &= \text{Var}_\eta [F[\tau]]
\end{aligned}
```
当 $\eta = 0$ 时可得到原始作用量下的观测量期望值与方差。在 RL 中，可以把观测量作为源项，并进一步引入长度惩罚、KL 惩罚等惩罚项
```math
F[\tau; \lambda_G, \lambda_N, \lambda_{KL}] = \lambda_G G[\tau] - \lambda_N N[\tau] - \lambda_{KL} K[\tau]
```
这里的 KL 惩罚可选择
- 严格KL散度(完整分布)
```math
K[\tau] = \sum_{t=0}^T \sum_{i=L_{t}}^{L_{t+1}-1} \int d\tilde{a}_i ~ \pi(\tilde{a}_i|h_{i,t}) \log \frac{\pi(\tilde{a}_i|h_{i,t})}{\pi_\mathrm{ref}(\tilde{a}_i|h_{i,t})}
```
- 采样KL散度(只看被采样的)
```math
K[\tau] = \sum_{t=0}^T \sum_{i=L_{t}}^{L_{t+1}-1} \log \frac{\pi(a_i|h_{i,t})}{\pi_\mathrm{ref}(a_i|h_{i,t})}
```

轨迹 $\tau$ 在 $\eta$ 下的概率分布如下
```math
q_{\eta}(\tau) = \frac{1}{Z[\eta]} \exp\left( - S_0[\tau] + \eta F[\tau] \right)
```

使用哈密顿量的定义，可以得到等效哈密顿量 $H_{\eta}[\tau] = H_0[\tau] - \frac{\eta}{\beta}F[\tau]$，因此
```math
q_\eta[\tau] \propto \exp\left( - \beta H_0[\tau] + \eta F[\tau] \right)
```

这个分布对应统计力学中的吉布斯分布，惩罚项可视作化学势等等效势能。该吉布斯分布是偏置分布；加入源项后，路径分布更偏向高回报、低惩罚区域，与 RL 的目标一致。若要回到原始分布，则需使用重加权法
```math
\begin{aligned}
& \mathbb{E}_{\eta=0} \left[F[\tau] \right] = \frac{\int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} e^{- \eta F[\tau]} F[\tau]}{\int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} e^{- \eta F[\tau]}}  \\
=~& \frac{\mathbb{E}_{\eta} \left[F[\tau]e^{- \eta F[\tau]} \right]}{\mathbb{E}_{\eta} \left[e^{- \eta F[\tau]} \right]} = \frac{\sum_{b}F[\tau_b]e^{-\eta F[\tau_b]}}{\sum_b e^{-\eta F[\tau_b]}}
\end{aligned}
```
或者使用 $\eta \rightarrow 0$ 的外插法。

### 4.3. MCMC 路径采样

给定当前路径 $\tau$ 时，MCMC 的候选路径不能理解成对整条已经生成好的路径做任意扰动。由于 History-based RL 的路径具有因果结构，后面的观测量和回报依赖前面的历史与动作。如果直接改掉某个动作，却保留后面的观测量和回报，那么后半段很可能已经不是新动作下的合法反馈。因此，候选路径 $\tau'$ 必须由合法提议分布生成。

这里的提议分布可以只扰动动作前缀，也可以同时扰动动作、允许扰动的观测量以及软回报。只扰动观测量或回报也可以，但如果这些字段对模型后续决策不重要，模型可能继续生成同样的动作，路径分支不会发生明显变化。因此更一般地，候选路径的提议分布可以写成联合形式
```math
\begin{aligned}
& (c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}}) \sim q_{\mathrm{prop}}(c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}}\mid \tau) \\
& \tau' = \mathrm{Rollout}(c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}})
\end{aligned}
```

其中 $`c'_{a}`$ 表示扰动后的动作前缀，可以是 token 前缀、隐状态前缀、工具调用前缀或 agent 步前缀。 $`o'_{\mathrm{allow}}`$ 表示**允许扰动**的观测字段。$`r'_{\mathrm{soft}}`$ 表示软评分、verifier 评分、reward model 评分等可以扰动的回报或评分。若只扰动动作，则固定 $`o'_{\mathrm{allow}}=o_{\mathrm{allow}}`$、 $`r'_{\mathrm{soft}}=r_{\mathrm{soft}}`$ 。若做联合扰动，则三者一起采样。

在 LLM 自回归情形下，动作前缀可以写成 $`c_{a,i,t} = (h_t,a_{L_t},a_{L_t+1},\cdots,a_i)`$，扰动后得到 $`c'_{a,i,t} \sim q_a(c'_{a,i,t}\mid c_{a,i,t})`$ 。允许扰动的观测量和回报可以写成掩码形式
```math
\begin{aligned}
o'_{t+1} &= \mathcal E_o(o_{t+1};m_o,\xi_o) \\
r'_t &= \mathcal E_r(r_t;m_r,\xi_r)
\end{aligned}
```
这里 $\mathcal{E}_o ,~ \mathcal{E}_r$ 表示受掩码约束的随机扰动算子。它可以是对某些数值的高斯扰动，也可以是字段级随机编辑，关键是只作用在允许扰动的观测字段或软评分上。其中 $m_o,m_r$ 只打开允许扰动的字段。比如数值型返回值、置信度、软评分、元数据、不影响事实语义的辅助字段，都可以作为提议分布的一部分。对于工具调用的核心返回值、代码执行结果、判题结果、数据库查询结果等硬观测量，不能随意扰动，否则候选路径会破坏因果结构。

得到候选路径后，用带源项作用量比较当前路径和候选路径。Metropolis-Hastings 接受率为
```math
\begin{aligned}
A_k(\tau\rightarrow\tau') = \min\left( 1, \exp\left[-(S_\eta[\tau']-S_\eta[\tau])\right] \frac{q_{\mathrm{prop}}(\tau\mid\tau')} {q_{\mathrm{prop}}(\tau'\mid\tau)} \right)
\end{aligned}
```

候选路径如果有效作用量更低，就更容易被接受。即使候选路径暂时更差，只要接受率不为零，它仍然有机会被保留下来。这个机制让路径采样有机会从局部低点跳出去，而不是一直停在当前路径附近。

### 4.4. Langevin 路径采样

Langevin 法需要对作用量求导，因此更适合在连续自由度上做。对 LLM 来说，离散 token 不能直接做 Langevin 更新，但可以用隐状态作为连续动作变量，例如
```math
\begin{aligned}
\tau_z &= [o_0,z_0,r_0,o_1,\cdots,o_T,z_T,r_T,o_{T+1}] \\
z_t &= [z_{L_t},\cdots,z_{L_{t+1}-1}]
\end{aligned}
```

这里同样不能直接把完整隐状态路径随便改掉。更合理的方式是选择一段隐状态前缀，对这个前缀做 Langevin 更新，然后从更新后的前缀继续解码或 rollout，重新得到后续路径。令 $c_z^{(\ell)} = [h_t,z_{L_t}^{(\ell)},\cdots,z_i^{(\ell)}]$，Langevin 的更新如下
```math
z_{L_t:i}^{(\ell+1)} = z_{L_t:i}^{(\ell)} - \epsilon\nabla_{z_{L_t:i}^{(\ell)}}S_\eta[ \mathrm{Rollout}(c_z^{(\ell)},o_{\mathrm{allow}}^{(\ell)},r_{\mathrm{soft}}^{(\ell)}) ] + \sqrt{2\epsilon}\xi_\ell, \quad \xi_\ell \sim \mathcal N(0,I)
```
其中 $\ell$ 表示 Langevin 更新步。如果同时扰动允许的观测量和软回报，则可以写成 :
```math
o_{\mathrm{allow}}^{(\ell+1)} = \mathcal E_o(o_{\mathrm{allow}}^{(\ell)};m_o,\xi_{o,k}),\quad r_{\mathrm{soft}}^{(\ell+1)} = \mathcal E_r(r_{\mathrm{soft}}^{(\ell)};m_r,\xi_{r,k})
```

更新后得到新的候选前缀与反馈
```math
c_z^{(\ell+1)} = [h_t,z_{L_t}^{(\ell+1)},\cdots,z_i^{(\ell+1)}]
```

然后继续 rollout
```math
\tau_z^{(\ell+1)} = \mathrm{Rollout} (c_z^{(\ell+1)},o_{\mathrm{allow}}^{(\ell+1)},r_{\mathrm{soft}}^{(\ell+1)})
```

Langevin 中的梯度项把路径往低作用量方向拉，随机项保留热涨落。这里的目标不是让路径随便偏离任务，而是在动作前缀和允许反馈被扰动后，仍然能通过后续 rollout 回到可完成任务、低惩罚、低作用量的路径区域。真正稳定的好路径不是一个孤立点，而是一个路径盆地；轻微扰动之后，后续生成仍然能回到可完成任务的轨道上。


### 4.5. 相关研究

- 路径积分 RL
  - 对应文献方向: Path Integral Control / ${PI}^2$
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

- 隐状态空间上的 Langevin
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

---

# Appendix A : 时间方向重整化

因为这是一个一维时间路径积分，所以重整化主要沿时间方向做。非自回归版本中，设 $\ell$ 表示宏观时间块编号，把每 $b$ 个微观 action 合并成一个宏观块 : 

```math
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1})
```

多层压缩后 : 

```math
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}
```

在路径积分层面，微观路径到宏观路径的映射为 $\bar\tau=\mathcal C(\tau)$。如果讨论原始路径分布，则宏观基础作用量由积分掉微观自由度得到 : 

```math
\exp(-S_0[\bar\tau]) = \int_{\mathcal C(\tau)=\bar\tau} \exp(-S_0[\tau]) \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
```


LLM 自回归版本有两层时间结构。外层是环境 / agent step $t$，内层是 token 位置 $i$。第 $t$ 个外部 step 的 token 段为 :
```math
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
```

因此最基本的 LLM 时间粗粒化已经是把 token 微步积分成段级动作 $a_t$ :
```math
\prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i = \prod_{t=0}^{T}da_t
```

如果还要在一个外部 step 内部继续做 token-block 压缩，令第 $t$ 个外部 step 内部的第 $\ell$ 个 token 块为 $B_{t,\ell}=C_\phi(a_{L_t+\ell b},a_{L_t+\ell b+1},\ldots,a_{L_t+(\ell+1)b-1})$，则 LLM 自回归基础作用量的粗粒化写成
```math
\exp(-S_0^{\mathrm{AR}}[\bar\tau]) = \int_{\mathcal C(\tau)=\bar\tau} \exp(-S_0^{\mathrm{AR}}[\tau]) \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i\,do_{t+1}\,dr_t
```

其中 
```math
S_0^{\mathrm{AR}}[\tau] =- \sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t}) -\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
```

如果压缩块内部的有效奇异值谱快速衰减，则截断安全；如果谱近似平直，则硬截断会损失大量信息 : 
```math
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ 的截断会造成强信息损失}
```

**注意**: 这个“重整化”仅仅是在讨论 token 层面的压缩，而不是在做真正的“语义信息重整化”。真正的重整化指的是，对信息进行粗粒化后，能够在某个尺度下产生“语义不变量”。作者认为目前的各种 token 压缩技术都是基于可训练的权重对信息选择压缩，这个并不是真正的重整化，因为这种压缩无法保证也没有证据证明可以产生某种不变量，或者说 RG 稳定点。
