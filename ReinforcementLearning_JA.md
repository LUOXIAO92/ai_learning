# Reinforcement Learning

## 1. 強化学習の定式化

### 1.1 一般形式 General Reinforcement Learning
**論文** (数学定式化): [[arXiv:cs/0004001] A Theory of Universal Artificial Intelligence based on Algorithmic Complexity](https://arxiv.org/abs/cs/0004001)

別名: History-based RL

一般形式の強化学習は次のようにモデリングされている
- $\mathcal{O}$: 観測空間 - 観測の集合
  - $o_t\in\mathcal{O}$
  - 例: 二次元平面上の自動運転では、$o_t=(x_t,y_t, v_t,O_t)$
    - $(x_t,y_t)$: 車の位置
    - $v_t$: 車の速度
    - $O_t$: 半径 $d^{\mathrm{obstacle}}$ 内の障害物情報

- $r_t$: 報酬 - スコア
  - $r_t\in\mathbb{R}$
  - 報酬はルールベース、モデルで学習してもよい
  - $r_t = R(o_{t+1}, a_t, h_t)$: 報酬が含まない環境の場合

- $\mathcal{A}$: 行動空間 - 行動の集合
  - $a_t\in\mathcal{A}$
  - 例: 二次元平面上の自動運転では、$a_t=(\theta_t,u_t,b_t)$
    - $\theta_t\in[0,2\pi)$: ハンドル回転角度
    - $u_t\in[0,u^{\max}]$: アクセルの踏み加減
    - $b_t\in[0,b^{\max}]$: ブレーキの踏み加減
  
- $\mathcal{H}$: 履歴空間 - 過去の観測・行動・報酬の系列
  - $h_t=(o_0,a_0,r_0, o_1,a_1,r_1,\ldots,a_{t-1},r_{t-1}, o_t)$

- $\mu$: 環境測度 / Measure - 行動後に次の観測と報酬を返す (環境との相互作用). 
  - 隠れた変数 (hidden variable)による影響を取り込む
    - 隠れた変数による影響を**確率過程**として扱う
  - $(o_{t+1},r_t)\sim\mu(\cdot | h_t,a_t)$: ある分布に従うランダム環境
  - $(o_{t+1},r_t) = \mu(h_t,a_t)$: 決定論的な環境
  - $o_{t+1} = \mu(h_t,a_t)$: 報酬が含まない環境もある. ただしこの場合では報酬が別途でもとまる 

- $\gamma\in[0,1]$: 割引率 - 将来の報酬をどれだけ重視するかを表すパラメータ
  - 有限長タスクでは $\gamma=1$ も可能

- $\pi_\theta(a_t\mid h_t)$: 方策 policy - **ニューラルネットワーク / ルールベース / 回帰モデル / 確率分布モデル / etc.**
  - $a_t\sim\pi_\theta(\cdot\mid h_t)$
  - $a_t=\pi_\theta(h_t)$

#### 1.1.1 環境測度 / Measure $\mu$ の説明
例: 車が迫っていく時に、歩行者の気分によって退避するか、無視するか
  - 歩行者の気分: 隠れた変数、観測不可
  - $o_{t}$: 時刻 $t$ において、車の情報および歩行者の観測可能な情報 (進行方向、歩行速度など)
  - $a_t\sim\pi_\theta(\cdot|h_t)$: 車および履歴 ($o_{t}$が含まれる) に基づいて、車の動作を予測する
  - $o_{t+1} \sim\mu(\cdot| h_t, a_t)$: 時刻 $t+1$ における観測可能な情報を予測
    - 従来は運動方程式などの決定論的な方法を使うらしい

#### 1.1.2 最適化目標
**累積報酬 (Trajectory Return)** $G_0$
```math
\begin{align}
G_0 &= \sum_{t=0}^T \gamma^t r_{t} \\
a_t &\sim \pi_\theta(\cdot | h_t),\\
o_{t+1} &\sim \mu(\cdot |h_t, a_t), \\
r_t &= R(h_t, a_t, o_{t+1}) \\
h_{t+1} &= \text{concat}(h_t, [a_t, r_t, o_{t+1}])
\end{align}
```
を**最大**にする. つまり
```math
\begin{align}
\pi^* &= \underset{\pi}{\operatorname{argmax}} J(\pi) \quad \text{or} \quad \pi^*_\theta = \underset{\theta}{\operatorname{argmax}} J(\pi_\theta) \\
J(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}} \sum_{a_1 \in \mathcal{A}} \cdots \sum_{a_T \in \mathcal{A}}  \sum_{o_{T+1} \in \mathcal{O}} \nonumber \\
&\quad \left( \prod_{t=0}^T \pi_\theta(a_t|h_t)\mu(o_{t+1}|h_t, a_t)\right) \left(\sum_{t=0}^T \gamma^t r_t \right) \\
&\equiv \mathbb{E}_{a\sim \pi_\theta, o\sim\mu} [G_0 | o_0]
\end{align}
```
Eq(8)という形式は強化学習の定番となっているが、時間順序を省略するため良い表現ではない. また、$\mathbb{E}[\cdot|o_0]$が付いているのはは初期観測量$o_0$という条件が付いていることを意味する.

ただし、$t=k\sim L$の累計報酬の一般形式は以下となる
```math
\begin{equation}
G^{(L)}_k = \sum_{t=k}^{L} \gamma^{t-k} r_{t}
\end{equation}
```
- 任意の$k$ステップ目から始まり、合計$L$ステップの報酬をとるという意味


### 1.2 Markov decision process (MDP)

MDPは一般形式RLの特例 <- RLの定番定式化
- 履歴 $h_t$ を状態 $s_t$ に圧縮 / mapping: $s_t=f(h_t)$, $s_t \in \mathcal{S}$
- $s_t$ が $s_{t+1}$ を予測するために十分な情報が含まれている: $P(s_{t+1} | h_t, a_t) = P(s_{t+1} | s_t, a_t)$
  - 環境測度 $\mu$ $\rightarrow$ 遷移確率 $P$  
- 奨励が$s_t, a_t, s_{t+1}$のみで決まる: $r_t = r(s_t, a_t, s_{t+1}) = R(h_t, a_t, o_{t+1})$
- 方策が $s_t$ のみに依存: $\pi_\theta(a_t|s_t)$ or $a_t\sim \pi_\theta(\cdot | s_t)$

それ以外は一般形式と同様

#### 1.2.1 最適化目標: 一般形式とほぼ同様
**累積報酬 (Trajectory Return)** $G_0$
```math
\begin{align}
G_0 &= \sum_{t=0}^T \gamma^t r_{t} \\
a_t &\sim \pi_\theta(\cdot | s_t),\\
s_{t+1} &= P(\cdot|s_t, a_t), \\
r_t &= r(s_t, a_t, s_{t+1}), \\
\end{align}
```
を**最大**にする. つまり
```math
\begin{align}
\pi^* &= \underset{\pi}{\operatorname{argmax}} J(\pi) \quad \text{or} \quad \pi^*_\theta = \underset{\theta}{\operatorname{argmax}} J(\pi_\theta) \\
J(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \left(\sum_{t=0}^T \gamma^t r_t \right)
\end{align}
```


### 1.3 Bellman equation

報酬は以下の漸化式で表現できる
```math
\begin{align}
G_0 &= \sum_{t=0}^T \gamma^t r_t = r_0 + \gamma\left(\sum_{t=1}^T \gamma^{t-1}r_t \right) = r_0 + \gamma G_1 \nonumber \\
G_1 &= r_1 + \gamma \left( \sum_{t=2}^T \gamma^{t-2} r_t \right) = r_1 + \gamma G_2 \nonumber \\
&\cdots \nonumber \\
\Rightarrow G_k &= r_k + \gamma G_{k+1}
\end{align}
```

#### 1.3.1 MDP formulation RL
$t=0$に関して、$a_0, s_1$を先に積分する
```math
\begin{align}
J &= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right)  \left( r_0 + \gamma G_1 \right) \nonumber \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) r(s_0, a_0, s_1) \nonumber \\
&\quad + \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \gamma G_1 \nonumber \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \pi_\theta(a_0 | s_0) P(s_1 | s_0, a_0) r(s_0, a_0, s_1) \\
&\quad + \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \pi_\theta(a_0|s_0) P(s_{1} | s_0, a_0) \nonumber \\
&\quad \times \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=1}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \gamma G_1 \nonumber \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \pi_\theta(a_0 | s_0) P(s_1 | s_0, a_0) \left[ r(s_0, a_0, s_1) + \gamma V^{\pi_\theta}_1 (s_1) \right]
\end{align}
```

Eq(17)は以下の規格化条件が課されるから
```math
\begin{equation}
\sum_{a_k \in \mathcal{A}} \sum_{s_{k+1} \in \mathcal{S}}  \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \prod_{t=k}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) = 1
\end{equation}
```

Eq(18)も同様に, $t=1$において、$a_1, s_2$を先に積分する
```math
\begin{align}
V^{\pi_\theta}_1 (s_1) &\equiv \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=1}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) G_1 \nonumber \\
&= \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=1}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \left( r(s_1, a_1, s_2) + \gamma G_2 \right) \\
&= \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \pi_\theta(a_1 | s_1) P(s_2 | s_1, a_1) \left[ r(s_1, a_1, s_2) + \gamma V^{\pi_\theta}_2 (s_2) \right]
\end{align}
```

$t=k$では、
```math
\begin{align}
V^{\pi_\theta}_k (s_k) &= \sum_{a_k \in \mathcal{A}} \sum_{s_{k+1} \in \mathcal{S}} \pi_\theta(a_k | s_k) P(s_{k+1} | s_k, a_k) \left[ r(s_k, a_k, s_{k+1}) + \gamma V^{\pi_\theta}_{k+1} (s_{k+1}) \right] \\
&= \mathbb{E}_{a_k\sim\pi_\theta(\cdot|s_k), s_{k+1}\sim P(\cdot | s_k, a_k)} \left[ r(s_k, a_k) + \gamma V^{\pi_\theta}_{k+1} (s_{k+1}) | s_k\right]
\end{align}
```

ここで、$J_{s_0}(\pi_\theta) = V^{\pi_\theta}_{0}(s_0)$である. $J$ の添字 $s_0$ は初期状態 $s_0$ が与えられたことを意味する. 

$V$ は **価値関数 (Value function)** と呼ばれ、状態 $s_t$ におけるタスクの **将来どれくらい報酬をもらえるのか** を評価する関数である.  $V$をさらに $s_{k+1}$に関する積分を求めておくと
```math
\begin{align}
V^{\pi_\theta}_k (s_k) &= \sum_{a_k \in \mathcal{A}} \pi_\theta(a_k | s_k) \sum_{s_{k+1} \in \mathcal{S}} P(s_{k+1} | s_k, a_k)\left[ r(s_k, a_k, s_{k+1}) + \gamma V^{\pi_\theta}_{k+1} (s_{k+1}) \right] \\
&\equiv \sum_{a_k \in \mathcal{A}} \pi_\theta(a_k | s_k) Q^{\pi_\theta}_k(s_k, a_k)
\end{align}
```
**行動価値関数 (Action-value function, Q function)** $Q_k(s_k, a_k)$ が得られる. Q関数は状態 $s_t$ が与えられた時、行動 $a_t$ をとると、**将来どれくらい報酬をもらえるのか** を評価する関数である. 

強化学習におけるBellman方程式の定番表現は以下となる

```math
\begin{align}
V_{\pi} (s) &= \sum_{a \in \mathcal{A}} \pi_\theta(a | s) \sum_{s' \in \mathcal{S}} P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right] \\
Q_\pi(s,a) &= \sum_{s' \in \mathcal{S}} P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right]
\end{align}
```

#### 1.3.2 General (History-based) RL

- $t=0$, $a_0, o_1$を先に積分する(和をとる). $h_t=[o_0, a_0, r_0, o_1, a_1, r_1,\cdots,o_{t-1}, a_{t-1}, r_{t-1}, o_t]$

```math
\begin{align}
J(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}} \sum_{a_1 \in \mathcal{A}}\cdots \sum_{a_T \in \mathcal{A}} \sum_{o_{T+1} \in \mathcal{O}}  \left( \prod_{t=0}^T \pi_\theta(a_t|h_t)\mu(o_{t+1}|h_t, a_t)\right) \left(r(h_0, a_0, o_{1}) + \gamma G_1\right) \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}}   \pi_\theta(a_0|h_0)\mu(o_{1}|h_0, a_0) r(h_0, a_0, o_{1}) \nonumber \\ 
&\quad + \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}} \sum_{a_1 \in \mathcal{A}}\cdots \sum_{a_T \in \mathcal{A}} \sum_{o_{T+1} \in \mathcal{O}}  \left( \prod_{t=0}^T \pi_\theta(a_t|h_t)\mu(o_{t+1}|h_t, a_t)\right) \gamma G_1 \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}}   \pi_\theta(a_0|o_0)\mu(o_{1}|o_0, a_0) \left[ r(o_0, a_0, o_{1}) + \gamma V^{\pi_\theta}_{1}(o_0; a_0, o_1) \right] \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}}   \pi_\theta(a_0|h_0)\mu(o_{1}|h_0, a_0) \left[ r(h_0, a_0, o_{1}) + \gamma V^{\pi_\theta}_{1}(h_1) \right]
\end{align}
```
ここで、$h_1=[o_0;a_0, o_1]$である. $o_0$ は初期観測値であり、積分されないため、「;」で隔離しておく. 

MDPと同様に、$V$をさらに展開していくと
```math
\begin{align}
V^{\pi_\theta}_{1}(h_1) &= V^{\pi_\theta}_{1}(o_0; a_0, o_1) \\
&\cdots \nonumber \\
&= \sum_{a_1 \in \mathcal{A}} \sum_{o_2 \in \mathcal{O}} \pi_\theta(a_1|o_0;a_0,o_1)\mu(o_{2}|o_0;a_0,o_1, a_1) \nonumber \\
&\quad \times \left[ r(o_0;a_0,o_1, a_1, o_{2}) + \gamma V^{\pi_\theta}_{2}(o_0; a_0, o_1, a_1, o_{2}) \right] \\
&= \sum_{a_1 \in \mathcal{A}} \sum_{o_2 \in \mathcal{O}}   \pi_\theta(a_1|h_1)\mu(o_{2}|h_1, a_1) \left[ r(h_1, a_1, o_{2}) + \gamma V^{\pi_\theta}_{2}(h_2) \right]
\end{align}
```

最終的に、MDP定式化RLと同じ形式のBellman方程式が得られる
```math
\begin{align}
V^{\pi_\theta}_{k}(h_k) &= \sum_{a_k \in \mathcal{A}} \sum_{o_{k+1} \in \mathcal{O}}   \pi_\theta(a_k|h_k)\mu(o_{k+1}|h_k, a_k) \left[ r(h_k, a_k, o_{k+1}) + \gamma V^{\pi_\theta}_{k+1}(h_{k+1}) \right] \\
&= \sum_{a_k \in \mathcal{A}} \pi_\theta(a_k|h_k) Q_k(h_k, a_k)
\end{align}
```
ただし、異なるのは、一般形式のRLは(行動)価値関数の引数 $h_k$ は履歴であるため、増え続ける.

```math
\begin{align}
V_{\pi} (h) &= \sum_{a \in \mathcal{A}} \pi_\theta(a | h) \sum_{o' \in \mathcal{O}} P(o | h, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] \\
Q_\pi(h,a) &= \sum_{o' \in \mathcal{O}} P(o' | o, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] \\
h' &= \text{concat}[h, (a, r, o')]
\end{align}
```
