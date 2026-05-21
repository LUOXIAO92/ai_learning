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

#### 1.1.1 簡単な説明例
例: 車が迫っていく時に、歩行者の気分によって退避するか、無視するか

歩行者の気分: 隠れた変数、観測不可
  - $o_{t}$: 時刻 $t$ において、車の情報および歩行者の観測可能な情報 (進行方向、歩行速度など)
  - $a_t\sim\pi_\theta(\cdot|h_t)$: 車および履歴 ($o_{t}$が含まれる) に基づいて、車の動作を予測する
  - $o_{t+1} \sim\mu(\cdot| h_t, a_t)$: 時刻 $t+1$ における観測可能な情報を予測
    - 従来は運動方程式などの決定論的な方法を使う
  - $r_t = R(h_t, a_t, o_{t+1})$: 過去の履歴(経験)、新しい動作および観測を使って、奨励を評価する. 

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
J_{o_0}(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}} \sum_{a_1 \in \mathcal{A}} \cdots \sum_{a_T \in \mathcal{A}}  \sum_{o_{T+1} \in \mathcal{O}} \nonumber \\
&\quad \left( \prod_{t=0}^T \pi_\theta(a_t|h_t)\mu(o_{t+1}|h_t, a_t)\right) \left(\sum_{t=0}^T \gamma^t r_t \right) \\
&\equiv \mathbb{E}_{a\sim \pi_\theta, o\sim\mu} [G_0 | o_0]
\end{align}
```
$J_{o_0}$ の添字 $o_0$ は初期観測という初期条件が付いていることを意味する. Eq(8)という形式は強化学習の定番となっているが、時間順序を省略するため良い表現ではない. また、$\mathbb{E}[\cdot|o_0]$が付いているのはは初期観測量$o_0$という条件が付いていることを意味する.

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
J_{s_0}(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \left(\sum_{t=0}^T \gamma^t r_t \right)
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

#### 1.3.1 MDP formalized RL
$t=0$に関して、$a_0, s_1$を先に積分する
```math
\begin{align}
J_{s_0}(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right)  \left( r_0 + \gamma G_1 \right) \nonumber \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) r(s_0, a_0, s_1) \nonumber \\
&\quad + \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \gamma G_1 \nonumber \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \pi_\theta(a_0 | s_0) P(s_1 | s_0, a_0) r(s_0, a_0, s_1) \\
&\quad + \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \pi_\theta(a_0|s_0) P(s_{1} | s_0, a_0) \nonumber \\
&\quad \times \sum_{a_1 \in \mathcal{A}} \sum_{s_2 \in \mathcal{S}} \cdots \sum_{a_T \in \mathcal{A}} \sum_{s_{T+1} \in \mathcal{S}} \left( \prod_{t=1}^T \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \gamma G_1 \nonumber \\
&= \sum_{a_0 \in \mathcal{A}} \sum_{s_1 \in \mathcal{S}} \pi_\theta(a_0 | s_0) P(s_1 | s_0, a_0) \left[ r(s_0, a_0, s_1) + \gamma V^{\pi_\theta}_1 (s_1) \right]
\end{align}
```

Eq(17)は以下の規格化条件か課されるから
```math
\begin{equation}
\sum_{a_t \in \mathcal{A}} \sum_{s_{t+1} \in \mathcal{S}}  \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \equiv \sum_{s_{k+1} \in \mathcal{S}}  p(s_{t+1} | s_t)  = 1
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
&\equiv \sum_{a_k \in \mathcal{A}} \pi_\theta(a_k | s_k) Q_k(s_k, a_k)
\end{align}
```
**行動価値関数 (Action-value function, Q function)** $Q_k(s_k, a_k)$ が得られる. Q関数は状態 $s_t$ が与えられた時、行動 $a_t$ をとると、**将来どれくらい報酬をもらえるのか** を評価する関数である. 

強化学習におけるBellman方程式の定番表現は以下となる

```math
\begin{align}
V_{\pi} (s) &= \sum_{a \in \mathcal{A}} \pi_\theta(a | s) \sum_{s' \in \mathcal{S}} P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right] \\
Q(s,a) &= \sum_{s' \in \mathcal{S}} P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right]
\end{align}
```

#### 1.3.2 General (History-based) RL

- $t=0$, $a_0, o_1$を先に積分する(和をとる). $h_t=[o_0, a_0, r_0, o_1, a_1, r_1,\cdots,o_{t-1}, a_{t-1}, r_{t-1}, o_t]$

```math
\begin{align}
J_{o_0}(\pi_\theta) &= \sum_{a_0 \in \mathcal{A}} \sum_{o_1 \in \mathcal{O}} \sum_{a_1 \in \mathcal{A}}\cdots \sum_{a_T \in \mathcal{A}} \sum_{o_{T+1} \in \mathcal{O}}  \left( \prod_{t=0}^T \pi_\theta(a_t|h_t)\mu(o_{t+1}|h_t, a_t)\right) \left(r(h_0, a_0, o_{1}) + \gamma G_1\right) \\
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
V_{\pi} (h) &= \sum_{a \in \mathcal{A}} \pi_\theta(a | h) \sum_{o' \in \mathcal{O}} \mu(o' | h, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] \\
Q(h,a) &= \sum_{o' \in \mathcal{O}} \mu(o' | o, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] \\
h' &= \text{concat}[h, (a, r, o')]
\end{align}
```

### 1.4 積分表現
$\sum$ を $\int$ に直すだけで良い. ただし、この場合 $\pi_\theta(a|s') P(s'|a,s), ~\pi_\theta(a|s')\mu(s'|a,s)$ が確率密度になる.

**MDP formalized RL** :
```math
\begin{align}
J(\pi) &= \int \left( \prod_{t=0}^T da_t ds_{t+1} ~ \pi(a_t|s_t) P(s_{t+1}|s_t, a_t)  \right) \sum_{s=0}^T \gamma^s r_s \\
V_{\pi} (s) &= \int da ~ \pi_\theta(a | s) \int ds' ~ P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right] \\
Q(s,a) &= \int ds' ~ P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right]
\end{align}
```

**General RL** :
```math
\begin{align}
J(\pi) &= \int \left( \prod_{t=0}^T da_t do_{t+1} ~ \pi(a_t|h_t) \mu(o_{t+1}|h_t, a_t)  \right) \sum_{s=0}^T \gamma^s r_s \\
V_{\pi} (h) &= \int da ~ \pi(a | s) \int do' \mu(o'| h, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] \\
Q(h,a) &= \int do' ~ \mu(o' | h, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] \\
\end{align}
```

### 1.5 決定論 vs 確率分布

General RL の場合
- 観測 $o_{t+1}$ は決定論的に決まることがある
  - 運動方程式など
- 同様に、奨励 $r_t$ はランダムで決まることも不可ではない
  - ゲーム内のアイテム確率的に強化など

つまり、前に定義された、観測と奨励は分布 $o_{t+1}, r_t \sim \mu(\cdot |a_t,h_t)$ に従うことが許される. むしろこの定義がより一般的である. そのため、累計報酬は次のように再定義することができる
- 奨励に関する積分を追加するだけ

```math
\begin{equation}
J(\pi) = \int \left( \prod_{t=0}^T da_t do_{t+1} dr_t ~ \pi(a_t|h_t) \mu(o_{t+1}, r_t|h_t,a_t) \right) \sum_{s=0}^T \gamma^s r_s
\end{equation}
```

分布を決定論に直すには、まずDirac関数の積分
```math
\begin{equation}
\int dx ~ \delta(x-a) f(x) = f(a)
\end{equation}
```
を利用して、環境測度 $\mu$ を次のようにつくる
```math
\begin{equation}
\mu(o_{t+1}, r_t | h_t, a_t) = \delta(r_t - R_t(h_t,a_t,o_{t+1})) \delta(o_{t+1}-O_t(h_t, a_t))
\end{equation}
```
$R_t, O_t$ はそれぞれ時刻 $t$ における奨励と観測の決定論的な関数. 特別な決まりがなければ、$\forall t \in \mathbb{N}, R_t\equiv R, O_t\equiv O$ とおく.

```math
\begin{align}
J(\pi) &= \int \left( \prod_{t=0}^T da_t do_{t+1} dr_t ~ \pi(a_t|h_t) \delta(r_t - R_t(h_t,a_t,o_{t+1})) \delta(o_{t+1}-O(h_t,a_t)) \right) \sum_{s=0}^T \gamma^s r_s \nonumber \\
&= \int \left( \prod_{t=0}^T da_t ~ \pi(a_t|h_t)\right) \sum_{s=0}^T \gamma^s R\left(h_s, a_s, O(h_s, a_s) \right)
\end{align}
```


### 1.6 環境測度の分離変数

Hutterが定義された環境測度 $\mu(o_{t+1}, r_t | h_t, a_t)$ は、観測および報酬の同時分布になっている. ただし、観測と採点が別々で行われるの場合、以下の順序がより自然的である (2,3が逆さまでも可能)
1. 過去の履歴 $h_t$ に基づいて、新しい動作を予測 $a_t \sim \pi_\theta(\cdot | h_t)$
2. この動作および過去の履歴で、環境と相互作用し、新しい観測を予測 $o_{t+1}\sim \mu_O(\cdot | h_t, a_t)$
3. 最後に履歴、新しい動作および新しい観測に基づいて採点 $r_t \sim \mu_R ( \cdot | h_t, a_t, o_{t+1})$

以上を同時分布で表現すると

```math
\begin{equation}
p(a_t, o_{t+1}, r_t | h_t) = \pi_\theta (a_t | h_t) \mu_O(o_{t+1} | h_t, a_t) \mu_R (r_t | h_t, a_t, o_{t+1})
\end{equation}
```
$\mu_O, \mu_R$ の積は環境測度そのものである
```math
\begin{equation}
\mu(o_{t+1}, r_t | h_t, a_t) = \mu_O(o_{t+1} | h_t, a_t) \mu_R(r_t | h_t, a_t, o_{t+1})
\end{equation}
```
また、決定論的な場合は以下となる
```math
\begin{align}
\mu_O(o_{t+1} | h_t, a_t) &= \delta(o_{t+1}-O_t(h_t, a_t)) \\
\mu_R(r_t | h_t, a_t, o_{t+1}) &= \delta(r_t - R_t(h_t,a_t,o_{t+1}))
\end{align}
```


```math
\mu(o_{t+1},r_t|h_t,a_t) = \int \mathcal{D}z_{t} \mathcal{D}z_{t+1} ~ q(z_{t+1}|z_t) p(z_{t}|h_t,a_t) \mu_{\text{with\_hidden\_variables}}(o_{t+1},r_t,z_{t+1}|h_t,a_t,z_t)
```


## 1.7 Summary

**強化学習の定式化** :
- History-based RL / General RL: Marcus Hutterによって定式化されたRL
  - 隠れた変数による影響を **確率過程** にモデリングする
  - 新しい観測値およびスコアを推定するには履歴および動作が必要
  - $\mathcal{M} = (\mathcal{O}, \mathcal{A}, \mathcal{H}, \mu, r, \gamma, \pi)$

- Markov Decision Process formalized RL: 強化学習の定番
  - **マルコフ性 (Markov property )** を満たす必要がある
  すなわち 新しい状態は現在の状態しか依存しないこと: $P(s_{t+1}|s_t, a_t)$
  - $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mu, r, \gamma, \pi)$

**最適化目標** : 累積報酬を最大にするように方策モデルを調整する. 
```math
\pi^*_\theta = \underset{\theta}{\operatorname{argmax}} ~J(\pi_\theta)
```

- History-based RL / General RL: 報酬も分布し従うとする
```math
\begin{align*}
J_{o_0}(\pi_\theta) &= \prod_{t=0}^T \left( \sum_{ a_t \in \mathcal{A}} \sum_{o_{t+1} \in \mathcal{O}} \sum_{r_t \in \mathbb{R}} \pi_\theta(a_t|h_t)\mu(o_{t+1}, r_t|h_t, a_t)\right) \left(\sum_{s=0}^T \gamma^s r_s \right) \\
&\equiv \mathbb{E}_{a\sim \pi_\theta, (o,r)\sim\mu} [G_0 | o_0]
\end{align*}
```

- Markov Decision Process formalized RL:
```math
\begin{align*}
J_{s_0}(\pi_\theta) &= \prod_{t=0}^T \left( \sum_{a_t \in \mathcal{A}} \sum_{s_{t+1} \in \mathcal{S}} \pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right) \left(\sum_{s=0}^T \gamma^s r_s \right) \\
&\equiv \mathbb{E}_{a\sim \pi_\theta, s \sim\mu} [G_0 | s_0]
\end{align*}
```

**Bellman 方程式** : 

価値関数 (Value function, V function): 状態 $s_t$ or 履歴 $h_t$ が与えられる時、将来(時刻$t+1$以後)の報酬を予測
- History-based RL / General RL:
```math
V_{\pi} (h) = \sum_{a \in \mathcal{A}} \pi_\theta(a | h) \sum_{o' \in \mathcal{O}} \mu(o' | h, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right]
```
- Markov Decision Process formalized RL:
```math
V_{\pi} (s) = \sum_{a \in \mathcal{A}} \pi_\theta(a | s) \sum_{s' \in \mathcal{S}} P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right]
```


行動価値関数 (Action-value function, Q function): 状態 $s_t$ or 履歴 $h_t$ およびそれに基づいて予測した行動 $a_t$ が与えられる時、将来の報酬を予測
- History-based RL / General RL:
```math
Q(h,a) = \sum_{o' \in \mathcal{O}} \mu(o' | o, a) \left[ r(h, a, o') + \gamma V_{\pi}(h') \right] 
```
- Markov Decision Process formalized RL:
```math
Q(s,a) = \sum_{s' \in \mathcal{S}} P(s' | s, a) \left[ r(s, a, s') + \gamma V_{\pi}(s') \right]
```

# 2. 強化学習の学習手順

## 2.1 従来の RL


## 2.2 LLM RL

### 2.2.1 LLM RL のサンプリング 

LLM RL では、外部の軌跡ステップと自己回帰的な token ステップを区別する必要がある。まずはLLM RL に関わる定義を押さえておく
* $t=0,\ldots,T$ は**外部の軌跡ステップ**を表す。例えば、1 回の QA、1 回の tool call、1 回の環境との相互作用、または 1 回の agent step である。
* $i$ は、外部ステップ $t$ において、**内部で LLM が自己回帰的に生成する token の index**を表す。

$t$ 番目の外部ステップに対応する token 区間を $L_t\le i\le L_{t+1}-1$ とする。また、外部ステップの履歴を $h_{t} = \text{concat} (h_{t-1}, [a_{L_{t-1}}, \cdots a_{L_t-1}, o_{t}, r_{t-1}]) $ とする。このとき、同じ外部ステップ区間内の token-prefix 履歴は $h_{i,t} \equiv \text{concat}(h_t, [a_{L_t},a_{L_t+1},\ldots,a_{i-1}])$ で書ける。このとき、モデルはこの区間内で以下のステップの通り token を生成する
* 第 $i$ 個の token の logit を予測する
  ```math
  z_i^{\mathrm{logit}} = f_\theta(h_{i,t})
  ```
* サンプリングを行う。top-p や top-k などのサンプリング手法はここで使用可能
  ```math
  \pi_{\theta,T_{\mathrm{dec}}}(a_i\mid h_{i,t}) =  \frac{\exp(z^{\mathrm{logit}}_{i,a_i}/T_{\mathrm{dec}})} {\sum_{a'}\exp(z^{\mathrm{logit}}_{i,a'}/T_{\mathrm{dec}})}
  ```
* Token を連結する。すなわち、上で与えた $h_{i,t}$ の定義に従って、新たに生成された token を連結する

したがって、元の**粗粒度**動作 $a_t$ は、自己回帰的に生成される token 列となる。この token 列は、次の同時分布に従う
```math
\pi_\theta(a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}|h_t) = \prod_{i=L_t}^{L_{t+1}-1} \pi_\theta(a_i|h_{i,t})
```

したがって、粗粒度の動作 $a_t$ は、この同時分布からサンプリングされる
```math
\begin{align}
a_t \sim \prod_{i=L_{t}}^{L_{t+1}-1} \pi_\theta(\cdot|h_{i,t}), \quad a_t = [a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}]
\end{align}
```

このため、積分形式における自己回帰構造を含む動作測度は、次のようになる
```math
\begin{equation}
\prod_{t=0}^{T} da_t~ \pi_\theta(a_t|h_{t})  \equiv \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1} da_i~ \pi_\theta(a_i|h_{i,t}) 
\end{equation}
```

LLM が token 列 $a_t$ を生成し、その token 列が環境と相互作用して観測値 $o_{t+1}$ および報酬 $r_t$ を得たとき、履歴は $h_{t+1} = \text{concat}(h_t, [a_t, o_{t+1}, r_t])$ と書くことができるので、LLM の軌跡は従来の粗粒度の形式を保つことができる
```math
\begin{equation}
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
\end{equation}
```

ここでもう一回強調しておくと、ここでの(LLM RL) $a_t$ は単一の token ではなく、自己回帰的に生成された token 列全体である。まとめると、LLM の自己回帰構造は、外部の 1 ステップ $a_t$ をブロック内の token 積分へ展開しているだけであり、その積分を行った後、外側の軌跡変数の表現は依然として $a_t$ のままである。

連続空間での軌跡サンプリングを行う場合は、token hidden を連続自由度として扱うことができる。ブロック内の hidden は次のように書ける。
```math
\begin{equation}
z_t\equiv[z_{L_t},z_{L_t+1},\ldots,z_{L_{t+1}-1}],\quad z_i \in \mathbb{R}^{d_\mathrm{model}}
\end{equation}
```

連続動作の軌跡は、次のように表せる。
```math
\begin{equation}
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T)
\end{equation}
```

ここで $z_t$ も同様に、第 $t$ 個の外部ステップ内部における hidden 列全体を表す。$b$ 番目のサンプルを $b$ で表すと、サンプル軌跡は次のように書ける
```math
\begin{equation}
\tau_b=(a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
\end{equation}
```

LLM RLの場合、累計報酬は次の形で書くことができる。
```math
\begin{align}
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t} 
\end{align}
```

ここで、$\omega^{i-L_t}$ と $\gamma^t$ はそれぞれ token レベルの割引率とタスクステップレベルの割引率を表す。$R(h_{i,t}, a_{i})$ は token レベルの報酬であり、$\phi(h_{i,t}, a_{i}, o_{i+1})$ は段階的な報酬として扱うことができる。特に $t=T$ の場合、これはラスト報酬へ帰着できる。

