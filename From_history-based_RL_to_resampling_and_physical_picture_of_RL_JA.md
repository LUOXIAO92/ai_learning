# History-Based RL から再サンプリングと RL の物理的描像へ

## 0. 目的

この文章では、RL に対する**物理的描像**を組み立てる。RL を厳密な場の理論として定式化し直すことは目標にしない。扱う対象は、経路サンプリングを広げるための物理的手法である。主線は二つある。
- **再サンプリング**: RL の元の累積報酬定義の下で、観測 $o_{t+1}$ と報酬 $r_t$ を再サンプリングまたは再推定し、rollout、収益推定、advantage 推定、PPO / GRPO / GSPO 更新を改善する
- **経路積分**: 経路積分描像の下で、有効 Hamiltonian、Boltzmann 重み、Gibbs サンプリング、MCMC / Langevin、逆温度アニーリングを導入し、経路空間でサンプリングを広げ、低 Hamiltonian の経路を探す

RL と物理は、二つの基本要素を共有する。
- history-based RL は、一次元の時間方向に沿う経路積分として書ける
- 元の期待値に現れる収益 $G[\tau]$ は、経路上の観測量である

---

## 1. History-Based RL の元の軌道積分

最も原始的な形から始める。相互作用履歴を
```math
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\cdots,a_{t-1},r_{t-1},o_t)
```
と書く。

方策を $\pi(a_t\mid h_t)$、環境の条件付き密度を $\mu(o_{t+1},r_t\mid a_t,h_t)$ とする。有限 horizon $T$ の中で、軌道全体の期待収益は次のように書ける。
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
```

経路全体は
```math
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
```
と書ける。

経路収益は
```math
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s
```
である。

元の RL 目的関数は、方策と環境が与える重みによって、可能な全経路を積分する。各経路の値は割引収益 $G[\tau]$ で決まる。

### 1.1. 決定論的環境における Dirac Delta への退化

環境が決定論的なら、$a_t,h_t$ を固定した後、次の観測と報酬は決定論的な関数で決まる。
```math
o_{t+1}=O(a_t,h_t),\quad r_t=R(o_{t+1},a_t,h_t)
```

環境の条件付き密度は Dirac delta に退化する。
```math
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))
```

これを元の軌道積分へ代入する。
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right)
```

Dirac delta の基本的な積分性質を使う。
```math
\int \delta(x-x_0)f(x)\,dx=f(x_0)
```

環境部分が潰れると、行動のサンプリングだけが残る。
```math
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right]
```

ここから次のことが分かる。
- 決定論的環境は経路の分岐を生まない。分岐は方策のサンプリングからしか生じない
- 方策も決定論的なら、経路全体は一本に潰れる

### 1.2. 有限 Horizon と有限報酬
報酬の発散を防ぐには、割引率を $0<\gamma\leq 1$ に置くだけでなく、次の方法も使える。
- 最大経路長を $T\le T_{\max}$ に制限し、経路積分を有限時間区間に限る。
```math
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right)
```

- 報酬をクリップする。
```math
\bar r_t=\mathrm{clip}(r_t,-r_{\max},r_{\max}),\quad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t
```

### 1.3. LLM RL における経路サンプリングと Hidden Action

LLM RL では、外部の軌道ステップと、自己回帰的な token ステップを分けて扱う。この文章では次の約束を使う。
- $t=0,\ldots,T$ は外部の軌道ステップを表す。例えば、一回の問答、一回の tool call、一回の環境との相互作用、一回の agent step である
- $i$ は、外部ステップ $t$ の内部で自己回帰的に生成される token 位置を表す

外部ステップ $t$ に対応する token 区間を $L_t\le i\le L_{t+1}-1$ とする。外部ステップの履歴を $h_{t} = [h_{t-1}, a_{L_{t-1}}, \cdots a_{L_t-1}, o_{t}, r_{t-1}] $ とし、同じ外部ステップ区間の token prefix 履歴を $h_{i,t} \equiv \text{concat}(h_t, [a_{L_t},a_{L_t+1},\ldots,a_{i-1}])$ と定義する。この区間で、モデルは次の手順で token を生成する。
- token $i$ の logit を予測する。
```math
z_i^{\mathrm{logit}} = f_\theta(h_{i,t})
```
- このステップで使える top-p、top-k、その他のデコード戦略に従ってサンプリングする。
```math
\pi_{\theta,T_{\mathrm{dec}}}(a_i\mid h_{i,t}) =  \frac{\exp(z^{\mathrm{logit}}_{i,a_i}/T_{\mathrm{dec}})} {\sum_{a'}\exp(z^{\mathrm{logit}}_{i,a'}/T_{\mathrm{dec}})}
```
- 上の $h_{i,t}$ の定義に従って、新しい token を連結する


したがって、元の粗視化された行動 $a_t$ は、自己回帰 LLM では生成された一つのセグメント全体を指す。token 列は次の同時分布に従う。
```math
\pi_\theta(a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}|h_t) = \prod_{i=L_t}^{L_{t+1}-1} \pi_\theta(a_i|h_{i,t})
```
粗視化された行動 $a_t$ は、この同時分布からのサンプルとして表せる。
```math
a_t \sim \prod_{i=L_{t}}^{L_{t+1}-1} \pi_\theta(\cdot|h_{i,t}), \quad a_t = [a_{L_t},a_{L_t+1},\ldots,a_{L_{t+1}-1}]
```

積分形式では、自己回帰構造の行動測度全体は次の形になる。
```math
\prod_{t=0}^{T} da_t~ \pi_\theta(a_t|h_{t})  \equiv \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1} da_i~ \pi_\theta(a_i|h_{i,t}) 
```

LLM が完全な token 列 $a_t$ を生成し、環境と相互作用して観測 $o_{t+1}$ と報酬 $r_t$ を得た後も、履歴は $h_{t+1} = [h_t, (a_t, o_{t+1}, r_t)]$ と書ける。この約束の下で、LLM の軌道は元の粗視化された形を保つ。
```math
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T)
```

ここでの $a_t$ は、自己回帰的に生成されたセグメントであり、単一 token ではない。自己回帰 LLM の構造は、一つの外部行動 $a_t$ をブロック内の token 積分へ展開する。積分した後、外側の経路変数は stage-level の行動 $a_t$ のまま残る。連続空間で経路をサンプリングしたい場合、token の hidden を連続自由度として扱える。ブロック内 hidden を次のように書く。
```math
z_t\equiv[z_{L_t},z_{L_t+1},\ldots,z_{L_{t+1}-1}],\quad z_i \in \mathbb{R}^{d_\mathrm{model}}
```

連続行動の経路は次のように書ける。
```math
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T)
```

ここで $z_t$ も、外部ステップ $t$ の内部にある hidden 変数のセグメント全体を表す。$b$ が $b$ 番目のサンプルを表すなら、サンプル軌道は
```math
\tau_b=(a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
```
と書ける。

この時点で、報酬は次の形を取る。
```math
\begin{aligned}
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t}
\end{aligned}
```
$\omega^{i-L_t}$ と $\gamma^t$ は、それぞれ token-level discount と task-step-level discount である。$R(h_{i,t}, a_{i})$ は token-level reward であり、$\phi(h_{i,t}, a_{i}, o_{i+1})$ は stage reward として使える。$t=T$ では terminal reward に簡約できる。

---

## 2. 元の History-Based RL の物理的描像

### 2.1. 軌道積分から経路積分へ

まず、元の確率密度の積を経路密度として書く。
```math
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T} \pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)
```

累積報酬の期待値 $J$ は汎関数積分として書ける。
```math
J(\pi)=\int \mathcal{D}\tau ~ P_{\pi,\mu}[\tau]G[\tau],\quad \mathcal{D}\tau = \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
```

ここで作用 $S_{\pi,\mu}[\tau]$ を定義する。
```math
S_{\pi,\mu}[\tau] = - \log P_{\pi,\mu}[\tau] = - \sum_{t=0}^T \left[ \log \pi(a_t|h_t) + \log \mu(o_{t+1},r_t | h_t, a_t) \right]
```
元の軌道積分は経路積分として表せる。
```math
J(\pi)=\int \mathcal{D}\tau ~ e^{-S_{\pi,\mu}[\tau]} G[\tau]
```

LLM の自己回帰生成では、Section 1.3 の約束に従って
```math
da_t\equiv\prod_{i=L_t}^{L_{t+1}-1}da_i,\quad
\pi(a_t\mid h_t)\equiv\prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t})
```
と置く。

したがって LLM の経路密度は
```math
P^{AR}_{\pi,\mu}[\tau] = \prod_{t=0}^{T} \left[ \prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t}) \right]
\mu(o_{t+1},r_t\mid h_t,a_t)
```
となる。

作用は
```math
S^{AR}_{\pi,\mu}[\tau] = - \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \left( \log \pi(a_i|h_{i,t}) + \delta_{i,L_{t+1}-1} \log \mu(o_{t+1},r_t | h_t, a_t) \right)
```
である。

時刻 $t$ の propagator は
```math
\begin{aligned}
f(h_{t+1}, h_t) &= \left[\prod_{i=L_t}^{L_{t+1}-1} \pi(a_i|h_{i,t})\right]\mu(o_{t+1},r_t|h_t,a_t)\\
\Delta h_t &= [a_t,o_{t+1},r_t] ,~ h_{t+1} = \text{concat}(h_t, \Delta h_t)
\end{aligned}
```
と書ける。

対応する測度は
```math
\mathcal{D}\tau = \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \equiv \prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right] do_{t+1}\,dr_t
```
である。

経路積分表現の下で、強化学習には次の物理的解釈がある。
- $a_t,o_t,r_t$: 一次元時間経路上の場の自由度。ここでの場はスカラー場、つまり行動場、観測場、収益場である
- $\mathcal{D}\tau$: 全格子点上の場の測度
- $G[\tau]$: 物理的観測量
- $f(h_{t+1},h_t)$: 伝播子。history-based RL では履歴が状態を定義するため、ここでの変分は無限小の場の変分ではない。変化は系列 $\Delta h_t = h_{t+1}-h_{t}\equiv [a_t, o_{t+1}, r_t]$ として入る
- $e^{-S_{\pi,\mu}[\tau]}$: 経路積分の重み。$S_{\pi,\mu}$ が作用である
- History-based RL は、三つの自由度 $a,o,r$ を持つ複雑な一次元一粒子系と見なせる。複雑さは履歴軌道 $h_t$ に基づく長距離結合から来る。この結合は $w_{o_0,a_0,o_1,r_0,\cdots,a_t,o_{t+1},r_t}$ のような単純な係数ではない。方策 $\pi$ と環境測度 $\mu$ が結合を決める。
- 経路密度の負対数 $- \log P_{\pi,\mu}$ を作用と定義したが、逆温度 $\beta$ によって全体結合の強さを制御できるため、作用は再定義できる。
```math
S_{\pi,\mu}[\tau] = \beta H_{\pi,\mu}[\tau],\quad H_{\pi,\mu}[\tau] \equiv - \log P_{\pi,\mu}
```

### 2.2. 割引率と Laplace 正則化

連続時間の収益を

```math
G[\tau]=\int_0^\infty r(t)\,dt
```

と書くと、発散する可能性がある。指数減衰を加えると

```math
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt
```

となる。

離散時間では

```math
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t
```

と書く。時間刻みを $\Delta t$ とすると、対応は $\gamma=e^{-\lambda\Delta t}, ~\gamma^t=e^{-\lambda t\Delta t}$ である。報酬が有界で、$|r_t|\le r_{\max}$ なら、割引収益も有界になる。
```math
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}
```

---

## 3. 元の RL 定義の下での観測と報酬の再サンプリング

元の RL の累積報酬定義から出発すると、方策更新器を変えずに、観測 $o_{t+1}$ と報酬 $r_t$ の局所的なサンプリング空間を再サンプリングで広げられる。これによりモデルは経路を探索しやすくなる。環境測度の定義から、潜在変数を観測値や報酬へのランダム摂動として扱えば、確率性を導入できる。LLM RL では観測と報酬が決定論的になりやすい。その場合、経路探索は方策モデルに強く依存し、探索能力が制限される。そこで観測と報酬にランダム摂動を入れ、LLM に広い探索空間を与える。

- 注: tool call など一部の task では、返ってきた観測値にランダム摂動を加えることが難しい、または不可能である

より複雑な方法では、実環境からの反復サンプリング、world model、reward model、verifier、SMC、CEM 提案分布を使える。この節ではそれらの再サンプリング法を扱わない。ここでは基本的な Gaussian noise 提案分布を使う。

### 3.1. 決定論的条件下での観測 $o$ と報酬 $r$ の近似
Section 1 では、$\delta$ 関数により、ランダムな観測と報酬が決定論的な値へ退化した。この値は決定論的な物理モデルまたは数学モデルから得られる。ここでは $\delta$ 関数の定義を使う。
```math
\delta(x-x_0) = \lim_{\sigma\rightarrow 0}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( -\frac{(x-x_0)^2}{2\sigma^2} \right)
```
まず小さい $\sigma$ を取り、決定論的な値を近似的な「確率分布」へ広げる。極小分散の提案分布は、分散ゼロの Dirac delta を近似できる。このとき、$o_{t+1}, r_t$ は近似的に
```math
o_{t+1},r_t \sim \mu_\sigma(\cdot|h_t,a_t) \quad\text{or}\quad o_{t+1}\sim \mu_{\sigma_O}(\cdot|h_t,a_t),~ r_t \sim \mu_{\sigma_R}(\cdot|h_t, a_t, o_{t+1})
```
に従う。

決定論的な観測 $o$ と報酬 $r$ を平均とする Gaussian 分布を近似として選ぶ。ランダム過程 $o\rightarrow o', r\rightarrow r'$ の遷移確率は
```math
q(o'|o) = \frac{1}{\sqrt{2\pi\sigma^2_O}}\exp\left( -\frac{(o'-o)^2}{2\sigma^2_O} \right) ,\quad q(r'|r) = \frac{1}{\sqrt{2\pi\sigma^2_R}}\exp\left( -\frac{(r'-r)^2}{2\sigma^2_R} \right)
```
である。

この近似から、simulated annealing を自然に導入できる。まず **scalar** case を考える。$\beta$ を逆温度とする。このとき
```math
q_{\beta} (x'|x) = \sqrt{\frac{\beta}{2\pi\sigma^2}} \exp\left( - \beta \frac{(x'-x)^2}{2\sigma^2} \right)
```
となる。

有効分散は $\sigma^2_{\text{eff}}(\beta) = \sigma^2 / \beta$ である。simulated annealing では、逆温度を調整してサンプリング振幅を制御できる。近似 Gaussian 分布を標準正規分布の形に書き直す。
```math
q(\xi) = \frac{1}{\sqrt{2\pi}} \exp\left(- \frac{\xi^2}{2} \right), \quad \xi^2 = \beta\frac{(x'-x)^2}{\sigma^2}
```

新しい変数 $x' = x + \xi / \sqrt{\beta} ,~ \xi \sim \mathcal{N}(0, \sigma^2)$ を使って Gaussian sampling を行える。手順は、1. 標準正規分布から乱数を生成する、2. 新しい $x'$ を計算する、である。

次に多変量 Gaussian の場合へ拡張する。$\boldsymbol{x} = (x_1, \cdots, x_d)$ とし、$\Sigma$ を正定値共分散行列とする。
```math
q_{\beta}(\boldsymbol{x}'|\boldsymbol{x}) = \frac{\beta_k^{d/2}} {(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{\beta}{2} (\boldsymbol{x}'-\boldsymbol{x})^T\Sigma^{-1}(\boldsymbol{x}'-\boldsymbol{x}) \right)
```

$\Sigma$ の全固有値が 0 に近づくと、$\beta=1$ でこの式は multidimensional Dirac delta $q_{\beta}(\boldsymbol{x}'|\boldsymbol{x}) \rightarrow \delta^{(d)}(\boldsymbol{x}'-\boldsymbol{x})$ に退化する。決定論的なベクトルに simulated annealing を使う場合、共分散行列を $\Sigma = \sigma^2 I + \epsilon^2 (U U^{T} - \mathrm{diag}{UU^T})$ と構成できる。ここで $I$ は単位行列、$U$ は正規化されたランダム実行列、$\epsilon$ は小さい非対角相関を与える量で、$0 < \epsilon \ll \sigma$ を満たす。新しい分布は $\boldsymbol{x}' = \boldsymbol{x} + \boldsymbol{\eta} / \sqrt{\beta},~ \boldsymbol{\eta}\sim \mathcal{N}(0, \Sigma)$ となる。

上の議論は、決定論的なベクトル値の観測と報酬を対象にする。特に LLM 強化学習では、実際の訓練環境からずれる場合がある。それでも参照用の描像として使える。


### 3.2. ランダム再サンプリングと PPO / GRPO / GSPO の接続

前節では、決定論的条件下の観測と報酬を、小さい分散を持つ有効環境測度として近似した。ここでその近似を元の RL 更新過程へ接続する。元の PPO / GRPO / GSPO 更新器は変えない。変化は rollout 中に起こる。元は環境測度 $\mu$ が与えていたフィードバックを、小分散の有効環境測度 $\mu_\sigma$ が与える。

元の環境測度の下では、方策と環境が共同で経路を生成する。
```math
\tau_b = (a_{b,0},o_{b,1},r_{b,0},a_{b,1},o_{b,2},r_{b,1},\ldots,a_{b,T_b},o_{b,T_b+1},r_{b,T_b})
```

ここで $b$ はサンプル番号である。Gaussian 近似された有効環境測度の下では、対応する経路は
```math
\tau'_b = (a'_{b,0},o'_{b,1},r'_{b,0},a'_{b,1},o'_{b,2},r'_{b,1},\ldots,a'_{b,T_b},o'_{b,T_b+1},r'_{b,T_b})
```
となる。

prime は、摂動を受けた環境フィードバックの下で生成された経路を表す。摂動後の観測は次ステップの履歴に入る。
```math
h'_{b,t+1}=\mathrm{concat}(h'_{b,t},a'_{b,t},o'_{b,t+1},r'_{b,t})
```

方策は次の行動を生成し続ける。ただし、条件付ける対象は摂動後の履歴になる。
```math
a'_{b,t}\sim \pi_\theta(\cdot\mid h'_{b,t})
```

scalar approximation では、観測と報酬の摂動を次のように書ける。
```math
\begin{aligned}
o'_{b,t+1} &= o_{b,t+1} + \sigma_O\xi_{O,b,t}, \quad \xi_{O,b,t} \sim \mathcal N(0,1) \\
r'_{b,t} &= r_{b,t} + \sigma_R\xi_{R,b,t}, \quad \xi_{R,b,t} \sim \mathcal N(0,1)
\end{aligned}
```

報酬を摂動後の観測から再計算する場合は
```math
r'_{b,t} = R(o'_{b,t+1},a'_{b,t},h'_{b,t})
```
と書く。

その結果、経路収益は
```math
G[\tau'_b] = \sum_{t=0}^{T_b} \gamma^t r'_{b,t}
```
となる。

Section 1.3 の LLM 自己回帰型の収益形式を使うなら
```math
G[\tau'_b] = \sum_{t=0}^{T_b} \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r'_{b,i,t}
```
である。

ここで $r'_{b,i,t}$ は、摂動後のフィードバックの下で、外部ステップ $t$ 内の token $i$ に割り当てる訓練報酬である。報酬を token-level reward と stage reward に分解すると
```math
r'_{b,i,t} = \omega^{i-L_t}R(h'_{b,i,t},a'_{b,i}) + \delta_{i,L_{t+1}-1}\phi(h'_{b,t},a'_{b,t},o'_{b,t+1})
```
と書ける。

ここから先では、表記を簡潔にするため prime を省く。以下の摂動後の量には、元の記号 $h, a, r, o$ などを使う。PPO を使う場合、摂動経路 $\tau_b$ から新しい収益と advantage を作る。摂動経路上の advantage を $A_{b,i,t}$ とする。
```math
\begin{aligned}
A_{b,i,t} &= Q(h_{b,i,t},a_{b,i}) - V(h_{b,i,t}) \\
Q(h_{b,i,t},a_{b,i}) &\simeq \widehat{G}_{b,i,t} = \sum_{j=i}^{L_{t+1}-1} r_{b,j,t} + \sum_{s=t+1}^{T} \gamma^{s-t} \sum_{j=L_s}^{L_{s+1}-1} r_{b,j,s}
\end{aligned}
```


方策比は、同じ行動に対する現在方策と旧方策の確率比をそのまま使う。
```math
\rho_{b,t}(\theta) = \frac{ \pi_\theta(a_{b,t}\mid h_{b,t}) }{ \pi_{\theta_{\mathrm{old}}}(a_{b,t}\mid h_{b,t}) }
```

PPO の clipped objective は
```math
L_{\mathrm{PPO}} = \mathbb E_{b,t} \left[ \min \left( \rho_{b,t}(\theta)A_{b,t}, \mathrm{clip}(\rho_{b,t}(\theta),1-\epsilon,1+\epsilon)A_{b,t} \right) \right]
```
と書ける。

LLM の自己回帰生成では、粗視化された行動 $a_{b,t}$ は token 列である。そのため、方策比は token-level の確率比の積へ展開する。
```math
\rho_{b,t}^{\mathrm{AR}}(\theta) = \prod_{i=L_t}^{L_{t+1}-1} \frac{ \pi_\theta(a_{b,i}\mid h_{b,i,t}) }{ \pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t}) }
```

同じことを log-ratio の和としても書ける。
```math
\log\rho_{b,t}^{\mathrm{AR}}(\theta)=\sum_{i=L_t}^{L_{t+1}-1}\left[\log\pi_\theta(a_{b,i}\mid h_{b,i,t})-\log\pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t})\right]
```

GRPO / GSPO でも同じ構造を使える。同じ入力 $x$ に対して、方策は摂動フィードバックの下で経路の group $\tau_1,\tau_2,\ldots,\tau_{K_s}$ を生成する。各経路はそれぞれの収益 $G_b = G[\tau_b],~ b=1, \ldots, K_s$ を持つ。group 内平均と分散は
```math
\bar G = \frac{1}{K_s} \sum_{b=1}^{K_s}G_b, \quad (\sigma_G)^2 = \frac{1}{K_s} \sum_{b=1}^{K_s} (G_b-\bar G)^2
```
である。

group 内で正規化した advantage は
```math
A_b = \frac{G_b-\bar G}{\sigma_G+\epsilon}
```
である。

GSPO では、生成されたシーケンス全体をサンプリング単位として扱える。各サンプルの収益 $G_b$ は完全なシーケンス経路 $\tau_b$ から得る。一方、方策更新はシーケンス内の各 token の log-prob または log-ratio を通じてモデルパラメータに作用する。シーケンス $b$ の内部 token を展開すると、sequence-level log-ratio は
```math
\log\rho_b^{\mathrm{seq}}(\theta) = \sum_{t=0}^{T_b} \sum_{i=L_t}^{L_{t+1}-1} \left[ \log\pi_\theta(a_{b,i}\mid h_{b,i,t}) - \log\pi_{\theta_{\mathrm{old}}}(a_{b,i}\mid h_{b,i,t}) \right]
```
となる。

Gaussian 近似した環境フィードバックは、PPO / GRPO / GSPO に直接接続できる。rollout 経路上の観測、報酬、収益は変わるが、方策更新は元の ratio、advantage、clipping、group normalization の構造を使い続ける。

### 3.3. Simulated Annealing

前節では、決定論的な観測と報酬を小分散の Gaussian 提案分布へ広げた。ここで simulated annealing を導入し、temperature scheduler によってこの提案分布の摂動振幅を制御する。この操作は観測と報酬の Gaussian 摂動に作用し、フィードバック空間における rollout の探索幅を調整する。

round $k$ の逆温度を
```math
\beta_k=\mathcal B(k),\quad \beta_k>0
```

とする。$B(k)$ は手で指定する annealing scheduler である。scalar 変数 $x$ に対して、round $k$ の Gaussian 提案分布は
```math
q_{\beta_k}(x'\mid x) = \sqrt{\frac{\beta_k}{2\pi\sigma^2}} \exp\left( -\frac{\beta_k(x'-x)^2}{2\sigma^2} \right)
```
と書ける。

対応するサンプリング形式は
```math
x^{(k)} = x+\frac{\sigma}{\sqrt{\beta_k}}\xi_k, \quad \xi_k \sim \mathcal N(0,1) 
```
である。

有効分散は
```math
\sigma_{\mathrm{eff}}^2(k) = \frac{\sigma^2}{\beta_k}
```
である。

したがって、小さい $\beta_k$ は広い提案分布を作り、rollout は観測空間と報酬空間で大きくずれる。大きい $\beta_k$ は狭い提案分布を作り、経路は元の決定論的フィードバックに近づく。逆温度ではなく温度 $T_k$ を使うなら
```math
T_k=\frac{1}{\beta_k}, \quad \sigma_{\mathrm{eff}}^2(k)=\sigma^2T_k
```
となる。

多次元変数 $\boldsymbol{x}$ と共分散行列 $\Sigma$ に対して、round $k$ の annealed proposal は
```math
q_{\beta_k}(\boldsymbol{x}'\mid\boldsymbol{x}) = \frac{\beta_k^{d/2}} {(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{\beta_k}{2} (\boldsymbol{x}'-\boldsymbol{x})^T\Sigma^{-1}(\boldsymbol{x}'-\boldsymbol{x}) \right)
```
である。

対応する有効共分散は
```math
\Sigma_{\mathrm{eff}}(k) = \frac{1}{\beta_k}\Sigma
```
である。

simulated annealing では、base Gaussian 提案分布の形を固定し、scheduler $\mathcal{B(k)}$ で全体のスケールを制御する。monotone cooling は early rollout に大きなフィードバック摂動を与え、その後、摂動を元のフィードバックへ縮める。cyclic heating and cooling は局所的に収縮した後で摂動を再び広げ、quenching を模倣し、局所領域から抜ける可能性を上げる。

### 3.4. 統計力学的解釈

Section 2 の物理的描像から、元の history-based RL の経路重みはすでに Boltzmann 重みとして書ける。
```math
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)=\exp(-\beta H_0[\tau])
```

ここで $H_0[\tau]$ は、元の方策と環境が誘導する base Hamiltonian であり、$\beta$ は逆温度である。LLM の自己回帰形式では、方策部分は
```math
P_{\pi,\mu}^{\mathrm{AR}}[\tau]= \prod_{t=0}^{T} \left[ \prod_{i=L_t}^{L_{t+1}-1}\pi(a_i\mid h_{i,t}) \right] \mu(o_{t+1},r_t\mid h_t,a_t) =\exp(-\beta H_0^{\mathrm{AR}}[\tau])
```
と展開される。

Sections 3.1 から 3.3 では、token 生成測度そのものを変えずに $o_{t+1}$ と $r_t$ を再サンプリングした。LLM の場合、$a_t$ は自己回帰的な token 列と読めばよい。
```math
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
```

収益は経路上の統計的観測量のまま残る。
```math
G[\tau_b]=\sum_{t=0}^{T}\gamma^t r_{b,t} \quad \text{or} \quad G[\tau_b] = \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{b,i,t}
```

同じ入力に対してサンプリングした経路 group は、局所摂動を含む元の過程、または rollout を拡張した過程の下の経路分布と見なせる。つまり $\tau_b\sim P_{\pi,\mu}[\tau],~ b=1,\ldots,K_s$ であり、$b$ はサンプル番号である。group 内の収益平均と分散は
```math
\bar G=\frac{1}{K_s}\sum_{b=1}^{K_s}G[\tau_b],\quad \sigma_G^2=\frac{1}{K_s}\sum_{b=1}^{K_s}(G[\tau_b]-\bar G)^2
```
である。

ここで $\bar G$ は group 内の平均収益観測量、$\sigma_G^2$ は収益観測量の揺らぎの強さである。

Section 3.3 では、$\beta_k$ が round $k$ の simulated annealing における逆温度パラメータとして直接働く。これを Section 2 の熱力学的描像と合わせるため、$\beta_k$ を逆温度 $\beta$ と round $k$ の局所再サンプリング強度 $\alpha_{\,\mathrm{res}}^{(k)}$ の積として扱う。
```math
\beta_k=\beta\alpha_{\mathrm{res}}^{(k)}
```

ここで $\alpha_{\mathrm{res}}^{(k)}$ は、局所的な観測 / 報酬摂動ポテンシャルの強度である。以下では簡潔に $\beta_k$ と書く。元の rollout 経路 $\tau$ が与えられると、Gaussian 再サンプリングによって経路 $\tau^k=(a_0,o_1^{(k)},r_0^{(k)},\ldots,a_T,o_{T+1}^{(k)},r_T^{(k)})$ が得られる。これが LLM の自己回帰経路なら、各 $a_t$ は同じ token 列のままであり、再サンプリングは $o_{t+1}$、$r_t$、またはそれらの連続表現にだけ作用する。観測の Gaussian 再サンプリングは、各局所時刻 $t$ における二次 Hamiltonian に対応する。同じスケールの独立 Gaussian 摂動を使うなら
```math
H_o(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{(o_{t+1}^{(k)}-o_{t+1})^2}{2\sigma_o^2}
```
と書く。

観測 noise の共分散行列 $\Sigma_o$ を使うなら

```math
H_o(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{1}{2}(o_{t+1}^{(k)}-o_{t+1})^\top\Sigma_o^{-1}(o_{t+1}^{(k)}-o_{t+1})
```

となる。

報酬にも Gaussian 摂動を直接加えるなら、報酬摂動 Hamiltonian を加える。
```math
H_r(\tau^{(k)}\mid\tau)=\sum_{t=0}^{T}\frac{(r_t^{(k)}-r_t)^2}{2\sigma_r^2}
```

再サンプリング Hamiltonian は
```math
H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)=H_o(\tau^{(k)}\mid\tau)+H_r(\tau^{(k)}\mid\tau)
```
の形を取る。

報酬に直接 noise を入れず、摂動後の観測から再計算する場合は、$H_o$ だけを残し、$r_t^{(k)}=R(o_{t+1}^{(k)},a_t,h_t)$ と置く。

元の経路と再サンプリング経路の joint weight は
```math
P_k(\tau,\tau^{(k)})\propto \exp\left(-\beta H_0[\tau]-\beta_k H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)\right)
```
である。

したがって、サンプリング拡張された作用は

```math
S_{\mathrm{augmented}}^{(k)}[\tau,\tau^{(k)}]=\beta H_0[\tau]+\beta_k H_{\mathrm{resampled}}(\tau^{(k)}\mid\tau)
```

となる。

この式から、次の統計力学的解釈が得られる。
- base Hamiltonian $H_0[\tau]$ は元の経路を重み付けする。その経路の近傍で観測と報酬を再サンプリングすると、局所摂動 Hamiltonian $H_{\mathrm{resampled}}$ を通じて熱揺らぎが生じる
- $\beta_k$ は局所揺らぎの強度を制御する。小さい $\beta_k$ は局所的な高温に対応し、再サンプリング経路は元の経路の近傍で大きな熱揺らぎを持つ。後で $\beta_k$ が大きくなると局所的な低温に対応し、再サンプリング経路は元の観測と報酬へ縮む

### 3.5. Related Work

- $(o,r)$ resampling
  - 研究方向: Gaussian noise proposal / noisy environment augmentation / observation noise / reward noise
  - 直接関連する文献:
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
    - [[arXiv:2106.11420] Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/abs/2106.11420)
    - [[arXiv:1810.01032] Reinforcement Learning with Perturbed Rewards](https://arxiv.org/abs/1810.01032)
    - [[PMLR 2020] Deep Reinforcement Learning with Robust and Smooth Policy](https://proceedings.mlr.press/v119/shen20b.html)
  - 近い方向の文献:
    - [[arXiv:2310.00344] HarmonyDream: Task Harmonization Inside World Models](https://arxiv.org/abs/2310.00344)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)

- Within-group statistics
  - 研究方向: GRPO / GSPO
  - 直接関連する文献:
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
  - 近い方向の文献:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- Proposal expansion
  - 研究方向: SMC policy optimization / CEM / iCEM
  - 直接関連する文献:
    - [[arXiv:2402.07963] SPO: Sequential Monte Carlo Policy Optimisation](https://arxiv.org/abs/2402.07963)
    - [[arXiv:2505.16732] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](https://arxiv.org/abs/2505.16732)
    - [[arXiv:2008.06389] Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/abs/2008.06389)
    - [[arXiv:2112.07746] CEM-GD: Cross-Entropy Method with Gradient Descent Planner for Model-Based Reinforcement Learning](https://arxiv.org/abs/2112.07746)
  - 近い方向の文献:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)

- 元の RL estimator と PPO / GRPO / GSPO の接続
  - 研究方向: PPO-family + noisy rollout augmentation / group-level RL
  - 直接関連する文献:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
  - 近い方向の文献:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)


---

## 4. 経路積分に基づく経路サンプリング

Section 3 では、元の RL 更新器を変えずに観測 $o_{t+1}$ と報酬 $r_t$ へ局所 Gaussian 摂動を加え、rollout のフィードバック空間を広げた。この方法は、元の方策が生成する経路の近傍を拡張する。主に環境フィードバックを変える方法であり、経路空間全体でより良い経路を直接探す方法ではない。

ここでは視点を変える。経路積分表現から出発し、軌道全体 $\tau$ そのものをサンプリング対象として扱う。経路の質は、単一ステップの報酬や局所摂動だけでは決まらない。経路全体の作用と目標観測量が一緒に決める。サンプル経路には、タスク完了だけでなく、KL、長さ、形式、リソースなどの制約も満たしてほしい。そのような経路は、より低い有効作用に対応するか、より安定した低エネルギー領域に属する。この理由で経路積分を導入する。

そこで、元の作用 $S_0[\tau]$ にソースタームを加え、報酬、罰則、制約を目標観測量 $F[\tau]$ にまとめる。これによりソースターム付きの Gibbs 分布が得られる。この分布は経路サンプリングを高報酬かつ低罰則の領域へ偏らせる。次に MCMC や Langevin などのサンプリング法を使う。目的は $a,o,r$ を別々の状態として局所ランダムウォークさせることではない。経路空間全体で低作用経路を探すことである。MCMC は局所的な低点にはまった経路にも抜け出す確率を残す。Langevin は連続的な隠れ状態経路上で作用の降下とランダム摂動を組み合わせる。安定した高品質経路は、摂動後に目標から離れていく経路ではなく、タスクを完了できる領域へ戻る経路盆地に属する。

### 4.1. 復習: RL の経路積分表現

Section 2 では RL の経路積分表現を与えた。
```math
J(\pi)=\int \mathcal{D}\tau ~ e^{-S_{\pi,\mu}[\tau]} G[\tau]
```
自己回帰形式と非自己回帰形式がある。自己回帰形式は LLM の next-token prediction に対応し、複数の新しい token が一つの新しい行動 $a_t$ を形成する。非自己回帰形式は従来の next-action prediction に対応し、モデルが行動 $a_t$ を直接生成する。二つの形式における作用、測度、観測量は次の通りである。
- 自己回帰 (Autoregressive):
```math
\begin{aligned}
S^{AR}_{\pi,\mu}[\tau] &= - \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \left( \log \pi(a_i|h_{i,t}) + \delta_{i,L_{t+1}-1} \log \mu(o_{t+1},r_t | h_t, a_t) \right) \\
\mathcal{D}\tau &= \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \equiv \prod_{t=0}^{T}
\left[\prod_{i=L_t}^{L_{t+1}-1}da_i\right] do_{t+1}\,dr_t\\
G[\tau] &= \sum_{t=0}^T \gamma^t \left[ \sum_{i=L_t}^{L_{t+1}-1}  \left(\omega^{i-L_t} R(h_{i,t}, a_{i}) + \delta_{i,L_{t+1}-1}\phi(h_{t}, a_{t}, o_{t+1})\right) \right] \\
&\equiv \sum_{t=0}^T \sum_{i=L_t}^{L_{t+1}-1} \gamma^t r_{i,t}
\end{aligned}
```

- 非自己回帰 (Non-autoregressive):
```math
\begin{aligned}
S_{\pi,\mu}[\tau] &= - \sum_{t=0}^T \left[ \log \pi(a_t|h_t) + \log \mu(o_{t+1},r_t | h_t, a_t) \right] \\
\mathcal{D}\tau &= \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t \\
G[\tau] &= \sum_{t=0}^T \gamma^t r_{t}
\end{aligned}
```

この表現の下で、強化学習には次の物理的解釈がある。
- $a_t,o_t,r_t$: 一次元時間経路上の場の自由度。ここでの場はスカラー場、つまり行動場、観測場、収益場である
- $\mathcal{D}\tau$: 全格子点上の場の測度
- $G[\tau]$: 物理的観測量
- $e^{-S_{\pi,\mu}[\tau]}$: 経路積分の重み。$S_{\pi,\mu}$ が作用である
- History-based RL: 三つの自由度 $a,o,r$ を持つ複雑な一次元一粒子系。複雑さは履歴軌道 $h_t$ に基づく長距離結合から来る。方策 $\pi$ と環境測度 $\mu$ が結合を決める。
- 等価な元の Hamiltonian: $S_{\pi,\mu}[\tau] = \beta H_{\pi,\mu}[\tau] \equiv S_0[\tau] = \beta H_0[\tau],~ H_{\pi,\mu}[\tau] = - \log P_{\pi,\mu}$


### 4.2. ソースタームと Gibbs 分布
場の理論におけるソースタームを導入し、RL の物理的描像との対応を作る。ソースターム付きの経路積分は
```math
Z[\eta] = \int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]}
```
である。
ソース係数 $\eta$ で微分すると
```math
\begin{aligned}
\frac{\partial}{\partial \eta} \log Z[\eta] &= \frac{1}{Z[\eta]} \int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} F[\tau] = \mathbb{E}_\eta [F[\tau]] \\
\frac{\partial^2}{\partial^2 \eta} \log Z[\eta] &= \text{Var}_\eta [F[\tau]]
\end{aligned}
```
となる。
$\eta = 0$ のとき、元の作用の下で観測量の期待値と分散が得られる。RL では、観測量をソースタームとして使い、さらに長さ罰則や KL 罰則などの罰則項を導入できる。
```math
F[\tau; \lambda_G, \lambda_N, \lambda_{KL}] = \lambda_G G[\tau] - \lambda_N N[\tau] - \lambda_{KL} K[\tau]
```
KL 罰則には次を選べる。
- 厳密な KL divergence。全分布を使う。
```math
K[\tau] = \sum_{t=0}^T \sum_{i=L_{t}}^{L_{t+1}-1} \int d\tilde{a}_i ~ \pi(\tilde{a}_i|h_{i,t}) \log \frac{\pi(\tilde{a}_i|h_{i,t})}{\pi_\mathrm{ref}(\tilde{a}_i|h_{i,t})}
```
- sampled KL divergence。サンプルされた行動だけを使う。
```math
K[\tau] = \sum_{t=0}^T \sum_{i=L_{t}}^{L_{t+1}-1} \log \frac{\pi(a_i|h_{i,t})}{\pi_\mathrm{ref}(a_i|h_{i,t})}
```

$\eta$ の下での軌道 $\tau$ の確率分布は
```math
q_{\eta}(\tau) = \frac{1}{Z[\eta]} \exp\left( - S_0[\tau] + \eta F[\tau] \right)
```
である。

Hamiltonian の定義を使うと、有効 Hamiltonian $H_{\eta}[\tau] = H_0[\tau] - \frac{\eta}{\beta}F[\tau]$ が得られる。したがって
```math
q_\eta[\tau] \propto \exp\left( - \beta H_0[\tau] + \eta F[\tau] \right)
```
である。

この分布は統計力学の Gibbs 分布に対応する。罰則項は化学ポテンシャルなどの等価ポテンシャルと見なせる。Gibbs 分布は偏りを持つ分布であり、ソースタームは経路分布を高収益かつ低罰則の方向へ寄せる。これは RL の目的と対応する。元の分布へ戻すには再重み付けを使う。
```math
\begin{aligned}
& \mathbb{E}_{\eta=0} \left[F[\tau] \right] = \frac{\int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} e^{- \eta F[\tau]} F[\tau]}{\int \mathcal{D}\tau ~ e^{-S_0[\tau] + \eta F[\tau]} e^{- \eta F[\tau]}}  \\
=~& \frac{\mathbb{E}_{\eta} \left[F[\tau]e^{- \eta F[\tau]} \right]}{\mathbb{E}_{\eta} \left[e^{- \eta F[\tau]} \right]} = \frac{\sum_{b}F[\tau_b]e^{-\eta F[\tau_b]}}{\sum_b e^{-\eta F[\tau_b]}}
\end{aligned}
```
別の方法として、$\eta \rightarrow 0$ への外挿を使える。

### 4.3. MCMC 経路サンプリング

現在の経路 $\tau$ が与えられたとき、MCMC の候補経路を、生成済みの完全経路に対する任意の摂動として理解してはいけない。history-based RL の経路には因果構造がある。後続の観測と収益は、先行する履歴と行動に依存する。ある行動だけを直接変え、後続の観測と報酬を保つと、後半は新しい行動の下で合法なフィードバックでなくなる可能性がある。したがって、合法な提案分布は候補経路 $\tau'$ を生成しなければならない。

提案分布は行動接頭列だけを摂動してもよいし、行動、摂動を許す観測、ソフト収益を一緒に摂動してもよい。観測または報酬だけを摂動する方法も使える。ただし、これらの場がモデルの後続の意思決定にあまり効かない場合、モデルは同じ行動を生成し続け、経路分岐はあまり変わらない。より一般的な候補経路の提案分布は同時分布の形で書ける。
```math
\begin{aligned}
& (c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}}) \sim q_{\mathrm{prop}}(c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}}\mid \tau) \\
& \tau' = \mathrm{Rollout}(c'_a,o'_{\mathrm{allow}},r'_{\mathrm{soft}})
\end{aligned}
```

ここで $`c'_{a}`$ は摂動後の行動接頭列を表す。token 接頭列、隠れ状態接頭列、tool-call 接頭列、agent-step 接頭列などを取りうる。$`o'_{\mathrm{allow}}`$ は**摂動を許す**観測場を表す。$`r'_{\mathrm{soft}}`$ はソフトスコア、verifier スコア、reward-model スコアなど、摂動可能な収益またはスコアを表す。行動だけを摂動するなら、$`o'_{\mathrm{allow}}=o_{\mathrm{allow}}`$ と $`r'_{\mathrm{soft}}=r_{\mathrm{soft}}`$ を固定する。共同摂動を使うなら、三つを一緒にサンプリングする。

LLM の自己回帰の場合、行動接頭列は $`c_{a,i,t} = (h_t,a_{L_t},a_{L_t+1},\cdots,a_i)`$ と書ける。摂動後の接頭列は $`c'_{a,i,t} \sim q_a(c'_{a,i,t}\mid c_{a,i,t})`$ である。摂動を許す観測と報酬はマスク形式で書ける。
```math
\begin{aligned}
o'_{t+1} &= \mathcal E_o(o_{t+1};m_o,\xi_o) \\
r'_t &= \mathcal E_r(r_t;m_r,\xi_r)
\end{aligned}
```
ここで $\mathcal{E}_o ,~ \mathcal{E}_r$ はマスクで制約されたランダム摂動演算子である。特定の数値への Gaussian 摂動でもよいし、場レベルのランダム編集でもよい。作用する対象は、摂動を許す観測場またはソフトスコアに限る。マスク $m_o,m_r$ は、摂動を許す場だけを開く。数値型の返り値、信頼度、ソフトスコア、メタデータ、事実的意味に影響しない補助場は、提案分布の一部になりうる。変更してはいけない観測、例えば core tool-call の返り値、コード実行結果、judge 結果、データベース照会結果は勝手に摂動できない。そうすると候補経路が因果構造を壊す。

候補経路を得た後、ソースタームを含む作用によって現在の経路と候補経路を比較する。Metropolis-Hastings acceptance rate は
```math
\begin{aligned}
A_k(\tau\rightarrow\tau') = \min\left( 1, \exp\left[-(S_\eta[\tau']-S_\eta[\tau])\right] \frac{q_{\mathrm{prop}}(\tau\mid\tau')} {q_{\mathrm{prop}}(\tau'\mid\tau)} \right)
\end{aligned}
```
である。

候補経路の有効作用が低いほど受理されやすい。候補経路が一時的に悪くても、受理率が 0 でなければ残る可能性がある。この仕組みにより、経路サンプリングは現在の経路近傍に止まらず、局所的な低点から抜け出せる。

### 4.4. Langevin 経路サンプリング

Langevin は作用の勾配を必要とするため、連続自由度に適している。LLM では離散 token に直接 Langevin 更新をかけられないが、隠れ状態を連続行動変数として使える。例えば
```math
\begin{aligned}
\tau_z &= [o_0,z_0,r_0,o_1,\cdots,o_T,z_T,r_T,o_{T+1}] \\
z_t &= [z_{L_t},\cdots,z_{L_{t+1}-1}]
\end{aligned}
```
と書ける。

ここでも、隠れ状態経路全体を任意に変えるべきではない。より妥当な方法は、隠れ状態接頭列を選び、その接頭列に Langevin 更新を適用し、更新後の接頭列からデコードまたは rollout を続けて後続経路を改めて得る方法である。$c_z^{(\ell)} = [h_t,z_{L_t}^{(\ell)},\cdots,z_i^{(\ell)}]$ とする。Langevin 更新は
```math
z_{L_t:i}^{(\ell+1)} = z_{L_t:i}^{(\ell)} - \epsilon\nabla_{z_{L_t:i}^{(\ell)}}S_\eta[ \mathrm{Rollout}(c_z^{(\ell)},o_{\mathrm{allow}}^{(\ell)},r_{\mathrm{soft}}^{(\ell)}) ] + \sqrt{2\epsilon}\xi_\ell, \quad \xi_\ell \sim \mathcal N(0,I)
```
である。
$\ell$ は Langevin 更新ステップを表す。摂動を許す観測とソフト収益も摂動するなら
```math
o_{\mathrm{allow}}^{(\ell+1)} = \mathcal E_o(o_{\mathrm{allow}}^{(\ell)};m_o,\xi_{o,k}),\quad r_{\mathrm{soft}}^{(\ell+1)} = \mathcal E_r(r_{\mathrm{soft}}^{(\ell)};m_r,\xi_{r,k})
```
と書く。

更新後の候補接頭列とフィードバックは
```math
c_z^{(\ell+1)} = [h_t,z_{L_t}^{(\ell+1)},\cdots,z_i^{(\ell+1)}]
```
である。

その後、rollout を続ける。
```math
\tau_z^{(\ell+1)} = \mathrm{Rollout} (c_z^{(\ell+1)},o_{\mathrm{allow}}^{(\ell+1)},r_{\mathrm{soft}}^{(\ell+1)})
```

Langevin では、勾配項が経路を低作用の方向へ引き、ランダム項が熱揺らぎを残す。目標は、経路をタスクから離れさせることではない。行動接頭列と許可されたフィードバックを摂動しながら、後続 rollout がタスクを完了できる低罰則・低作用の領域へ戻れるようにする。安定した良い経路は孤立点ではなく経路盆地である。小さな摂動の後でも、後続生成はタスクを完了できる軌道に戻れる。


### 4.5. Related Work

- Path-integral RL
  - 研究方向: Path Integral Control / ${PI}^2$
  - 直接関連する文献:
    - [[PMLR 2010] Learning Policy Improvements with Path Integrals](https://proceedings.mlr.press/v9/theodorou10a.html)
    - [[JMLR 2010] A Generalized Path Integral Control Approach to Reinforcement Learning](https://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf)
  - 近い方向の文献:
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)

- KL penalty / control cost
  - 研究方向: KL control / linearly-solvable MDP
  - 直接関連する文献:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)
  - 近い方向の文献:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

- RL as inference
  - 研究方向: maximum entropy RL / control as inference / SAC
  - 直接関連する文献:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  - 近い方向の文献:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)

- $e^{-\beta H}$ sampling
  - 研究方向: EBM / Gibbs / MCMC / Langevin
  - 直接関連する文献:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - 近い方向の文献:
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

- Langevin on hidden latent variables
  - 研究方向: latent EBM / energy-based text generation / continuous relaxation
  - 直接関連する文献:
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
    - [[ICML 2021] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://proceedings.mlr.press/v139/pang21a.html)
  - 近い方向の文献:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2511.07124] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought](https://arxiv.org/abs/2511.07124)

- LLM reasoning sampling
  - 研究方向: MCMC-inspired reasoning / constrained sampling
  - 直接関連する文献:
    - [[arXiv:2506.05754] Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - 近い方向の文献:
    - [[arXiv:2510.14901] Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)

---

## 5. Relationship Between Sections 3 and 4

---

# Appendix A: 時間方向の粗視化と「繰り込み」

これは一次元の時間経路積分なので、ここでいう「繰り込み」は主に時間方向の粗視化として現れる。非自己回帰版では、$\ell$ をマクロ時間ブロックの添字とし、$b$ 個の微視的行動を一つのマクロブロックにまとめる。

```math
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1})
```

多層圧縮を行うと、時間長は次のように短くなる。

```math
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}
```

となる。

経路積分のレベルでは、微視的経路から巨視的経路への写像は $\bar\tau=\mathcal C(\tau)$ である。元の経路分布を議論するなら、巨視的な基礎作用は微視的自由度を積分消去して得る。

```math
\exp(-S_0[\bar\tau]) = \int_{\mathcal C(\tau)=\bar\tau} \exp(-S_0[\tau]) \prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t
```


LLM の自己回帰版には二つの時間構造がある。外側は環境 / agent step $t$ であり、内側は token 位置 $i$ である。外部ステップ $t$ の token segment は
```math
a_t=[a_{L_t},\ldots,a_{L_{t+1}-1}],\quad da_t=\prod_{i=L_t}^{L_{t+1}-1}da_i
```
である。

したがって、LLM では最も基本的な時間方向の粗視化として、token レベルの微視的ステップをセグメント単位の行動 $a_t$ にまとめている。
```math
\prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i = \prod_{t=0}^{T}da_t
```

一つの外部ステップ内で token ブロックをさらに圧縮したいなら、外部ステップ $t$ 内部の $\ell$ 番目の token ブロックを $B_{t,\ell}=C_\phi(a_{L_t+\ell b},a_{L_t+\ell b+1},\ldots,a_{L_t+(\ell+1)b-1})$ とする。このとき、LLM の自己回帰的な基礎作用の粗視化は
```math
\exp(-S_0^{\mathrm{AR}}[\bar\tau]) = \int_{\mathcal C(\tau)=\bar\tau} \exp(-S_0^{\mathrm{AR}}[\tau]) \prod_{t=0}^{T}\prod_{i=L_t}^{L_{t+1}-1}da_i\,do_{t+1}\,dr_t
```
である。

ここで
```math
S_0^{\mathrm{AR}}[\tau] =- \sum_{t=0}^{T}\sum_{i=L_t}^{L_{t+1}-1}\log\pi(a_i\mid h_{i,t}) -\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid h_t,a_t)
```
である。

圧縮ブロック内部の有効特異値スペクトルが速く減衰するなら、切り捨ては安全である。スペクトルがほぼ平坦なら、強制的な切り捨ては大量の情報を失う。
```math
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ の切り捨ては大きな情報損失を起こす}
```

**Note**: ここで「繰り込み」と呼んでいるものは、厳密には token レベルの圧縮と粗視化にすぎない。真の「意味情報の繰り込み」を記述しているわけではない。物理でいう繰り込みでは、情報を粗視化した後、あるスケールで不変量や RG 固定点が現れる。筆者は、現在の token 圧縮技術は学習可能な重みによって情報を選び、圧縮していると考える。この操作は真の繰り込みではない。そのような圧縮は、不変量や RG 固定点の出現を保証できず、それを示す根拠もまだない。
