# history-based RL から再サンプリングおよび RL の物理的描像へ

## 0. 目的

ここで議論するのは一種の**物理的描像による解釈**であり、RL を厳密に場の理論として書き直すことではない。

本稿の意図は、サンプリング経路を拡張するための物理的手法を議論することであり、主線は二つある：

- **再サンプリング** : RL の元の累積リターン定義のもとで、観測量 $o_{t+1}$ と報酬 $r_t$ を再サンプリング / 再推定し、rollout、リターン推定、advantage 推定、および PPO / GRPO / GSPO 更新を改善するために用いる。
- **経路積分** : 経路積分の描像のもとで、有効ハミルトニアン、Boltzmann 重み、Gibbs サンプリング、MCMC / Langevin、および逆温度アニーリングを導入し、経路空間におけるサンプリングを直接拡張し、低ハミルトニアン経路を探索するために用いる。

共通の基盤は : 
- history-based RL は一次元の時間方向上の経路積分として書ける
- 元の期待値におけるリターン $G[\tau]$ は経路上の observable insertion である
- 熱力学的類比では、作用 $S$ は無次元量であり、ハミルトニアン $H$ はエネルギーの次元を持ち、$S=\beta H$ を満たす

---

## 1. 元の history-based RL の軌跡積分

最も原始的な形式から始める。相互作用履歴を : 

\[
\begin{equation}
h_t=(o_0,a_0,r_0,o_1,a_1,r_1,\ldots,a_{t-1},r_{t-1},o_t).
\tag{1}
\end{equation}
\]

とする。

方策を $\pi(a_t\mid h_t)$、環境の条件密度を $\mu(o_{t+1},r_t\mid a_t,h_t)$ とする。有限時間 $T$ 内で、軌跡全体の期待リターンは次のように書ける : 

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right).
\tag{2}
\end{equation}
\]

経路全体を : 

\[
\begin{equation}
\tau=(a_0,o_1,r_0,a_1,o_2,r_1,\ldots,a_T,o_{T+1},r_T).
\tag{3}
\end{equation}
\]

と書く。

経路リターンを : 

\[
\begin{equation}
G[\tau]=\sum_{s=0}^{T}\gamma^s r_s.
\tag{4}
\end{equation}
\]

と書く。

したがって、元の RL 目的は、すべての可能な経路に対して加重積分を行うことであり、各経路の重みは方策と環境によって共同で与えられ、各経路の値は割引リターン $G[\tau]$ によって与えられる。

### 1.1. 決定論的環境における Dirac delta への退化

環境が決定論的である場合、$a_t,h_t$ が与えられると、次の観測と報酬は決定関数によって与えられる : 

\[
\begin{equation}
o_{t+1}=O(a_t,h_t),\quad r_t=R(o_{t+1},a_t,h_t).
\tag{5}
\end{equation}
\]

環境の条件密度は Dirac delta に退化する : 

\[
\begin{equation}
\mu(o_{t+1},r_t\mid a_t,h_t)=\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t)).
\tag{6}
\end{equation}
\]

これを元の軌跡積分へ代入すると : 

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\delta(r_t-R(o_{t+1},a_t,h_t))\delta(o_{t+1}-O(a_t,h_t))\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T}\gamma^s r_s\right).
\tag{7}
\end{equation}
\]

Dirac delta の基本的な積分性質を用いる : 

\[
\begin{equation}
\int \delta(x-x_0)f(x)\,dx=f(x_0).
\tag{8}
\end{equation}
\]

環境部分が潰れると、動作サンプリングだけが残る : 

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T}\pi(a_t\mid h_t)\,da_t\right]\left[\sum_{s=0}^{T}\gamma^s R(O(a_s,h_s),a_s,h_s)\right].
\tag{9}
\end{equation}
\]

これは次を意味する : 
- 決定論的環境はもはや経路分岐を提供せず、経路分岐は方策サンプリングのみに由来する
- 方策も決定論的であるなら、経路全体は一本の経路へ潰れる

### 1.2. 有限ホライズン (finite horizon)、報酬クリッピング (reward clipping)

最大経路長 $T\le T_{\max}$ を制限する場合、経路積分は有限時間区間上でのみ行われる : 

\[
\begin{equation}
J(\pi)=\int\left[\prod_{t=0}^{T_{\max}}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t)\,da_t\,do_{t+1}\,dr_t\right]\left(\sum_{s=0}^{T_{\max}}\gamma^s r_s\right).
\tag{10}
\end{equation}
\]

報酬をクリップする場合 : 

\[
\begin{equation}
\bar r_t=\operatorname{clip}(r_t,-r_{\max},r_{\max}),\quad \bar G[\tau]=\sum_{t=0}^{T}\gamma^t\bar r_t.
\tag{11}
\end{equation}
\]

意味 : 
- 有限ホライズンは時間切断であり、報酬クリッピングはリターン観測量の有界化である

### 1.3. LLM RL における経路サンプリングと hidden 動作

LLM にとって、動作 $a_t$ は token、完全な回答断片、tool call、コード patch、または agent step であり得る。動作が token である場合、モデルの forward は logits を与え、その後サンプリングを通じて token を得る : 

\[
\begin{equation}
z_t^{\mathrm{logit}}=f_\theta(h_t),\quad \pi_{\theta,T_{\mathrm{dec}}}(a_t\mid h_t)=\frac{\exp(z^{\mathrm{logit}}_{t,a_t}/T_{\mathrm{dec}})}{\sum_{a'}\exp(z^{\mathrm{logit}}_{t,a'}/T_{\mathrm{dec}})}.
\tag{12}
\end{equation}
\]

連続空間で経路サンプリングを行うなら、prefill 後に得られる hidden を連続動作表現として用いることができる : 

\[
\begin{equation}
a_t=z_t,\quad z_t=\operatorname{hidden}_\theta(\operatorname{prefill}(a_{\le t})).
\tag{13}
\end{equation}
\]

したがって、連続動作経路は依然として同じ経路記号で書ける : 

\[
\begin{equation}
\tau=(z_0,o_1,r_0,z_1,o_2,r_1,\ldots,z_T,o_{T+1},r_T).
\tag{14}
\end{equation}
\]

ここでは新しい経路変数を導入しているのではなく、離散 token action $a_t$ を連続 hidden action $z_t$ に置き換えているだけである。hidden を強調しない場合は依然として $a_t$ と書く。

第 $i$ 本目のサンプルを : 

\[
\begin{equation}
\tau_i=(a_{i,0},o_{i,1},r_{i,0},a_{i,1},o_{i,2},r_{i,1},\ldots,a_{i,T_i},o_{i,T_i+1},r_{i,T_i}).
\tag{15}
\end{equation}
\]

とする。

hidden を動作として用いる場合 : 

\[
\begin{equation}
a_{i,t}=z_{i,t},\quad z_{i,t}=\operatorname{hidden}_\theta(\operatorname{prefill}(a_{i,\le t})).
\tag{16}
\end{equation}
\]

ここで $i$ はサンプル番号、$t$ は経路上の時間ステップまたは token 位置である。

---

## 2. 元の history-based RL の物理的描像による解釈

### 2.1. 軌跡積分から経路積分へ

元の密度の連乗を経路密度として書く : 

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{17}
\end{equation}
\]

すると、元の目的は経路積分形式で書ける : 

\[
\begin{equation}
J(\pi)=\int P_{\pi,\mu}[\tau]G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{18}
\end{equation}
\]

物理的描像 : 
- $\tau$ は一本の worldline であり、$P_{\pi,\mu}[\tau]$ は経路重み、$G[\tau]$ は経路利得汎関数、$J(\pi)$ はすべての経路の加重平均である

### 2.2. 基礎作用、基礎ハミルトニアンと Boltzmann 重み

経路密度から出発する : 

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{19}
\end{equation}
\]

経路密度の負の対数を取り、基礎作用を得る : 

\[
\begin{equation}
S_{\pi,\mu}[\tau]=-\log P_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{20}
\end{equation}
\]

熱力学的類比では、Boltzmann 重みは $\exp(-\beta H)$ と書かれる。したがって、基礎作用と基礎ハミルトニアンの関係は : 

\[
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau].
\tag{21}
\end{equation}
\]

したがって、元の経路密度は次のように書ける : 

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\exp(-S_{\pi,\mu}[\tau])=\exp(-\beta H_0[\tau]).
\tag{22}
\end{equation}
\]

元の RL 目的は次のようになる : 

\[
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{23}
\end{equation}
\]

ここで $G[\tau]$ は observable insertion であり、ハミルトニアンの一部ではない。

### 2.3. 一次元時間方向の単体複雑系

経路変数は次のように書ける : 

\[
\begin{equation}
x_t=(a_t,o_{t+1},r_t),\quad \tau=(x_0,x_1,\ldots,x_T).
\tag{24}
\end{equation}
\]

これは一次元時間方向上の経路系である。複雑性は history coupling から来る。なぜなら、各ステップの方策と環境はいずれも完全な履歴に依存するからである : 

\[
\begin{equation}
\pi(a_t\mid h_t),\quad \mu(o_{t+1},r_t\mid a_t,h_t).
\tag{25}
\end{equation}
\]

対応する基礎作用項は : 

\[
\begin{equation}
S_{\pi,\mu}[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{26}
\end{equation}
\]

物理的描像 : 
- RL path integral は一次元時間方向上の複雑な単体系であり、内部自由度は action、observation、reward であり、複雑性はこれらの自由度が履歴 $h_t$ を通じて長距離の時間結合を生じることに由来する

### 2.4. 割引因子と Laplace 正則化

連続時間リターンを : 

\[
\begin{equation}
G[\tau]=\int_0^\infty r(t)\,dt,
\tag{27}
\end{equation}
\]

と書くと、発散し得る。指数減衰を加えると : 

\[
\begin{equation}
G_\lambda[\tau]=\int_0^\infty e^{-\lambda t}r(t)\,dt.
\tag{28}
\end{equation}
\]

離散時間では : 

\[
\begin{equation}
G_\gamma[\tau]=\sum_{t=0}^{\infty}\gamma^t r_t,
\tag{29}
\end{equation}
\]

時間ステップ幅を $\Delta t$ とすると、対応関係は : 

\[
\begin{equation}
\gamma=e^{-\lambda\Delta t},\quad \gamma^t=e^{-\lambda t\Delta t}.
\tag{30}
\end{equation}
\]

報酬が有界である場合 : 

\[
\begin{equation}
|r_t|\le r_{\max},
\tag{31}
\end{equation}
\]

割引リターンは有界である : 

\[
\begin{equation}
|G_\gamma[\tau]|\le \sum_{t=0}^{\infty}\gamma^t|r_t|\le r_{\max}\sum_{t=0}^{\infty}\gamma^t=\frac{r_{\max}}{1-\gamma}.
\tag{32}
\end{equation}
\]

物理的描像 : 
- discount factor は時間方向上の Laplace damping であり、無限の未来を有限の有効寄与へ押し込む

---

## 3. RL の元の定義のもとでの観測量と報酬の再サンプリング

元の RL の累積リターン定義から出発し、方策更新器を変えない前提で、観測量 $o_{t+1}$ と報酬 $r_t$ の局所サンプリング空間を拡張し、モデルが経路をよりよく探索できるようにすることを目的とする。より複雑な方法として、実環境の反復サンプリング、world model、reward model、verifier、SMC、または CEM proposal などを用いることもできるが、本節ではこれらの複雑な反復サンプリング法は議論せず、最も基本的なガウスノイズ proposal のみを用いる。

### 3.1. $o$ と $r$ のガウスノイズ再サンプリング / 再推定

$m=1,\ldots,M$ は同じ $a_t,h_t,o_{t+1},r_t$ 条件下での第 $m$ 回目の再サンプリングを表すものとする。シミュレーテッド・アニーリングを用いるなら、$k=0,1,\ldots,K_{\mathrm{ann}}$ は外部アニーリング反復ステップを表し、経路内部の時間ステップ $t$ ではない。第 $k$ ラウンドでは逆温度 $\beta_k>0$ を用いる。$\sigma_o>0$ は観測量の基礎ノイズ尺度、$\sigma_r>0$ は報酬の基礎ノイズ尺度とする。$\sigma_o,\sigma_r$ は人為的に指定される proposal ノイズ強度であり、グループ内統計から得られる分散ではない。

まず一次元スカラー変数から始める。現在値 $x_0$ と第 $k$ ラウンドのノイズ幅 $s_k>0$ が与えられたとき、最も基本的な一次元ガウス proposal を次のように定義する : 

\[
\begin{equation}
q_k(x'\mid x_0)=\frac{1}{\sqrt{2\pi}s_k}\exp\left(-\frac{(x'-x_0)^2}{2s_k^2}\right).
\tag{33}
\end{equation}
\]

逆温度によって proposal の幅を制御するために、次を定義する : 

\[
\begin{equation}
s_k=\frac{\sigma}{\sqrt{\beta_k}},
\tag{34}
\end{equation}
\]

ここで $\sigma>0$ は基礎ノイズ尺度である。式 (34) を式 (33) に代入すると、次を得る : 

\[
\begin{equation}
q_k(x'\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x'-x_0)^2}{2\sigma^2}\right).
\tag{35}
\end{equation}
\]

標準ガウスノイズ $\xi^{(m)}$ の密度は : 

\[
\begin{equation}
p(\xi^{(m)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi^{(m)})^2}{2}\right).
\tag{36}
\end{equation}
\]

第 $m$ 回目の再サンプリング値を : 

\[
\begin{equation}
x^{(m,k)}=x_0+\frac{\sigma}{\sqrt{\beta_k}}\xi^{(m)}.
\tag{37}
\end{equation}
\]

とおく。

式 (37) より : 

\[
\begin{equation}
\xi^{(m)}=\frac{\sqrt{\beta_k}}{\sigma}(x^{(m,k)}-x_0).
\tag{38}
\end{equation}
\]

したがって、$x^{(m,k)}$ の密度はまさに $x_0$ を中心とし、幅が $\sigma/\sqrt{\beta_k}$ のガウス proposal である : 

\[
\begin{equation}
q_k(x^{(m,k)}\mid x_0)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\beta_k(x^{(m,k)}-x_0)^2}{2\sigma^2}\right).
\tag{39}
\end{equation}
\]

したがって、小さい $\beta_k$ はより広い proposal に対応し、大きい $\beta_k$ はより狭い proposal に対応する。これが本節におけるシミュレーテッド・アニーリングの制御方式である。

$o_{t+1}$ が一次元連続観測量である場合、$x_0=o_{t+1}$、$\sigma=\sigma_o$ と取ると、観測量の再サンプリング式を得る : 

\[
\begin{equation}
o_{t+1}^{(m,k)}=o_{t+1}+\frac{\sigma_o}{\sqrt{\beta_k}}\xi_o^{(m,t)}.
\tag{40}
\end{equation}
\]

ここで $\xi_o^{(m,t)}$ は第 $m$ 回目の再サンプリング、経路時間ステップ $t$ 上の標準ガウスノイズである : 

\[
\begin{equation}
p(\xi_o^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_o^{(m,t)})^2}{2}\right).
\tag{41}
\end{equation}
\]

対応する観測量 proposal は : 

\[
\begin{equation}
q_k(o'_{t+1}\mid o_{t+1})=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_o}\exp\left(-\frac{\beta_k(o'_{t+1}-o_{t+1})^2}{2\sigma_o^2}\right).
\tag{42}
\end{equation}
\]

$o_{t+1}$ が多次元ベクトルまたはテキスト embedding である場合、各座標に同じ一次元ガウス摂動を用いることができる。ここでは相関ノイズ行列を導入しない。将来、相関ノイズが必要であれば、行列の意味を先に定義してから書かなければならない。

スカラー報酬に直接ガウス摂動を加える場合、$x_0=r_t$、$\sigma=\sigma_r$ と取ると、次を得る : 

\[
\begin{equation}
r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\xi_r^{(m,t)}.
\tag{43}
\end{equation}
\]

ここで : 

\[
\begin{equation}
p(\xi_r^{(m,t)})=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(\xi_r^{(m,t)})^2}{2}\right).
\tag{44}
\end{equation}
\]

対応する報酬 proposal は : 

\[
\begin{equation}
q_k(r'_t\mid r_t)=\frac{\sqrt{\beta_k}}{\sqrt{2\pi}\sigma_r}\exp\left(-\frac{\beta_k(r'_t-r_t)^2}{2\sigma_r^2}\right).
\tag{45}
\end{equation}
\]

しかし、単に $r_t$ にゼロ中心のガウスノイズを加え、その後サンプル平均を取るだけなら、平均値は元の $r_t$ の近くへ戻る。サンプル平均の元の定義に従うと : 

\[
\begin{equation}
\bar r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}r_t^{(m,k)}=r_t+\frac{\sigma_r}{\sqrt{\beta_k}}\left(\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\right).
\tag{46}
\end{equation}
\]

ガウスノイズの正負がほぼ打ち消し合うとき : 

\[
\begin{equation}
\frac{1}{M}\sum_{m=1}^{M}\xi_r^{(m,t)}\approx 0,
\tag{47}
\end{equation}
\]

したがって : 

\[
\begin{equation}
\bar r_t^{(k)}\approx r_t.
\tag{48}
\end{equation}
\]

ゆえに、報酬へ直接ノイズを加えることは主にロバスト性の摂動である。より意味のある方法は、まず観測量または観測 hidden に対してガウス再サンプリングを行い、その後、摂動後の観測を用いて報酬を再計算することである : 

\[
\begin{equation}
r_t^{(m,k)}=R(o_{t+1}^{(m,k)},a_t,h_t).
\tag{49}
\end{equation}
\]

対応する局所報酬推定は : 

\[
\begin{equation}
\widehat r_t^{(k)}=\frac{1}{M}\sum_{m=1}^{M}R(o_{t+1}^{(m,k)},a_t,h_t).
\tag{50}
\end{equation}
\]

第 $i$ 本目の動作経路について、第 $m$ 回目の再サンプリング経路は : 

\[
\begin{equation}
\tau_i^{(m,k)}=(a_{i,0},o_{i,1}^{(m,k)},r_{i,0}^{(m,k)},\ldots,a_{i,T_i},o_{i,T_i+1}^{(m,k)},r_{i,T_i}^{(m,k)}).
\tag{51}
\end{equation}
\]

第 $m$ 回目の再サンプリング経路のリターンは : 

\[
\begin{equation}
G_i^{(m,k)}=\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{52}
\end{equation}
\]

サンプル平均の元の定義を用いて、第 $k$ ラウンドの再サンプリングリターン推定を得る : 

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}G_i^{(m,k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{53}
\end{equation}
\]

これが本節の核心である : 元の RL 累積リターン定義のもとで、ガウスノイズ proposal を用いて $o_{t+1}$、$r_t$、またはそれらの連続表現を拡張し、その後サンプル平均によってより安定したリターンまたは advantage を推定する。

### 3.2. PPO / GRPO / GSPO への接続

GRPO / GSPO は本節の細分化方向として扱うことができる。それらは自然にグループ内サンプル構造を持つため、再サンプリング後のリターンに対してグループ内統計を行うのに適している。しかし、この方法は同様に PPO にも利用できる。なぜなら、PPO も rollout、reward、advantage、および方策更新比率だけを必要とするからである。

$K_s$ を第 $k$ ラウンドのグループ内サンプル数、$i,j$ をグループ内サンプル番号、$t$ を経路内部の時間ステップまたは token 位置とする。同じ入力 $x$ に対して、第 $k$ ラウンドで $K_s$ 本の経路をサンプリングする : 

\[
\begin{equation}
\tau_1^{(k)},\tau_2^{(k)},\ldots,\tau_{K_s}^{(k)}\sim q_k(\tau\mid x).
\tag{54}
\end{equation}
\]

各経路はガウスノイズ再サンプリングを用いてリターン推定を得る : 

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{55}
\end{equation}
\]

グループ内平均と分散は : 

\[
\begin{equation}
\bar G^{(k)}=\frac{1}{K_s}\sum_{i=1}^{K_s}\widehat G_i^{(k)},\quad (\sigma_G^{(k)})^2=\frac{1}{K_s}\sum_{i=1}^{K_s}(\widehat G_i^{(k)}-\bar G^{(k)})^2.
\tag{56}
\end{equation}
\]

標準化 advantage は : 

\[
\begin{equation}
A_i^{(k)}=\frac{\widehat G_i^{(k)}-\bar G^{(k)}}{\sigma_G^{(k)}+\epsilon}.
\tag{57}
\end{equation}
\]

観測量と報酬そのものに対して局所統計を行う場合、次のように書ける : 

\[
\begin{equation}
\bar o_{t+1}^{(k)}=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}o_{i,t+1}^{(m,k)},\quad \bar r_t^{(k)}=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}r_{i,t}^{(m,k)}.
\tag{58}
\end{equation}
\]

対応する分散は : 

\[
\begin{equation}
(s_o^{(k)})^2=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}(o_{i,t+1}^{(m,k)}-\bar o_{t+1}^{(k)})^2,
\quad
(s_r^{(k)})^2=\frac{1}{K_sM}\sum_{i=1}^{K_s}\sum_{m=1}^{M}(r_{i,t}^{(m,k)}-\bar r_t^{(k)})^2.
\tag{59}
\end{equation}
\]

ここで $s_o^{(k)}$ と $s_r^{(k)}$ はグループ内統計によって得られるフィードバック幅であり、proposal における基礎ノイズ尺度 $\sigma_o$ と $\sigma_r$ ではない。分散が小さくなることは、フィードバック推定が安定へ向かっている、または現在の方策があるより安定した局所領域に入っている、と解釈できる。

GSPO は、生成系列全体をサンプリング単位とするものと見なせる : 

\[
\begin{equation}
\tau_i=(a_{i,0},o_{i,1},r_{i,0},\ldots,a_{i,T_i},o_{i,T_i+1},r_{i,T_i}).
\tag{60}
\end{equation}
\]

モデル動作、ツール返却、環境観測、および報酬テキストが同じ sequence 内に混在している場合、sequence-level サンプリングは近似的に $a$ と $o$ を混ぜてサンプリングすることになる。より合理的な因果分解は : 

\[
\begin{equation}
q(\tau)=q_a(a_{0:T})q_o(o_{1:T+1}\mid a_{0:T},h_{0:T})q_r(r_{0:T}\mid a_{0:T},o_{1:T+1},h_{0:T}).
\tag{61}
\end{equation}
\]

hidden を動作として用いる場合 : 

\[
\begin{equation}
q(\tau)=q_z(z_{0:T})q_o(o_{1:T+1}\mid z_{0:T},h_{0:T})q_r(r_{0:T}\mid z_{0:T},o_{1:T+1},h_{0:T}).
\tag{62}
\end{equation}
\]

意味 : 
- 観測量と報酬フィードバック空間を拡張する
- GRPO / GSPO のグループ内統計は、$G$、advantage、$o$ と $r$ の安定性をより自然に推定できる
- PPO も同じ $o,r$ 再サンプリングデータを使用でき、後続の方策更新器が異なるだけである

### 3.3. シミュレーテッド・アニーリング

シミュレーテッド・アニーリングは本節においてガウスノイズ proposal のスケジューリングに用いられる。第 $k$ ラウンドの逆温度 $\beta_k$ は、式 (40) と式 (43) を通じてガウスノイズ幅を制御する : 

\[
\begin{equation}
\frac{\sigma_o}{\sqrt{\beta_k}},\quad \frac{\sigma_r}{\sqrt{\beta_k}}.
\tag{63}
\end{equation}
\]

ここで逆温度は増加する : 

\[
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}.
\tag{64}
\end{equation}
\]

小さい $\beta_k$ は初期のより広いガウスノイズ proposal に対応し、大きい $\beta_k$ は後期のより狭く、より保守的なフィードバック推定に対応する。ここでの $\beta_k$ はシミュレーテッド・アニーリングのスケジューリングパラメータであり、サンプリング幅とフィードバック揺らぎを説明するために用いられるもので、decoder temperature ではない。

### 3.4. 統計力学的解釈

第二節の物理的描像から見ると、元の history-based RL はすでに一次元時間経路系と見なすことができる。方策と環境は元の経路分布を誘導する : 

\[
\begin{equation}
P_{\pi,\mu}[\tau]=\prod_{t=0}^{T}\pi(a_t\mid h_t)\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{65}
\end{equation}
\]

本節では、リターンは経路上の統計的観測量として扱われる : 

\[
\begin{equation}
G[\tau]=\sum_{t=0}^{T}\gamma^t r_t.
\tag{66}
\end{equation}
\]

同一入力からサンプリングして得られる一組の経路は、元の経路分布またはその rollout augmentation 下の局所 ensemble と見なせる : 

\[
\begin{equation}
\tau_i\sim P_{\pi,\mu}[\tau],\quad i=1,\ldots,K_s.
\tag{67}
\end{equation}
\]

グループ内リターン平均と分散は : 

\[
\begin{equation}
\bar G=\frac{1}{K_s}\sum_{i=1}^{K_s}G[\tau_i],\quad \sigma_G^2=\frac{1}{K_s}\sum_{i=1}^{K_s}(G[\tau_i]-\bar G)^2.
\tag{68}
\end{equation}
\]

ここで $\bar G$ はこの局所 ensemble の平均リターンを記述し、$\sigma_G^2$ はリターンの揺らぎを記述する。$o_{t+1}$ と $r_t$ に対するガウスノイズ再サンプリングは、この局所 ensemble を拡張し、advantage またはリターン推定が単一回の rollout の偶然の結果だけに依存しないようにするものと見なせる。

シミュレーテッド・アニーリングは本節ではまずサンプリング / 推定のスケジューリング機構である。初期にはより小さい $\beta_k$ を用い、ガウス proposal をより広くして、より大きなフィードバック揺らぎを許す。後期には徐々に $\beta_k$ を増大させ、proposal をより安定したフィードバック推定へ収縮させる。統計力学の言葉で見ると、これは高温探索から低温安定化への冷却過程に類似している。ここでのリターンは依然として統計的観測量であり、advantage の推定、サンプルの順位付け、PPO / GRPO / GSPO の更新に用いられる。

### 3.5. 関連研究

- $(o,r)$ 再サンプリング
  - 対応する文献方向: Gaussian noise proposal / noisy environment augmentation / observation noise / reward noise
  - 直接関連:
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
    - [[arXiv:2106.11420] Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/abs/2106.11420)
    - [[arXiv:1810.01032] Reinforcement Learning with Perturbed Rewards](https://arxiv.org/abs/1810.01032)
    - [[PMLR 2020] Deep Reinforcement Learning with Robust and Smooth Policy](https://proceedings.mlr.press/v119/shen20b.html)
  - 同方向:
    - [[arXiv:2310.00344] HarmonyDream: Task Harmonization Inside World Models](https://arxiv.org/abs/2310.00344)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)

- グループ内統計
  - 対応する文献方向: GRPO / GSPO
  - 直接関連:
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
  - 同方向:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

- proposal 拡張
  - 対応する文献方向: SMC policy optimization / CEM / iCEM
  - 直接関連:
    - [[arXiv:2402.07963] SPO: Sequential Monte Carlo Policy Optimisation](https://arxiv.org/abs/2402.07963)
    - [[arXiv:2505.16732] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](https://arxiv.org/abs/2505.16732)
    - [[arXiv:2008.06389] Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/abs/2008.06389)
    - [[arXiv:2112.07746] CEM-GD: Cross-Entropy Method with Gradient Descent Planner for Model-Based Reinforcement Learning](https://arxiv.org/abs/2112.07746)
  - 同方向:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)

- 元の RL 推定器を用いた PPO / GRPO / GSPO への接続
  - 対応する文献方向: PPO-family + noisy rollout augmentation / group-level RL
  - 直接関連:
    - [[arXiv:1707.06347] Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - [[arXiv:2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
    - [[arXiv:2507.18071] Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
    - [[arXiv:2305.02882] Simple Noisy Environment Augmentation for Reinforcement Learning](https://arxiv.org/abs/2305.02882)
  - 同方向:
    - [[arXiv:1906.08253] When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
    - [[arXiv:2301.04104] Mastering Diverse Domains through World Models / DreamerV3](https://arxiv.org/abs/2301.04104)


---

## 4. 経路積分 / 有効ハミルトニアン / 経路サンプリング

### 4.1. 外場ハミルトニアン：RL の報酬とペナルティ項の物理的表現

本節は第二節の基礎ハミルトニアン表示から出発する。基礎ハミルトニアン $H_0[\tau]$ は、方策と環境の元の経路重みに由来する。外場ハミルトニアンは、RL における報酬、KL ペナルティ、経路長コストなどの目的項を表すために用いられる。ここでの外場ハミルトニアンは目的構造であり、サンプリング方法ではない。

リターン外場、KL 一般化化学ポテンシャル、および経路長化学ポテンシャルを導入すると、外場ハミルトニアンは次のように書ける : 

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_G G[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_N N[\tau].
\tag{69}
\end{equation}
\]

ここで $\lambda_G$ は高リターン経路への選好を制御する外場強度、$\lambda_{\mathrm{KL}}$ は一般化化学ポテンシャルまたは KL 引き戻し強度、$\lambda_N$ は経路長または相互作用回数の化学ポテンシャルコストである。前文で clipped return を使用した場合、ここでは $G[\tau]$ の代わりに $\bar G[\tau]$ を使用してもよい。

総ハミルトニアンは : 

\[
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]+H_{\mathrm{field}}[\tau].
\tag{70}
\end{equation}
\]

対応する有効作用は : 

\[
\begin{equation}
S_{\mathrm{eff}}[\tau]=\beta H_{\mathrm{eff}}[\tau]=\beta H_0[\tau]+\beta H_{\mathrm{field}}[\tau].
\tag{71}
\end{equation}
\]

展開すると : 

\[
\begin{equation}
S_{\mathrm{eff}}[\tau]=S_{\pi,\mu}[\tau]-\beta \lambda_GG[\tau]+\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\beta\lambda_NN[\tau].
\tag{72}
\end{equation}
\]

経路 KL 項は : 

\[
\begin{equation}
D_{\mathrm{KL}}[\tau]=\sum_{t=0}^{T}D_{\mathrm{KL}}\left(\pi(\cdot\mid h_t)\Vert \pi_{\mathrm{ref}}(\cdot\mid h_t)\right).
\tag{73}
\end{equation}
\]

狭義の粒子数または資源数は $N[\tau]$ と書ける。例えば token 数、step 数、tool-call 数、または interaction 数である。$\lambda_NN[\tau]$ は資源化学ポテンシャルコストに対応する。

もし直接 : 

\[
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]-\lambda_GG[\tau]+\lambda_NN[\tau],
\tag{74}
\end{equation}
\]

を使用すると、$\lambda_N$ が過大である場合、低ハミルトニアン経路は過度に短い経路へ偏り、タスクを完了できなくなる可能性がある。より安全な方式は、経路長を制約区間として扱うことである : 

\[
\begin{equation}
C_N[\tau]=\mu_+\max(0,N[\tau]-N_{\max})^2+\mu_-\max(0,N_{\min}-N[\tau])^2.
\tag{75}
\end{equation}
\]

対応する外場ハミルトニアンは : 

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+C_N[\tau].
\tag{76}
\end{equation}
\]

物理的描像 :
- $N[\tau]$ は小さいほどよいものではなく、資源制約である。本当の低ハミルトニアン経路は、成功していて短い経路であり、短いが失敗する経路ではない

### 4.2. Boltzmann 重み：ハミルトニアンから経路重みを得る

等価ハミルトニアンが与えられると、Boltzmann 重みは : 

\[
\begin{equation}
W_\beta[\tau]=\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{77}
\end{equation}
\]

元の経路密度の形式へ展開すると : 

\[
\begin{equation}
W_\beta[\tau]=P_{\pi,\mu}[\tau]\exp(\beta \lambda_GG[\tau]-\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]-\beta\lambda_NN[\tau]).
\tag{78}
\end{equation}
\]

したがって、$\beta\lambda_G$ は reward tilt の実際の指数強度、$\beta\lambda_{\mathrm{KL}}$ は KL 引き戻しの実際の指数強度、$\beta\lambda_N$ は経路長ペナルティの実際の指数強度である。$\beta$ は熱力学的類比における逆温度であり、実装上は annealing parameter と見なせるが、decoder temperature ではない。

### 4.3. Gibbs サンプリング：Boltzmann 重みに基づく一つのサンプリング方法

新しい Gibbs サンプリング分布を構成せず、元の observable insertion を純指数形式として書きたいだけなら、まず正値化リターン $\widetilde G[\tau]>0$ を取ることができる。$G[\tau]$ が負値を許す場合、形式的には負号を複素位相として書くことで、元の経路積分を複素重み経路積分へ変換し、さらに複素 Langevin 法を考えることもできる。しかし、これは不必要な phase/sign problem と複素確率過程の複雑性を導入する。本稿ではこの経路を取らず、実数重みを保つために正値化リターン $\widetilde G[\tau]>0$ を用いる。

\[
\begin{equation}
\widetilde G[\tau]=G[\tau]+c,\quad \widetilde G[\tau]>0.
\tag{79}
\end{equation}
\]

このとき : 

\[
\begin{equation}
\exp(-\beta H_0[\tau])\widetilde G[\tau]=\exp(-\beta H_0[\tau]+\log\widetilde G[\tau]).
\tag{80}
\end{equation}
\]

observable 等価作用を定義する : 

\[
\begin{equation}
S_{\mathrm{obs}}[\tau]=\beta H_0[\tau]-\log\widetilde G[\tau].
\tag{81}
\end{equation}
\]

したがって : 

\[
\begin{equation}
\widetilde J(\pi)=\int \exp(-S_{\mathrm{obs}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{82}
\end{equation}
\]

基礎経路密度が正規化されている場合、元の期待値と正値化期待値の間には次が成立する : 

\[
\begin{equation}
J(\pi)=\widetilde J(\pi)-c.
\tag{83}
\end{equation}
\]

この経路は元の observable insertion の指数化にすぎず、Gibbs サンプリングではない。

Gibbs サンプリングは別の事柄である。それは Boltzmann 重みに基づき、新しいサンプリング分布を構成する : 

\[
\begin{equation}
q_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}(\tau\mid\pi,\mu)=\frac{1}{Z_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}}\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{84}
\end{equation}
\]

分配関数は : 

\[
\begin{equation}
Z_{\beta,\lambda_G,\lambda_{\mathrm{KL}},\lambda_N}=\int \exp(-\beta H_{\mathrm{eff}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{85}
\end{equation}
\]

元の rollout 分布は : 

\[
\begin{equation}
p_0(\tau\mid\pi,\mu)=P_{\pi,\mu}[\tau]=\exp(-\beta H_0[\tau]).
\tag{86}
\end{equation}
\]

Gibbs 傾斜分布は : 

\[
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}p_0(\tau\mid\pi,\mu)\exp(\beta \lambda_GG[\tau]-\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]-\beta\lambda_NN[\tau]).
\tag{87}
\end{equation}
\]

外場がオフのときにのみ、Gibbs 分布は元のサンプリングへ退化する : 

\[
\begin{equation}
\lambda_G=0,\quad \lambda_{\mathrm{KL}}=0,\quad \lambda_N=0\quad\Longrightarrow\quad q(\tau\mid\pi,\mu)=p_0(\tau\mid\pi,\mu).
\tag{88}
\end{equation}
\]

したがって、Gibbs サンプリングは元のサンプリングではない。それは経路探索と訓練サンプル構成のために導入される reward-tilted ensemble である。

Gibbs 分布からサンプリングする場合 : 

\[
\begin{equation}
\tau_i\sim q(\tau\mid\pi,\mu),\quad i=1,\ldots,K,
\tag{89}
\end{equation}
\]

サンプル平均が推定するのは Gibbs ensemble 下の期待値である : 

\[
\begin{equation}
\widehat{\mathbb E}_{q}[G]=\frac{1}{K}\sum_{i=1}^{K}G[\tau_i].
\tag{90}
\end{equation}
\]

これは元の rollout 分布下の $J(\pi)$ ではない。元の期待値は : 

\[
\begin{equation}
J(\pi)=\mathbb E_{p_0}[G[\tau]].
\tag{91}
\end{equation}
\]

Gibbs サンプリング下の期待値は : 

\[
\begin{equation}
\mathbb E_q[G[\tau]]=\frac{\partial \log Z}{\partial(\beta \lambda_G)}.
\tag{92}
\end{equation}
\]

Gibbs サンプルから元の $J(\pi)$ を逆推定したい場合は、importance reweighting が必要である : 

\[
\begin{equation}
w_i=\exp(-\beta \lambda_GG[\tau_i]+\beta\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau_i]+\beta\lambda_NN[\tau_i]).
\tag{93}
\end{equation}
\]

したがって : 

\[
\begin{equation}
\widehat J(\pi)=\frac{\sum_{i=1}^{K}w_iG[\tau_i]}{\sum_{i=1}^{K}w_i}.
\tag{94}
\end{equation}
\]

外場が強いとき、逆向き重みの分散は大きくなる。したがって、目的が元の $J(\pi)$ の推定であるなら、$p_0$ から直接サンプリングすべきであり、または非常に小さい $\beta \lambda_G$ を用いて外挿するべきである。目的が探索を拡張し、高価値の訓練サンプルを生成することであるなら、Gibbs サンプリング分布 $q$ を直接使用できる。

$\beta \lambda_G$ が小さいとき : 

\[
\begin{equation}
\exp(\beta \lambda_GG[\tau])=1+\beta \lambda_GG[\tau]+O((\beta \lambda_G)^2).
\tag{95}
\end{equation}
\]

このとき Gibbs 分布は元の rollout 分布に近い : 

\[
\begin{equation}
q(\tau\mid\pi,\mu)\approx p_0(\tau\mid\pi,\mu).
\tag{96}
\end{equation}
\]

複数の小さい外場強度のもとでサンプリングする場合 : 

\[
\begin{equation}
g_j=\frac{1}{K_j}\sum_{i=1}^{K_j}G[\tau_i^{(j)}],\quad \tau_i^{(j)}\sim q_{\beta \lambda_G^{(j)}}(\tau\mid\pi,\mu),
\tag{97}
\end{equation}
\]

$g_j$ を用いて $\beta \lambda_G\to 0$ へ外挿することで、元の $J(\pi)$ を近似できる。代価はリソースコストの増加である。なぜなら、各外場強度ごとにサンプリングが必要だからである。

### 4.4. MCMC 経路サンプリング

シミュレーテッド・アニーリングを用いる場合、$k=0,1,\ldots,K_{\mathrm{ann}}$ は外部サンプラーのアニーリング反復ステップを表し、$\beta_k$ は第 $k$ ラウンドの逆温度である。目標経路分布は : 

\[
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{98}
\end{equation}
\]

現在の経路から候補経路を生成する : 

\[
\begin{equation}
\tau'\sim q_{\mathrm{prop}}(\tau'\mid\tau).
\tag{99}
\end{equation}
\]

Metropolis-Hastings 受理率は : 

\[
\begin{equation}
A_k(\tau\rightarrow\tau')=\min\left(1,\exp\left(-\beta_k(H_{\mathrm{eff}}[\tau']-H_{\mathrm{eff}}[\tau])\right)\frac{q_{\mathrm{prop}}(\tau\mid\tau')}{q_{\mathrm{prop}}(\tau'\mid\tau)}\right).
\tag{100}
\end{equation}
\]

ハミルトニアン差を展開すると : 

\[
\begin{align}
H_{\mathrm{eff}}[\tau']-H_{\mathrm{eff}}[\tau]
&=H_0[\tau']-H_0[\tau]-\lambda_G(G[\tau']-G[\tau]) \nonumber \\
&\quad +\lambda_{\mathrm{KL}}(D_{\mathrm{KL}}[\tau']-D_{\mathrm{KL}}[\tau])+\lambda_N(N[\tau']-N[\tau]).
\tag{101}
\end{align}
\]

したがって、高リターン、低 KL、短経路は $H_{\mathrm{eff}}$ を低下させ、より受理されやすくなる。MCMC は単にランダム性を増やすものではなく、Boltzmann 受理率を用いて経路を低ハミルトニアン領域へ押しやる。小さい $\beta_k$ のとき受理率はより寛容で、探索により有利である。大きい $\beta_k$ のとき受理率はより低ハミルトニアン経路に偏り、収束により有利である。

### 4.5. Langevin 経路サンプリング

Langevin は連続自由度上で行うのにより適しており、特に hidden action $a_t=z_t$ に適している。第 $k$ ラウンドのアニーリング反復における無次元作用は : 

\[
\begin{equation}
S_{\mathrm{eff}}^{(k)}[\tau]=\beta_k H_{\mathrm{eff}}[\tau].
\tag{102}
\end{equation}
\]

局所断片 $a_{u:v}$ に対して Langevin 更新を行う : 

\[
\begin{equation}
a_{u:v}^{(k+1)}=a_{u:v}^{(k)}-\epsilon\nabla_{a_{u:v}}S_{\mathrm{eff}}^{(k)}[\tau^{(k)}]+\sqrt{2\epsilon}\,\xi_k=a_{u:v}^{(k)}-\epsilon\beta_k\nabla_{a_{u:v}}H_{\mathrm{eff}}[\tau^{(k)}]+\sqrt{2\epsilon}\,\xi_k.
\tag{103}
\end{equation}
\]

ここで : 

\[
\begin{equation}
\xi_k\sim \mathcal N(0,I).
\tag{104}
\end{equation}
\]

物理的描像 : 
- Langevin = 作用勾配降下 + ランダム摂動
- $\beta_k$ は $S_{\mathrm{eff}}^{(k)}=\beta_k H_{\mathrm{eff}}$ を通じて、サンプリングにおけるハミルトニアン勾配の強度を制御する
- 小さい $\beta_k$ のとき、勾配選択圧は弱く、探索により有利である
- 大きい $\beta_k$ のとき、勾配選択圧は強まり、低ハミルトニアン経路へより偏る

### 4.6. 逆温度アニーリングと冷却描像

$k=0,1,\ldots,K_{\mathrm{ann}}$ は外部アニーリング反復ステップを表すものとする。第 $k$ ラウンドの Gibbs サンプリング分布は : 

\[
\begin{equation}
q_k(\tau)\propto \exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{105}
\end{equation}
\]

シミュレーテッド・アニーリングは逆温度を増加させることで実現される : 

\[
\begin{equation}
\beta_0<\beta_1<\cdots<\beta_{K_{\mathrm{ann}}}.
\tag{106}
\end{equation}
\]

小さい $\beta_k$ は高温探索に対応し、多くの高ハミルトニアン経路もなお保持され得る。大きい $\beta_k$ は低温収束に対応し、経路分布は徐々に低ハミルトニアン領域へ集中する。

第三節でもシミュレーテッド・アニーリングを用いることができるが、それがスケジューリングするのは $q_o^{(k)}$ と $q_r^{(k)}$ のような観測量と報酬 proposal であり、本節のシミュレーテッド・アニーリングがスケジューリングするのは Gibbs / Boltzmann 分布における逆温度 $\beta_k$ である。両者はいずれも早期に局所最適へ陥ることを防ぐことができるが、数学的対象は異なる。

### 4.7 関連研究

- 経路積分 RL
  - 対応する文献方向: Path Integral Control / PI${}^2$
  - 直接関連:
    - [[PMLR 2010] Learning Policy Improvements with Path Integrals](https://proceedings.mlr.press/v9/theodorou10a.html)
    - [[JMLR 2010] A Generalized Path Integral Control Approach to Reinforcement Learning](https://www.jmlr.org/papers/volume11/theodorou10a/theodorou10a.pdf)
  - 同方向:
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)

- KL ペナルティ / 制御コスト
  - 対応する文献方向: KL control / linearly-solvable MDP
  - 直接関連:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)
  - 同方向:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

- RL as inference
  - 対応する文献方向: maximum entropy RL / control as inference / SAC
  - 直接関連:
    - [[arXiv:1805.00909] Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
  - 同方向:
    - [[arXiv:0901.0633] Optimal Control as a Graphical Model Inference Problem](https://arxiv.org/abs/0901.0633)
    - [[NeurIPS 2006] Linearly-solvable Markov Decision Problems](https://papers.nips.cc/paper/3002-linearly-solvable-markov-decision-problems)

- $e^{-\beta H}$ サンプリング
  - 対応する文献方向: EBM / Gibbs / MCMC / Langevin
  - 直接関連:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2111.15141] Path Integral Sampler: a Stochastic Control Approach for Sampling](https://arxiv.org/abs/2111.15141)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - 同方向:
    - [[arXiv:1801.01290] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

- hidden latent 上の Langevin
  - 対応する文献方向: latent EBM / energy-based text generation / continuous relaxation
  - 直接関連:
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
    - [[ICML 2021] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://proceedings.mlr.press/v139/pang21a.html)
  - 同方向:
    - [[arXiv:1903.08689] Implicit Generation and Generalization in Energy-Based Models](https://arxiv.org/abs/1903.08689)
    - [[arXiv:2511.07124] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought](https://arxiv.org/abs/2511.07124)

- LLM 推論サンプリング
  - 対応する文献方向: MCMC-inspired reasoning / constrained sampling
  - 直接関連:
    - [[arXiv:2506.05754] Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective](https://arxiv.org/abs/2506.05754)
    - [[arXiv:2202.11705] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics](https://arxiv.org/abs/2202.11705)
  - 同方向:
    - [[arXiv:2510.14901] Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)

---

## 5. 二つの分岐の関係

二つの経路はいずれもサンプリング経路を拡張するために用いられるが、数学的対象は異なる。

第三節は RL estimator / rollout augmentation である。それは元の累積リターン定義から出発し、$o_{t+1}$ と $r_t$ に対して再サンプリングを行う : 

\[
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t).
\tag{107}
\end{equation}
\]

その後、リターンまたは advantage を推定する : 

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{108}
\end{equation}
\]

第四節は経路空間の Gibbs サンプリングである。それは新しい経路分布を構成する : 

\[
\begin{equation}
q_k(\tau\mid\pi,\mu)=\frac{1}{Z_k}\exp(-\beta_k H_{\mathrm{eff}}[\tau]).
\tag{109}
\end{equation}
\]

したがって、第三節は PPO / GRPO / GSPO に接続でき、重点は rollout データとリターン推定を改善することにある。第四節は MCMC / Langevin / アニーリングを使用して、低ハミルトニアン経路を直接サンプリングする。統計力学の描像は、両者における経路選択とアニーリング挙動を説明するのに役立つ。

---

## 6. まとめ

基礎作用と基礎ハミルトニアンは次を満たす : 

\[
\begin{equation}
S_{\pi,\mu}[\tau]=\beta H_0[\tau]=-
\sum_{t=0}^{T}\log\pi(a_t\mid h_t)-
\sum_{t=0}^{T}\log\mu(o_{t+1},r_t\mid a_t,h_t).
\tag{110}
\end{equation}
\]

元の RL 目的は : 

\[
\begin{equation}
J(\pi)=\int \exp(-\beta H_0[\tau])G[\tau]\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{111}
\end{equation}
\]

第三節は元の RL 累積リターン定義を用いて、観測量と報酬に対して再サンプリングを行う : 

\[
\begin{equation}
(o_{t+1}^{(m,k)},r_t^{(m,k)})\sim q_o^{(k)}(o_{t+1}\mid a_t,h_t)q_r^{(k)}(r_t\mid o_{t+1},a_t,h_t).
\tag{112}
\end{equation}
\]

第三節で得られるリターン推定は PPO / GRPO / GSPO へ渡すことができる : 

\[
\begin{equation}
\widehat G_i^{(k)}=\frac{1}{M}\sum_{m=1}^{M}\sum_{t=0}^{T_i}\gamma^t r_{i,t}^{(m,k)}.
\tag{113}
\end{equation}
\]

第四節は外場ハミルトニアンを導入する : 

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_NN[\tau].
\tag{114}
\end{equation}
\]

等価ハミルトニアンは : 

\[
\begin{equation}
H_{\mathrm{eff}}[\tau]=H_0[\tau]+H_{\mathrm{field}}[\tau].
\tag{115}
\end{equation}
\]

等価作用は : 

\[
\begin{equation}
S_{\mathrm{eff}}[\tau]=\beta H_{\mathrm{eff}}[\tau].
\tag{116}
\end{equation}
\]

第四節の Gibbs サンプリング分布は : 

\[
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}\exp(-S_{\mathrm{eff}}[\tau])=\frac{1}{Z}\exp(-\beta H_{\mathrm{eff}}[\tau]).
\tag{117}
\end{equation}
\]

経路サンプリングの流れは次のようにまとめられる : 

\[
\begin{align*}
&~ \text{rollout / proposal} \\
\rightarrow &~ \text{branch 1: resample }o,r\text{ and estimate }G \\
\rightarrow &~ \text{or branch 2: evaluate }H_{\mathrm{eff}}[\tau]\text{ and sample by Boltzmann weight} \\
\rightarrow &~ \text{PPO / GRPO / GSPO update or MCMC / Langevin search} \\
\rightarrow &~ \text{distill back to }\pi_\theta.
\end{align*}
\]

物理的描像 : 
- history-based RL は一次元時間経路積分
- $a,o,r$ は経路内部自由度
- hidden $z$ は連続動作 $a$ として用いることができる
- 元の期待値は $\int e^{-\beta H_0}G$ 
- 第三節は元の RL 推定器上で $o,r$ フィードバック空間を拡張
- 第四節は外場ハミルトニアン $H_{\mathrm{field}}$ を導入し、Boltzmann 重み $e^{-\beta(H_0+H_{\mathrm{field}})}$ を通じて、高リターン、低 KL、制御された経路長に偏った経路を意図的に選択

---

# Appendix A : 任意の観測信頼度と報酬信頼度

世界モデルまたは報酬モデル自体に信頼度ペナルティが必要である場合、これらの項は任意の外場として加えることができ、主体公式に書き込む必要はない。観測信頼度項は次のように書ける : 

\[
\begin{equation}
C_o[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_o(o_{t+1}\mid a_t,h_t).
\tag{A1}
\end{equation}
\]

報酬信頼度項は次のように書ける : 

\[
\begin{equation}
C_r[\tau]=-
\sum_{t=0}^{T}\log \hat\mu_r(r_t\mid o_{t+1},a_t,h_t).
\tag{A2}
\end{equation}
\]

これらの任意項を使用する場合、外場ハミルトニアンは次のように拡張される : 

\[
\begin{equation}
H_{\mathrm{field}}[\tau]=-\lambda_GG[\tau]+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}[\tau]+\lambda_NN[\tau]+\rho_o C_o[\tau]+\rho_r C_r[\tau].
\tag{A3}
\end{equation}
\]

対応する Gibbs サンプリング分布は依然として : 

\[
\begin{equation}
q(\tau\mid\pi,\mu)=\frac{1}{Z}\exp(-\beta(H_0[\tau]+H_{\mathrm{field}}[\tau])).
\tag{A4}
\end{equation}
\]

である。

---

# Appendix B : 時間方向の繰り込み

これは一次元時間経路積分であるため、繰り込みは主に時間方向に沿って行われる。$\ell$ を巨視的時間ブロック番号とし、各 $b$ 個の微視的 action を一つの巨視的ブロックへ統合する : 

\[
\begin{equation}
A_\ell=C_\phi(a_{\ell b},a_{\ell b+1},\ldots,a_{(\ell+1)b-1}).
\tag{B1}
\end{equation}
\]

多層圧縮後 : 

\[
\begin{equation}
T\longrightarrow \frac{T}{b}\longrightarrow \frac{T}{b^2}\longrightarrow \cdots \longrightarrow \frac{T}{b^N}.
\tag{B2}
\end{equation}
\]

経路積分の層では、微視的経路から巨視的経路への写像を $\bar\tau=\mathcal C(\tau)$ とし、巨視的有効作用は微視的自由度を積分消去することで得られる : 

\[
\begin{equation}
\exp(-S_{\mathrm{eff}}[\bar\tau])=\int_{\mathcal C(\tau)=\bar\tau}\exp(-S_{\mathrm{eff}}[\tau])\prod_{t=0}^{T}da_t\,do_{t+1}\,dr_t.
\tag{B3}
\end{equation}
\]

圧縮ブロック内部の有効特異値スペクトルが急速に減衰するなら、切断は安全である。スペクトルがほぼ平坦であるなら、硬い切断は大量の情報損失をもたらす : 

\[
\begin{equation}
\sigma_1\approx\sigma_2\approx\cdots\approx\sigma_m\quad\Longrightarrow\quad m\rightarrow\chi\text{ の切断は強い情報損失を引き起こす}.
\tag{B4}
\end{equation}
\]

物理的描像 : 
- temporal RG は長時間経路を巨視的経路へ圧縮できるが、モデルは compressed token / hidden macro-action をサポートしていなければならない。そうでなければ、圧縮は単なる要約であり、有効自由度ではない
