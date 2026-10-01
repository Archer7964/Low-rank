# Adjoint for DLR

## 1. Original Problem

Given an initial distribution $f_0$ and a target distribution $f^{\mathrm{eq}}$, choose the external field $H$ to minimize the terminal mismatch.

**Vlasov–Poisson equation with $H$:**

$$
\begin{aligned}
\partial_t f + v\,\partial_x f - (E+H)\,\partial_v f &= 0,
\qquad 0<t\leq T, \\
\partial_x E &= 1-\rho_f, \\
\rho_f(t,x) &= \int_{\mathbb R} f(t,x,v)\,dv, \\
f(0,x,v) &= f_0(x,v).
\end{aligned}
$$

**Objective:**

$$
\min_H J(H),
\qquad
J(H)=\frac12\left\|f(H)(T)-f^{\mathrm{eq}}\right\|_{L^2_{x,v}}^2,
$$

where $f(H)$ denotes the solution corresponding to $H$.

**Setup used in the code:**

$$
x\in[0,L_x),\qquad L_x=\frac{2\pi}{\beta}=10\pi,
\qquad v\in[-6,6),\qquad \beta=0.2,\qquad T=20.
$$

Periodic boundary conditions are used in both $x$ and the truncated velocity domain. The grid has $n_x=n_v=128$ points. The periodic FFT Poisson solver imposes

$$
\widehat E_0=0
\qquad\Longrightarrow\qquad
\Delta x\sum_{i=0}^{n_x-1}E_i=0.
$$

The control is independent of time:

$$
H(x)=\sum_{j=1}^{k}a_j\psi_j(x),
\qquad a=(a_1,\ldots,a_k)\in\mathbb R^k,
\qquad k=10,
$$

$$
\begin{aligned}
\psi_{2m-1}(x)&=\sqrt{\frac{2}{L_x}}\cos(m\beta x),\\
\psi_{2m}(x)&=\sqrt{\frac{2}{L_x}}\sin(m\beta x),
\qquad m=1,\ldots,5.
\end{aligned}
$$

Thus $H$ has zero spatial mean, and the optimization is over the coefficients $a$, with no coefficient bounds or control regularization.

## 2. Forward: K–S–L

With $W_x=\Delta x\,I$ and $W_v=\Delta v\,I$, write

$$
\begin{aligned}
f_r^n &= U^nS^n(V^n)^\top,\\
(U^n)^\top W_xU^n &= I_r,
\qquad (V^n)^\top W_vV^n=I_r.
\end{aligned}
$$

Let $F_a^n=\operatorname{RHS}(f_a^n,H)$ denote the spatially discretized Vlasov–Poisson RHS. One K–S–L step, using forward Euler in each substep, is

**K-step:**

$$
\begin{aligned}
f_0^n &= U^nS^n(V^n)^\top,\\
K_*^n &= U^nS^n+\Delta t\,F_0^nW_vV^n,\\
(U^{n+1},\widehat S^n)
&=\operatorname{QR}_{W_x,+}(K_*^n).
\end{aligned}
$$

**S-step:**

$$
\begin{aligned}
f_1^n &= U^{n+1}\widehat S^n(V^n)^\top,\\
\widetilde S^n
&=\widehat S^n-\Delta t\,(U^{n+1})^\top W_xF_1^nW_vV^n.
\end{aligned}
$$

**L-step:**

$$
\begin{aligned}
f_2^n &= U^{n+1}\widetilde S^n(V^n)^\top,\\
L_*^n &= V^n(\widetilde S^n)^\top
+\Delta t\,(F_2^n)^\top W_xU^{n+1},\\
(V^{n+1},(S^{n+1})^\top)
&=\operatorname{QR}_{W_v,+}(L_*^n).
\end{aligned}
$$

Here $\operatorname{QR}_{W,+}(A)=(Q,R)$ means $A=QR$, $Q^\top WQ=I_r$, with positive diagonal entries in $R$. The updated state is

$$
f_r^{n+1}=U^{n+1}S^{n+1}(V^{n+1})^\top.
$$

## 3. Discrete Adjoint

### Lagrangian

$N\Delta t=T$, $w=\Delta x\,\Delta v$. Inner product:

$$
\langle A,B\rangle_F=\operatorname{tr}(A^\top B).
$$

Objective:

$$
J_h(U^N,S^N,V^N)
=\frac{w}{2}
\left\|U^NS^N(V^N)^\top-f^{\mathrm{eq}}\right\|_F^2,
$$

$$
\mathcal L_d
=J_h+\sum_{n=0}^{N-1}
\left(\ell_K^n+\ell_S^n+\ell_L^n\right),
$$

$$
\begin{aligned}
\ell_K^n
&=\left\langle P_K^n,
U^nS^n+\Delta t\,F_0^nW_vV^n-K_*^n
\right\rangle_F,\\
\ell_S^n
&=\left\langle P_S^n,
\widehat S^n-\Delta t\,(U^{n+1})^\top W_xF_1^nW_vV^n
-\widetilde S^n\right\rangle_F,\\
\ell_L^n
&=\left\langle P_L^n,
V^n(\widetilde S^n)^\top
+\Delta t\,(F_2^n)^\top W_xU^{n+1}-L_*^n
\right\rangle_F.
\end{aligned}
$$

Free variables: $H$ and $\{K_*^n,\widetilde S^n,L_*^n\}_{n=0}^{N-1}$; $U^{n+1},\widehat S^n,V^{n+1},S^{n+1}$ come from QR. Other free variables stay fixed in each variation.

$$
\delta\mathcal L_d
=\delta J_h+\sum_{n=0}^{N-1}
\left(\delta\ell_K^n+\delta\ell_S^n+\delta\ell_L^n\right),
$$

Forward equations $\Rightarrow$ terms with $\delta P_K^n,\delta P_S^n,\delta P_L^n$ are zero. The remaining terms:

$$
\begin{aligned}
\delta\ell_K^n
&=\Big\langle P_K^n,
(\delta U^n)S^n+U^n\delta S^n
+\Delta t\,\delta F_0^nW_vV^n\\
&\hspace{38mm}+\Delta t\,F_0^nW_v\delta V^n-\delta K_*^n
\Big\rangle_F,
\end{aligned}
$$

$$
\begin{aligned}
\delta\ell_S^n
&=\Big\langle P_S^n,
\delta\widehat S^n
-\Delta t\,(\delta U^{n+1})^\top W_xF_1^nW_vV^n\\
&\hspace{20mm}-\Delta t\,(U^{n+1})^\top W_x\delta F_1^nW_vV^n\\
&\hspace{20mm}-\Delta t\,(U^{n+1})^\top W_xF_1^nW_v\delta V^n
-\delta\widetilde S^n\Big\rangle_F,
\end{aligned}
$$

$$
\begin{aligned}
\delta\ell_L^n
&=\Big\langle P_L^n,
(\delta V^n)(\widetilde S^n)^\top
+V^n(\delta\widetilde S^n)^\top\\
&\hspace{20mm}+\Delta t\,(\delta F_2^n)^\top W_xU^{n+1}
+\Delta t\,(F_2^n)^\top W_x\delta U^{n+1}
-\delta L_*^n\Big\rangle_F.
\end{aligned}
$$

$$
\begin{aligned}
\delta J_h
&=\Big\langle w\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)V^N(S^N)^\top,
\delta U^N\Big\rangle_F\\
&\quad+\Big\langle w(U^N)^\top\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)V^N,
\delta S^N\Big\rangle_F\\
&\quad+\Big\langle w\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)^\top U^NS^N,
\delta V^N\Big\rangle_F.
\end{aligned}
$$

**$\bar F_a^n$ and $\bar f_a^n$.**

$$
\begin{aligned}
\left\langle P_K^n,\Delta t\,\delta F_0^nW_vV^n\right\rangle_F
&=\Delta t\,\operatorname{tr}
\left((P_K^n)^\top\delta F_0^nW_vV^n\right)\\
&=\left\langle
\Delta t\,P_K^n(V^n)^\top W_v,\delta F_0^n
\right\rangle_F.
\end{aligned}
$$

Set

$$
\begin{aligned}
\bar F_0^n&=\Delta t\,P_K^n(V^n)^\top W_v,\\
\bar F_1^n&=-\Delta t\,W_xU^{n+1}P_S^n(V^n)^\top W_v,\\
\bar F_2^n&=\Delta t\,W_xU^{n+1}(P_L^n)^\top.
\end{aligned}
$$

$H$ fixed:

$$
\begin{aligned}
\delta F_a^n
&=D_f\operatorname{RHS}(f_a^n,H)[\delta f_a^n],\\
\bar f_a^n
&=\left[D_f\operatorname{RHS}(f_a^n,H)\right]^*[\bar F_a^n],\\
\left\langle\bar F_a^n,\delta F_a^n\right\rangle_F
&=\left\langle\bar f_a^n,\delta f_a^n\right\rangle_F,
\qquad a=0,1,2.
\end{aligned}
$$

### $\delta L_*^n$

**Related:**

$$
L_*^n
\xrightarrow{\mathrm{QR}_{W_v,+}}
\big(V^{n+1},(S^{n+1})^\top\big)
\longrightarrow
\ell_K^{n+1},\ \ell_S^{n+1},\ \ell_L^{n+1}.
$$

For $n=0,\ldots,N-2$,

$$
\delta_{L_*^n}\mathcal L_d
=-\langle P_L^n,\delta L_*^n\rangle_F
+\delta_{L_*^n}\ell_K^{n+1}
+\delta_{L_*^n}\ell_S^{n+1}
+\delta_{L_*^n}\ell_L^{n+1}.
$$

**From $\ell_K^{n+1}$:**

$$
\begin{aligned}
\delta_{L_*^n}\ell_K^{n+1}
&=\left\langle (U^{n+1})^\top P_K^{n+1},\delta S^{n+1}\right\rangle_F\\
&\quad+\left\langle
\Delta t\,W_v(F_0^{n+1})^\top P_K^{n+1},\delta V^{n+1}
\right\rangle_F
+\left\langle\bar F_0^{n+1},\delta_{L_*^n}F_0^{n+1}\right\rangle_F.
\end{aligned}
$$

$$
\begin{aligned}
\left\langle\bar F_0^{n+1},\delta_{L_*^n}F_0^{n+1}\right\rangle_F
&=\left\langle\bar f_0^{n+1},\delta_{L_*^n}f_0^{n+1}\right\rangle_F\\
&=\Big\langle\bar f_0^{n+1},
U^{n+1}(\delta S^{n+1})(V^{n+1})^\top
+U^{n+1}S^{n+1}(\delta V^{n+1})^\top\Big\rangle_F\\
&=\left\langle
(U^{n+1})^\top\bar f_0^{n+1}V^{n+1},\delta S^{n+1}
\right\rangle_F\\
&\quad+\left\langle
(\bar f_0^{n+1})^\top U^{n+1}S^{n+1},\delta V^{n+1}
\right\rangle_F.
\end{aligned}
$$

So

$$
\begin{aligned}
\delta_{L_*^n}\ell_K^{n+1}
&=\Big\langle
(U^{n+1})^\top P_K^{n+1}
+(U^{n+1})^\top\bar f_0^{n+1}V^{n+1},
\delta S^{n+1}\Big\rangle_F\\
&\quad+\Big\langle
\Delta t\,W_v(F_0^{n+1})^\top P_K^{n+1}
+(\bar f_0^{n+1})^\top U^{n+1}S^{n+1},
\delta V^{n+1}\Big\rangle_F.
\end{aligned}
$$

**From $\ell_S^{n+1}$:**

$$
\begin{aligned}
\delta_{L_*^n}\ell_S^{n+1}
&=\Big\langle
-\Delta t\,W_v(F_1^{n+1})^\top W_xU^{n+2}P_S^{n+1},
\delta V^{n+1}\Big\rangle_F\\
&\quad+\left\langle\bar F_1^{n+1},\delta_{L_*^n}F_1^{n+1}\right\rangle_F\\
&=\Big\langle
-\Delta t\,W_v(F_1^{n+1})^\top W_xU^{n+2}P_S^{n+1},
\delta V^{n+1}\Big\rangle_F\\
&\quad+\left\langle\bar f_1^{n+1},
U^{n+2}\widehat S^{n+1}(\delta V^{n+1})^\top\right\rangle_F\\
&=\Big\langle
-\Delta t\,W_v(F_1^{n+1})^\top W_xU^{n+2}P_S^{n+1}
+(\bar f_1^{n+1})^\top U^{n+2}\widehat S^{n+1},
\delta V^{n+1}\Big\rangle_F.
\end{aligned}
$$

**From $\ell_L^{n+1}$:**

$$
\begin{aligned}
\delta_{L_*^n}\ell_L^{n+1}
&=\left\langle P_L^{n+1}\widetilde S^{n+1},\delta V^{n+1}\right\rangle_F
+\left\langle\bar F_2^{n+1},\delta_{L_*^n}F_2^{n+1}\right\rangle_F\\
&=\left\langle P_L^{n+1}\widetilde S^{n+1},\delta V^{n+1}\right\rangle_F
+\left\langle\bar f_2^{n+1},
U^{n+2}\widetilde S^{n+1}(\delta V^{n+1})^\top\right\rangle_F\\
&=\Big\langle
P_L^{n+1}\widetilde S^{n+1}
+(\bar f_2^{n+1})^\top U^{n+2}\widetilde S^{n+1},
\delta V^{n+1}\Big\rangle_F.
\end{aligned}
$$

**QR:** $Q_W,R_W$ are the two outputs of $\operatorname{QR}_{W,+}$.

$$
\begin{aligned}
\delta V^{n+1}&=DQ_{W_v}(L_*^n)[\delta L_*^n],\\
\delta S^{n+1}&=\big(DR_{W_v}(L_*^n)[\delta L_*^n]\big)^\top.
\end{aligned}
$$

Add the three terms above and set $\delta_{L_*^n}\mathcal L_d=0$:

$$
\begin{aligned}
P_L^n
&=DQ_{W_v}(L_*^n)^*\left[
\begin{aligned}
&\Delta t\,W_v(F_0^{n+1})^\top P_K^{n+1}
+(\bar f_0^{n+1})^\top U^{n+1}S^{n+1}\\
&-\Delta t\,W_v(F_1^{n+1})^\top W_xU^{n+2}P_S^{n+1}
+(\bar f_1^{n+1})^\top U^{n+2}\widehat S^{n+1}\\
&+P_L^{n+1}\widetilde S^{n+1}
+(\bar f_2^{n+1})^\top U^{n+2}\widetilde S^{n+1}
\end{aligned}\right]\\
&\quad+DR_{W_v}(L_*^n)^*\left[
\Big((U^{n+1})^\top P_K^{n+1}
+(U^{n+1})^\top\bar f_0^{n+1}V^{n+1}\Big)^\top
\right].
\end{aligned}
$$

**At $n=N-1$:** next-step terms $\to J_h$.

$$
\delta_{L_*^{N-1}}\mathcal L_d
=-\langle P_L^{N-1},\delta L_*^{N-1}\rangle_F
+\delta_{L_*^{N-1}}J_h=0.
$$

Here $\delta U^N=0$, so the $\delta S^N$ and $\delta V^N$ terms in $\delta J_h$ give

$$
\begin{aligned}
P_L^{N-1}
&=DQ_{W_v}(L_*^{N-1})^*\left[
w\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)^\top U^NS^N
\right]\\
&\quad+DR_{W_v}(L_*^{N-1})^*\left[
\Big(w(U^N)^\top\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)V^N\Big)^\top
\right].
\end{aligned}
$$

### $\delta\widetilde S^n$

**Related:** $\ell_S^n$, $\ell_L^n$; $f_2^n=U^{n+1}\widetilde S^n(V^n)^\top$.

$$
\delta_{\widetilde S^n}\ell_S^n
=-\langle P_S^n,\delta\widetilde S^n\rangle_F.
$$

$$
\begin{aligned}
\delta_{\widetilde S^n}\ell_L^n
&=\left\langle P_L^n,V^n(\delta\widetilde S^n)^\top\right\rangle_F
+\Delta t\left\langle P_L^n,
(\delta_{\widetilde S^n}F_2^n)^\top W_xU^{n+1}\right\rangle_F,\\
\delta_{\widetilde S^n}F_2^n
&=D_f\operatorname{RHS}(f_2^n,H)
\left[U^{n+1}(\delta\widetilde S^n)(V^n)^\top\right].
\end{aligned}
$$

For the RHS term,

$$
\begin{aligned}
&\Delta t\left\langle P_L^n,
(\delta_{\widetilde S^n}F_2^n)^\top W_xU^{n+1}\right\rangle_F\\
&\quad=\left\langle\Delta t\,W_xU^{n+1}(P_L^n)^\top,
\delta_{\widetilde S^n}F_2^n\right\rangle_F\\
&\quad=\left\langle\bar F_2^n,\delta_{\widetilde S^n}F_2^n\right\rangle_F
=\left\langle\bar f_2^n,\delta_{\widetilde S^n}f_2^n\right\rangle_F\\
&\quad=\left\langle\bar f_2^n,
U^{n+1}(\delta\widetilde S^n)(V^n)^\top\right\rangle_F\\
&\quad=\left\langle(U^{n+1})^\top\bar f_2^nV^n,
\delta\widetilde S^n\right\rangle_F.
\end{aligned}
$$

So

$$
\begin{aligned}
\delta_{\widetilde S^n}\mathcal L_d
&=\Big\langle-P_S^n+(P_L^n)^\top V^n
+(U^{n+1})^\top\bar f_2^nV^n,
\delta\widetilde S^n\Big\rangle_F=0,\\
P_S^n&=(P_L^n)^\top V^n+(U^{n+1})^\top\bar f_2^nV^n.
\end{aligned}
$$

### $\delta K_*^n$

**Related:**

$$
K_*^n\xrightarrow{\mathrm{QR}_{W_x,+}}(U^{n+1},\widehat S^n)
\longrightarrow\ell_S^n,\ \ell_L^n,\ \ell_K^{n+1}.
$$

For $n=0,\ldots,N-2$,

$$
\delta_{K_*^n}\mathcal L_d
=-\langle P_K^n,\delta K_*^n\rangle_F
+\delta_{K_*^n}\ell_S^n
+\delta_{K_*^n}\ell_L^n
+\delta_{K_*^n}\ell_K^{n+1}.
$$

**From $\ell_S^n$:**

$$
\begin{aligned}
\delta_{K_*^n}f_1^n
&=(\delta U^{n+1})\widehat S^n(V^n)^\top
+U^{n+1}(\delta\widehat S^n)(V^n)^\top,\\
\delta_{K_*^n}\ell_S^n
&=\langle P_S^n,\delta\widehat S^n\rangle_F
-\Delta t\left\langle P_S^n,
(\delta U^{n+1})^\top W_xF_1^nW_vV^n\right\rangle_F\\
&\quad-\Delta t\left\langle P_S^n,
(U^{n+1})^\top W_x(\delta_{K_*^n}F_1^n)W_vV^n\right\rangle_F\\
&=\langle P_S^n,\delta\widehat S^n\rangle_F
+\left\langle-\Delta t\,W_xF_1^nW_vV^n(P_S^n)^\top,
\delta U^{n+1}\right\rangle_F
+\left\langle\bar f_1^n,\delta_{K_*^n}f_1^n\right\rangle_F\\
&=\Big\langle-\Delta t\,W_xF_1^nW_vV^n(P_S^n)^\top
+\bar f_1^nV^n(\widehat S^n)^\top,\delta U^{n+1}\Big\rangle_F\\
&\quad+\Big\langle P_S^n+(U^{n+1})^\top\bar f_1^nV^n,
\delta\widehat S^n\Big\rangle_F.
\end{aligned}
$$

**From $\ell_L^n$:** $\widetilde S^n$ stays fixed.

$$
\begin{aligned}
\delta_{K_*^n}f_2^n
&=(\delta U^{n+1})\widetilde S^n(V^n)^\top,\\
\delta_{K_*^n}\ell_L^n
&=\Delta t\left\langle P_L^n,
(\delta_{K_*^n}F_2^n)^\top W_xU^{n+1}\right\rangle_F
+\Delta t\left\langle P_L^n,(F_2^n)^\top W_x\delta U^{n+1}\right\rangle_F\\
&=\left\langle\bar f_2^n,\delta_{K_*^n}f_2^n\right\rangle_F
+\left\langle\Delta t\,W_xF_2^nP_L^n,\delta U^{n+1}\right\rangle_F\\
&=\left\langle\bar f_2^n,(\delta U^{n+1})\widetilde S^n(V^n)^\top\right\rangle_F
+\left\langle\Delta t\,W_xF_2^nP_L^n,\delta U^{n+1}\right\rangle_F\\
&=\Big\langle\Delta t\,W_xF_2^nP_L^n
+\bar f_2^nV^n(\widetilde S^n)^\top,\delta U^{n+1}\Big\rangle_F.
\end{aligned}
$$

**From $\ell_K^{n+1}$:** $S^{n+1}$ and $V^{n+1}$ stay fixed.

$$
\begin{aligned}
\delta_{K_*^n}f_0^{n+1}
&=(\delta U^{n+1})S^{n+1}(V^{n+1})^\top,\\
\delta_{K_*^n}\ell_K^{n+1}
&=\left\langle P_K^{n+1},(\delta U^{n+1})S^{n+1}\right\rangle_F
+\left\langle\bar F_0^{n+1},\delta_{K_*^n}F_0^{n+1}\right\rangle_F\\
&=\left\langle P_K^{n+1}(S^{n+1})^\top,\delta U^{n+1}\right\rangle_F
+\left\langle\bar f_0^{n+1},(\delta U^{n+1})S^{n+1}(V^{n+1})^\top\right\rangle_F\\
&=\Big\langle P_K^{n+1}(S^{n+1})^\top
+\bar f_0^{n+1}V^{n+1}(S^{n+1})^\top,\delta U^{n+1}\Big\rangle_F.
\end{aligned}
$$

**At $n=N-1$:** $\ell_K^{n+1}\to J_h$; $S^N,V^N$ stay fixed.

$$
\delta_{K_*^{N-1}}J_h
=\Big\langle w\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)V^N(S^N)^\top,
\delta U^N\Big\rangle_F.
$$

**Collect the coefficients:**

$$
\begin{aligned}
C_U^n
&=-\Delta t\,W_xF_1^nW_vV^n(P_S^n)^\top
+\bar f_1^nV^n(\widehat S^n)^\top\\
&\quad+\Delta t\,W_xF_2^nP_L^n
+\bar f_2^nV^n(\widetilde S^n)^\top\\
&\quad+\begin{cases}
P_K^{n+1}(S^{n+1})^\top
+\bar f_0^{n+1}V^{n+1}(S^{n+1})^\top,
&n<N-1,\\
w\big(U^NS^N(V^N)^\top-f^{\mathrm{eq}}\big)V^N(S^N)^\top,
&n=N-1,
\end{cases}\\
C_{\widehat S}^n
&=P_S^n+(U^{n+1})^\top\bar f_1^nV^n.
\end{aligned}
$$

**QR:**

$$
\delta U^{n+1}=DQ_{W_x}(K_*^n)[\delta K_*^n],
\qquad
\delta\widehat S^n=DR_{W_x}(K_*^n)[\delta K_*^n].
$$

$$
\begin{aligned}
\delta_{K_*^n}\mathcal L_d
&=-\langle P_K^n,\delta K_*^n\rangle_F
+\langle C_U^n,\delta U^{n+1}\rangle_F
+\langle C_{\widehat S}^n,\delta\widehat S^n\rangle_F\\
&=\Big\langle-P_K^n
+DQ_{W_x}(K_*^n)^*[C_U^n]
+DR_{W_x}(K_*^n)^*[C_{\widehat S}^n],
\delta K_*^n\Big\rangle_F=0,\\
P_K^n&=DQ_{W_x}(K_*^n)^*[C_U^n]
+DR_{W_x}(K_*^n)^*[C_{\widehat S}^n].
\end{aligned}
$$

### $\delta H$

Stage states fixed:

$$
\delta_H f_a^n=0,
\qquad
\delta_H F_a^n=D_H\operatorname{RHS}(f_a^n,H)[\delta H],
\qquad a=0,1,2.
$$

$$
\begin{aligned}
\delta_H\mathcal L_d
&=\sum_{n=0}^{N-1}\sum_{a=0}^{2}
\left\langle\bar F_a^n,
D_H\operatorname{RHS}(f_a^n,H)[\delta H]\right\rangle_F\\
&=\left\langle
\sum_{n=0}^{N-1}\sum_{a=0}^{2}
\left[D_H\operatorname{RHS}(f_a^n,H)\right]^*[\bar F_a^n],
\delta H\right\rangle.
\end{aligned}
$$

With the adjoint equations, $\delta J_h=\delta_H\mathcal L_d$. Hence

$$
\nabla_HJ_h
=\sum_{n=0}^{N-1}\sum_{a=0}^{2}
\left[D_H\operatorname{RHS}(f_a^n,H)\right]^*[\bar F_a^n].
$$

Here $\langle b,c\rangle=b^\top c$ for the grid values of $H$.

### QR

$W$ fixed; $A$ has full column rank:

$$
A=QR,\qquad Q^\top WQ=I_r,
\qquad R\text{ upper triangular with positive diagonal}.
$$

Differentiate:

$$
\delta A=(\delta Q)R+Q\delta R,
\qquad
(\delta Q)^\top WQ+Q^\top W\delta Q=0.
$$

Set

$$
M=Q^\top W\delta A\,R^{-1}
=Q^\top W\delta Q+\delta R\,R^{-1}.
$$

Then

$$
M+M^\top
=\delta R\,R^{-1}+(\delta R\,R^{-1})^\top.
$$

$\delta R\,R^{-1}$ is upper triangular. Set

$$
\mathcal U(M)_{ij}
=\begin{cases}
0,&i>j,\\
M_{ii},&i=j,\\
M_{ij}+M_{ji},&i<j.
\end{cases}
$$

So

$$
\begin{aligned}
DR_W(A)[\delta A]
=\delta R
&=\mathcal U(Q^\top W\delta A\,R^{-1})R,\\
DQ_W(A)[\delta A]
=\delta Q
&=\delta A\,R^{-1}
-Q\mathcal U(Q^\top W\delta A\,R^{-1}).
\end{aligned}
$$

For $DQ_W^*$ and $DR_W^*$, set

$$
\operatorname{copyltu}(M)_{ij}
=\begin{cases}
M_{ij},&i\geq j,\\
M_{ji},&i<j.
\end{cases}
$$

Then

$$
\langle N,\mathcal U(M)\rangle_F
=\left\langle\operatorname{copyltu}(N^\top),M\right\rangle_F.
$$

Then

$$
\begin{aligned}
\langle B,\delta Q\rangle_F
&=\left\langle B,\delta A\,R^{-1}\right\rangle_F
-\left\langle Q^\top B,\mathcal U(M)\right\rangle_F\\
&=\left\langle
\left[B-WQ\operatorname{copyltu}(B^\top Q)\right]R^{-\top},
\delta A\right\rangle_F,\\
\langle C,\delta R\rangle_F
&=\left\langle CR^\top,\mathcal U(M)\right\rangle_F\\
&=\left\langle
WQ\operatorname{copyltu}(RC^\top)R^{-\top},
\delta A\right\rangle_F.
\end{aligned}
$$

So

$$
\begin{aligned}
DQ_W(A)^*[B]
&=\left[B-WQ\operatorname{copyltu}(B^\top Q)\right]R^{-\top},\\
DR_W(A)^*[C]
&=WQ\operatorname{copyltu}(RC^\top)R^{-\top}.
\end{aligned}
$$

### RHS* (central differences)

Central-difference part:

$$
F_{\mathrm c}=\operatorname{RHS}_{\mathrm c}(f,H)
=-vD_xf+(E+H)D_vf,
\qquad D_x^*=-D_x,\quad D_v^*=-D_v.
$$

Pointwise products with $v$, $E+H$, and $\delta E$. $H$ fixed:

$$
\delta F_{\mathrm c}
=-vD_x\delta f+(E+H)D_v\delta f+\delta E\,D_vf.
$$

Poisson solver $\mathcal P$:

$$
\begin{aligned}
E&=\mathcal P(1-\rho_f),
\qquad (\rho_f)_i=\Delta v\sum_j f_{ij},\\
\delta E&=-\mathcal P\,\delta\rho_f,
\qquad (\delta\rho_f)_i=\Delta v\sum_j\delta f_{ij}.
\end{aligned}
$$

For $B$, set

$$
\bar E_i=\sum_j B_{ij}(D_vf)_{ij}.
$$

Field term:

$$
\begin{aligned}
\left\langle B,\delta E\,D_vf\right\rangle_F
&=\sum_i\delta E_i\sum_jB_{ij}(D_vf)_{ij}\\
&=\langle\bar E,\delta E\rangle
=-\langle\mathcal P^*\bar E,\delta\rho_f\rangle\\
&=-\Delta v\sum_{i,j}(\mathcal P^*\bar E)_i\,\delta f_{ij}.
\end{aligned}
$$

Transport terms:

$$
\left\langle B,-vD_x\delta f+(E+H)D_v\delta f\right\rangle_F
=\left\langle vD_xB-(E+H)D_vB,\delta f\right\rangle_F,
$$

So

$$
\left(\left[D_f\operatorname{RHS}_{\mathrm c}(f,H)\right]^*[B]\right)_{ij}
=\left(vD_xB-(E+H)D_vB\right)_{ij}
-\Delta v\,(\mathcal P^*\bar E)_i.
$$

$H$-derivative:

$$
D_H\operatorname{RHS}_{\mathrm c}(f,H)[\delta H]
=\delta H\,D_vf,
\qquad
\left[D_H\operatorname{RHS}_{\mathrm c}(f,H)\right]^*[B]=\bar E.
$$

**RMK.** $\mathcal P^*$ is the adjoint of the Poisson solver:

$$
\langle\bar E,\mathcal P\,\delta\rho_f\rangle
=\langle\mathcal P^*\bar E,\delta\rho_f\rangle.
$$

For the FFT solver used here, $m_\ell=-i/k_\ell$ at retained nonzero wave numbers. The adjoint conjugates this multiplier: $\overline{m_\ell}=-m_\ell$, hence $\mathcal P^*=-\mathcal P$. The zero and Nyquist modes remain zero.

### RHS* (upwind)

First-order upwind, written as central differences plus diffusion.

$$
\begin{aligned}
F_{\mathrm{up}}=\operatorname{RHS}_{\mathrm{up}}(f,H)
&=-vD_xf+(E+H)D_vf\\
&\quad+\frac{\Delta x}{2}|v|D_{xx}f
+\frac{\Delta v}{2}|E+H|D_{vv}f.
\end{aligned}
$$

$$
\begin{aligned}
(D_{xx}f)_{ij}
&=\frac{f_{i+1,j}-2f_{ij}+f_{i-1,j}}{\Delta x^2},\\
(D_{vv}f)_{ij}
&=\frac{f_{i,j+1}-2f_{ij}+f_{i,j-1}}{\Delta v^2}.
\end{aligned}
$$

Here we have:

$$
D_x^*=-D_x,\qquad D_v^*=-D_v,
\qquad D_{xx}^*=D_{xx},\qquad D_{vv}^*=D_{vv}.
$$

**$H$ fixed:**

$$
\delta|E+H|=\operatorname{sign}(E+H)\,\delta E.
$$

$$
\begin{aligned}
\delta F_{\mathrm{up}}
&=-vD_x\delta f+(E+H)D_v\delta f
+\delta E\,D_vf\\
&\quad+\frac{\Delta x}{2}|v|D_{xx}\delta f
+\frac{\Delta v}{2}|E+H|D_{vv}\delta f\\
&\quad+\frac{\Delta v}{2}\operatorname{sign}(E+H)\,\delta E\,D_{vv}f.
\end{aligned}
$$

Set

$$
\phi=D_vf+\frac{\Delta v}{2}\operatorname{sign}(E+H)D_{vv}f.
$$

Then

$$
\begin{aligned}
\delta F_{\mathrm{up}}
&=-vD_x\delta f+(E+H)D_v\delta f\\
&\quad+\frac{\Delta x}{2}|v|D_{xx}\delta f
+\frac{\Delta v}{2}|E+H|D_{vv}\delta f
+\delta E\,\phi.
\end{aligned}
$$

**Pair with $B$:**

$$
\begin{aligned}
\langle B,\delta F_{\mathrm{up}}\rangle_F
&=\Big\langle
vD_xB-(E+H)D_vB
+\frac{\Delta x}{2}|v|D_{xx}B
+\frac{\Delta v}{2}|E+H|D_{vv}B,
\delta f\Big\rangle_F\\
&\quad+\langle B,\delta E\,\phi\rangle_F.
\end{aligned}
$$

**Field term:**

$$
\bar E_i=\sum_j B_{ij}\phi_{ij}.
$$

With the same Poisson solver as above,

$$
\begin{aligned}
\langle B,\delta E\,\phi\rangle_F
&=\sum_i\delta E_i\sum_jB_{ij}\phi_{ij}\\
&=\langle\bar E,\delta E\rangle
=-\langle\mathcal P^*\bar E,\delta\rho_f\rangle\\
&=-\Delta v\sum_{i,j}(\mathcal P^*\bar E)_i\,\delta f_{ij}.
\end{aligned}
$$

So

$$
\begin{aligned}
\left(\left[D_f\operatorname{RHS}_{\mathrm{up}}(f,H)\right]^*[B]\right)_{ij}
&=\Big(vD_xB-(E+H)D_vB\\
&\qquad+\frac{\Delta x}{2}|v|D_{xx}B
+\frac{\Delta v}{2}|E+H|D_{vv}B\Big)_{ij}\\
&\quad-\Delta v\,(\mathcal P^*\bar E)_i.
\end{aligned}
$$

**$H$-derivative:** $f$ fixed $\Rightarrow\delta E=0$.

$$
\begin{aligned}
\delta_HF_{\mathrm{up}}
&=\delta H\,D_vf
+\frac{\Delta v}{2}\operatorname{sign}(E+H)\,\delta H\,D_{vv}f\\
&=\delta H\,\phi,\\
\langle B,\delta_HF_{\mathrm{up}}\rangle_F
&=\sum_i\delta H_i\sum_jB_{ij}\phi_{ij}
=\langle\bar E,\delta H\rangle.
\end{aligned}
$$

$$
\left[D_H\operatorname{RHS}_{\mathrm{up}}(f,H)\right]^*[B]
=\bar E.
$$

At each K/S/L stage, take $f=f_a^n$ and $B=\bar F_a^n$:

$$
\bar f_a^n
=\left[D_f\operatorname{RHS}_{\mathrm{up}}(f_a^n,H)\right]^*[\bar F_a^n],
\qquad a=0,1,2.
$$

## 4. Results and Analysis

### Config

Target and rank-one initial condition:

$$
\begin{aligned}
f^{\mathrm{eq}}(x,v)
&=\frac{e^{-(v-\bar v)^2/2}+e^{-(v+\bar v)^2/2}}{2\sqrt{2\pi}},\\
f_0(x,v)&=(1+\alpha\cos(\beta x))\,f^{\mathrm{eq}}(x,v),
\end{aligned}
\qquad
\alpha=10^{-3},\quad\beta=0.2,\quad\bar v=2.4.
$$

| Item | Setting |
| --- | --- |
| Domain | $x\in[0,10\pi)$, $v\in[-6,6)$; periodic in both directions |
| Grid | $n_x=n_v=128$; uniform cell centers; $\Delta x=10\pi/128$, $\Delta v=12/128$ |
| Time | $T=20$, $\Delta t=0.005$, $N=4000$ |
| Forward solve | KSL with explicit Euler substeps; first-order upwind RHS |
| Electric field | Periodic FFT Poisson solve; zero mean of $E$ |
| Control | First $k$ Fourier basis functions in the cosine–sine ordering of §1; independent of time |
| Initial coefficients | $a=(10^{-3},0,\ldots,0)$ |
| Optimizer | L-BFGS-B with the discrete adjoint gradient; no bounds or control regularization |
| Optimizer settings | `maxiter=200`, `gtol=1e-15`, `ftol=1e-14`, `maxcor=10`, `maxls=30` |
| Rank sweep | $r=1,\ldots,128$, fixed $k=10$ |
| Coefficient sweep | $k=1,\ldots,20$, fixed $r=5$ |

$J_{\mathrm{model}}$: low-rank objective. $J_{\mathrm{ref}}$: the same control evaluated with full-rank KSL. The separate full-rank optimization gives the baseline $J_{\mathrm{base}}\approx4.1974\times10^{-6}$.

### Rank

![Final objectives across ranks.](plots/01_rank_objectives.png)

- $r=1$: $J_{\mathrm{model}}\approx7.28\times10^{-7}$, but $J_{\mathrm{ref}}\approx2.46\times10^{-2}$. A small model objective can hide poor reference performance.
- $r=5$: $J_{\mathrm{ref}}\approx4.20\times10^{-6}$, close to the baseline. Increasing $r$ does not consistently improve the result.

![Optimization stops and gradients across ranks.](plots/02_optimization_status_all.png)

At low ranks, increasing the rank often improves accuracy. This level of accuracy is maintained over a certain range, as confirmed by comparison with the full-rank results. However, numerical instability appears at some larger ranks ($r>50$). This may be related to the inverse triangular factors involved in the QR derivatives.

### Number of control coefficients

![Final objectives versus the number of control coefficients at rank 5.](plots/04_k_objectives.png)

Most of the improvement occurs by $k=4$. From $k=4$ to $k=20$, $J_{\mathrm{ref}}$ decreases by only about $0.11\%$, while the accepted iteration count increases from 9 to 41.
