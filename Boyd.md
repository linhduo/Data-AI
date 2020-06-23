---
layout: default
title: Sparse Covariance Problem
---

## Sparse inverse covariance matrix problem

__Problem:__ Assume that $S$ is the empirical covariance matrix of the normal distribution $N(0,\Sigma)$, where $\Sigma^{-1}$ is sparse. We attempt to reconstruction $\Sigma$ (or equivalently $\Sigma^{-1}$.

The paper (Boyd et al) suggested to solve the optimization problem
$$\label{E:sparse} \tag{1} \arg\min_{X \in S_+} Tr(SX) - \log \det(X) + \lambda \|X\|_1,$$ to find $X= \Sigma^{-1}$.
Here, $S_+$ be the set all nonnegative definite symmetric matrices. 

The statistical derivation of (\ref{E:sparse}) will be considered later. We instead look at some intuition. Namely, we first prove:

__Lemma 1__ Assume that $S \in S_+$ is invertible. Then, the non-regularized problem
$$ \label{E:non-reg} \tag{1b} \arg\min_{X \in S_+} Tr(SX) - \log \det(X),$$ has a unique solution $X= S^{-1}$. 

__Proof__ 
We note that 
$$\log \det(X) = \log \det(SX)- \log \det(S).$$
Therefore, the (\ref{E:non-reg}) is equivalent to
$$\arg\min_{X \in S_+} Tr(SX) - \log \det(SX).$$
It now suffices to prove that the problem
$$\arg\min_{X \in S_+} Tr(X) - \log \det(X),$$ has a unique solution $X= I$. Indeed, let $\lambda_1,\dots, \lambda_n \geq 0$ be the eigenvalues of $X$. Then, 
$$Tr(X) - \log \det(X)= \sum_{i} \lambda_i - \log \lambda_i.$$
If is simple to see that $X \in \arg\min_{X \in S_+} Tr(X) - \log \det(X)$ if and only if $\lambda_1 = \dots =\lambda_n =1$. That is $X= I$. $\blacksquare$

The above lemma (at least) tells us that $X$ is a sparse approximate inverse of $S$, which is intuitively $\Sigma^{-1}$. 

Now, to solve (\ref{E:sparse}), we employ ADMM method, which reads as

\begin{eqnarray}\label{E:ADMM1} \tag{2a}
X^{n+1} &=& \arg \min_{X \in S_+} Tr(SX) - \log \det(X) + \rho/2 \|X- Z^k + U^k\|^2, \\
\label{E:ADMM2} \tag{2b}
Z^{k+1} &=& \arg \min_{Z \in S_+} \lambda \|Z\|_1 + \rho/2 \|X^{k+1}-Z +U^k\|^2 = S_{\lambda/\rho}(X^{k+1} + U^{k}),\\
\label{E:ADMM3}\tag{2c}
U^{k+1} &=& U^k + X^{k+1}-Z^{k+1}. \end{eqnarray}

The only complicated problem is (\ref{E:ADMM1}). In order to solve it, we will make use of the following results

__Lemma 2__ We have

- (i) $\nabla_X Tr{SX} = S^T = S$ (since $S$ is symmetric),

- (ii) $\nabla_X \log \det(X) = X^{-1}$.

__Proof__

(1) The proof follows easily from the formula $Tr(SX)= \sum_i S_{ij} X_{ji}$.

(2) First we compute $\nabla_X \det(X)$. In order to do so, fix an index pair $(i,j)$, we consider $X_t = X + t M_{ij}$ where $M_{ij}$ has value $1$ at $(i,j)$ and zero everywhere else. That is, 
$$X_t = (X_1,\dots, X_{j-1}, X_j + t e_i, X_{j+1}, \dots, X_n),$$ where $X_1,\dots,X_n$ are columns of $X$ and $e_i=(0,\dots,1,\dots,0)$ is the $i^{th}$ unit vector. Therefore, noting that $\det$ is linear in each column, 
$$\det X_t = \det X + t \det (X_1,\dots, X_{j-1}, e_i, X_{j+1}, \dots, X_n) = \det(X) + t (-1)^{i+j} \det (X_{ij}).$$
Here, as standard notations, $X_{ij}$ is the submatrix of $X$, obtained by deleting the $i^{th}$-row and $j^{th}$-column. Therefore,
$$\partial_{x_{ij}} \det X = (\det X_t)' = (-1)^{ij} \det (X_{ij}),$$
which implies
$$\partial_{x_{ij}} \log \det(X) = (-1)^{ij} \frac{\det (X_{ij})}{\det X}.$$
From the Crammer's formula
$$\nabla_X \log \det(X) = X^{-1}.$$ $\blacksquare$


We are now ready to compute a solution to (\ref{E:ADMM1}). By taking the derivative of the objective function, we obtain that $X^{k+1}$ solve the equation
$$S - X^{-1} + \rho (X - Z^{k} + U^k) = 0,$$
or 
$$ - X^{-1} + \rho X = \rho (Z^{k} - U^k) -S.$$
Let us diagonalize $\rho (Z^{k} - U^k) -S = Q \Lambda Q^T$, where $\Lambda = diag(\lambda_1,\dots, \lambda_n)$. We solve the above equation in the form $X = Q D Q^T$, where $D= diag (d_1,\dots,d_n)$. Then, $d_i$ satisfies the equation $-d_i^{-1} + \rho d_i = \lambda_i,$
or $$\rho d_i^2 - \lambda_i d_i - 1 =0.$$ That is,
 $$\boxed{d_i = \frac{\lambda_i + \sqrt{\lambda_i^2+4 \rho}}{2 \rho}.}$$

