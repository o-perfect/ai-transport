"""Stage 2: a small Gaussian HMM with diagonal covariance, in plain numpy.

Diagonal covariance keeps each week's state explainable as a sum of one
evidence term per feature plus a memory term (see `explain`).
"""

import numpy as np

# Named after what the fitted model learned (see `python -m strait_risk`):
# Reduced ~ 60% of normal traffic with tankers holding up better (Red Sea since
# 2024); Closed ~ 6% of normal with every ship type gone (Hormuz since Mar 2026).
STATES = ["Normal", "Reduced", "Closed"]


class DiagGaussianHMM:
    def __init__(self, n_states=3, sticky=20.0, n_iter=300, tol=1e-6, var_floor=1e-3):
        self.K, self.sticky, self.n_iter, self.tol, self.var_floor = n_states, sticky, n_iter, tol, var_floor

    # --- pieces -----------------------------------------------------------------
    def log_emission_parts(self, X):
        """(T, K, F): log N(x_tf | mu_kf, var_kf), one term per feature."""
        d = X[:, None, :] - self.means[None]
        return -0.5 * (d ** 2 / self.vars[None] + np.log(2 * np.pi * self.vars[None]))

    def _forward_backward(self, X):
        logB = self.log_emission_parts(X).sum(-1)
        off = logB.max(1, keepdims=True)
        B = np.exp(logB - off)
        T, K = B.shape
        alpha, beta, s = np.zeros((T, K)), np.ones((T, K)), np.zeros(T)
        alpha[0] = self.start * B[0]
        s[0] = alpha[0].sum()
        alpha[0] /= s[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            s[t] = alpha[t].sum()
            alpha[t] /= s[t]
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1]) / s[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = np.zeros((K, K))
        for t in range(T - 1):
            xi += alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None] / s[t + 1]
        return gamma, xi, np.log(s).sum() + off.sum()

    # --- fitting ----------------------------------------------------------------
    def fit(self, seqs, init_labels):
        """EM over several independent sequences (one per strait), seeded from
        Stage 1 labels so the states start out meaning Normal/Stressed/Disrupted."""
        X, L = np.vstack(seqs), np.concatenate(init_labels)
        self.means = np.array([X[L == k].mean(0) for k in range(self.K)])
        self.vars = np.array([X[L == k].var(0) for k in range(self.K)]) + self.var_floor
        self.start = np.full(self.K, 1 / self.K)
        prior = 1 + self.sticky * np.eye(self.K)  # Dirichlet pseudo-counts favouring persistence
        counts = prior.copy()
        for l in init_labels:
            np.add.at(counts, (l[:-1], l[1:]), 1)
        self.A = counts / counts.sum(1, keepdims=True)
        last = -np.inf
        for it in range(self.n_iter):
            g_all, xi_all, start, ll = [], np.zeros((self.K, self.K)), np.ones(self.K), 0
            for x in seqs:
                g, xi, l = self._forward_backward(x)
                g_all.append(g)
                xi_all += xi
                start += g[0]
                ll += l
            G = np.vstack(g_all)
            w = G.sum(0)
            self.means = (G.T @ X) / w[:, None]
            self.vars = np.array([(G[:, k:k + 1] * (X - self.means[k]) ** 2).sum(0) / w[k]
                                  for k in range(self.K)]) + self.var_floor
            self.A = (xi_all + prior - 1 + 1e-9)
            self.A /= self.A.sum(1, keepdims=True)
            self.start = start / start.sum()
            if ll - last < self.tol:
                break
            last = ll
        self.loglik, self.iterations = ll, it + 1
        # Name states by traffic level (feature 0): highest = Normal, lowest = Closed.
        order = np.argsort(-self.means[:, 0])
        self.means, self.vars = self.means[order], self.vars[order]
        self.A, self.start = self.A[np.ix_(order, order)], self.start[order]
        return self

    def n_params(self):
        F = self.means.shape[1]
        return self.K * 2 * F + self.K * (self.K - 1) + (self.K - 1)

    # --- inference --------------------------------------------------------------
    def filter(self, X):
        """Filtered P(state_t | x_1..x_t) (no look-ahead) and the one-step prior
        P(state_t | x_1..x_{t-1}) carried over from last week."""
        logB = self.log_emission_parts(X).sum(-1)
        B = np.exp(logB - logB.max(1, keepdims=True))
        T = len(X)
        post, prior = np.zeros((T, self.K)), np.zeros((T, self.K))
        p = self.start
        for t in range(T):
            prior[t] = p
            a = p * B[t]
            post[t] = a / a.sum()
            p = post[t] @ self.A
        return post, prior

    def explain(self, x, prior, a="Closed", b="Normal"):
        """Log-odds of state a vs b this week = memory term + one term per feature.
        Exact for diagonal covariance."""
        i, j = STATES.index(a), STATES.index(b)
        parts = self.log_emission_parts(x[None])[0]
        memory = np.log(prior[i] + 1e-300) - np.log(prior[j] + 1e-300)
        return memory, parts[i] - parts[j]

    def durations(self):
        return 1 / (1 - np.diag(self.A))
