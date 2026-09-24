"""A compact, dependency-free (NumPy) deep-Q agent for anti-jamming channel
access -- the learned baseline reproduced from Zhang, Wu & Hu (2025).

The agent senses the recent per-channel occupancy (a sliding window of the
spectrum-occupancy dataset) and predicts a clean channel.  Reward is 1 if the
chosen channel is clean, else 0 (one-step / coarse-grained spectrum
prediction, gamma = 0), which keeps training stable without a GPU.  The output
layer has one unit per channel, so the parameter count and per-step cost grow
as O(M) -- exactly the scaling the thesis probes.
"""
from __future__ import annotations
import time
import numpy as np
from .. import config


class DQNAgent:
    def __init__(self, M, window=config.WINDOW, hidden=64, lr=1e-3,
                 capacity=2000, batch=32, seed=0):
        self.M, self.W, self.H = M, window, hidden
        self.in_dim = window * M
        self.batch = batch
        g = np.random.default_rng(seed)
        self.W1 = (g.standard_normal((self.in_dim, hidden)) * np.sqrt(2 / self.in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (g.standard_normal((hidden, M)) * np.sqrt(2 / hidden)).astype(np.float32)
        self.b2 = np.zeros(M, np.float32)
        self._m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._t = 0
        self.lr = lr
        self.buf_s = np.zeros((capacity, self.in_dim), np.uint8)
        self.buf_a = np.zeros(capacity, np.int32)
        self.buf_r = np.zeros(capacity, np.float32)
        self.cap, self.size, self.ptr = capacity, 0, 0
        self.rng = g
        self.params = int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        q = a1 @ self.W2 + self.b2
        return z1, a1, q

    def q(self, x):
        return self._forward(x[None].astype(np.float32))[2][0]

    def act(self, state, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.M))
        return int(np.argmax(self.q(state)))

    def store(self, s, a, r):
        i = self.ptr
        self.buf_s[i] = s; self.buf_a[i] = a; self.buf_r[i] = r
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def _adam(self, grads):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, gd) in enumerate(zip((self.W1, self.b1, self.W2, self.b2), grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * gd
            self._v[i] = b2 * self._v[i] + (1 - b2) * gd * gd
            mhat = self._m[i] / (1 - b1 ** self._t)
            vhat = self._v[i] / (1 - b2 ** self._t)
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def train_step(self):
        if self.size < self.batch:
            return
        idx = self.rng.integers(0, self.size, self.batch)
        X = self.buf_s[idx].astype(np.float32)
        a = self.buf_a[idx]; r = self.buf_r[idx]
        z1, a1, q = self._forward(X)
        dq = np.zeros_like(q)
        rows = np.arange(self.batch)
        dq[rows, a] = (q[rows, a] - r) / self.batch          # MSE on chosen action
        gW2 = a1.T @ dq
        gb2 = dq.sum(0)
        da1 = dq @ self.W2.T
        da1[z1 <= 0] = 0.0
        gW1 = X.T @ da1
        gb1 = da1.sum(0)
        self._adam((gW1, gb1, gW2, gb2))


def train(jammer_factory, M, slots=config.SLOTS_PER_EPISODE, max_steps=15000,
          time_budget=None, seed=0, eval_every=2000):
    """Train the DQN against a jammer; return metrics incl. wall-clock cost.

    ``jammer_factory(seed)`` builds a fresh jammer.  Training is capped by
    ``max_steps`` and/or ``time_budget`` seconds (time-boxing large M)."""
    agent = DQNAgent(M, seed=seed)
    jam = jammer_factory(seed)
    t0 = time.perf_counter()
    step = 0
    eps = 1.0
    curve = []
    converged_step = None
    while step < max_steps:
        if hasattr(jam, "reset"):
            jam.reset()
        window = np.zeros((agent.W, M), np.uint8)
        history = []
        for t in range(slots):
            state = window.ravel()
            a = agent.act(state, eps)
            jammed = jam.step(t, history)
            r = 0.0 if jammed[a] else 1.0
            agent.store(state.copy(), a, r)
            agent.train_step()
            history.append(a)
            window = np.roll(window, -1, axis=0)
            window[-1] = jammed.astype(np.uint8)
            eps = max(0.05, eps * 0.9995)
            step += 1
            if step % eval_every == 0:
                av = evaluate(agent, jammer_factory(seed + 777), M, slots)
                curve.append((step, av, time.perf_counter() - t0))
                if converged_step is None and av >= 0.90:
                    converged_step = step
            if time_budget and (time.perf_counter() - t0) > time_budget:
                break
            if step >= max_steps:
                break
        if time_budget and (time.perf_counter() - t0) > time_budget:
            break
    wall = time.perf_counter() - t0
    final_av = evaluate(agent, jammer_factory(seed + 777), M, slots)
    return {"M": M, "final_avoidance": final_av, "train_seconds": wall,
            "steps": step, "params": agent.params,
            "converged_step": converged_step, "curve": curve}


def evaluate(agent, jam, M, slots):
    if hasattr(jam, "reset"):
        jam.reset()
    window = np.zeros((agent.W, M), np.uint8)
    history, clean = [], 0
    for t in range(slots):
        a = int(np.argmax(agent.q(window.ravel())))
        jammed = jam.step(t, history)
        clean += 0 if jammed[a] else 1
        history.append(a)
        window = np.roll(window, -1, axis=0)
        window[-1] = jammed.astype(np.uint8)
    return clean / slots
