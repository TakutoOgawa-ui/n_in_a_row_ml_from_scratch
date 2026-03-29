#!/usr/bin/env python3

"""
nn_loss.py（損失関数）

目的
- 出力 y と教師 t から損失（スカラー）を計算する
- 逆伝播の入り口となる dL/dy を計算する

対応
- mse（平均二乗誤差）
- huber（フーバー損失）
"""

from __future__ import annotations
from typing import List

Vector = List[float]


def calc_loss(y: Vector, t: Vector, name: str = "mse", delta: float = 1.0) -> float:
    if len(y) != len(t):
        raise ValueError("y と t の長さが一致しません")

    n = len(y)

    if name == "mse":
        # 平均二乗誤差
        # L = (1/n) * Σ (y - t)^2
        s = 0.0
        for i in range(n):
            e = y[i] - t[i]
            s += e * e
        return s / n

    if name == "huber":
        # 平均フーバー損失
        # L = (1/n) * Σ huber(y - t)
        s = 0.0
        d = float(delta)
        for i in range(n):
            e = y[i] - t[i]
            ae = abs(e)
            if ae <= d:
                s += 0.5 * e * e
            else:
                s += d * (ae - 0.5 * d)
        return s / n

    raise ValueError("loss は mse / huber のどちらかです")


def calc_loss_grad(y: Vector, t: Vector, name: str = "mse", delta: float = 1.0) -> Vector:
    if len(y) != len(t):
        raise ValueError("y と t の長さが一致しません")

    n = len(y)

    if name == "mse":
        # dL/dy = (2/n) * (y - t)
        return [(2.0 / n) * (y[i] - t[i]) for i in range(n)]

    if name == "huber":
        # dL/dy = (1/n) * huber'(y - t)
        d = float(delta)
        g: Vector = []
        for i in range(n):
            e = y[i] - t[i]
            if abs(e) <= d:
                g.append((1.0 / n) * e)
            else:
                g.append((1.0 / n) * (d if e > 0.0 else -d))
        return g

    raise ValueError("loss は mse / huber のどちらかです")

