#!/usr/bin/env python3


"""
nn_activation.py（順伝播用 活性化関数）
"""

from __future__ import annotations

from typing import List
import math

#ベクトル
Vector = List[float]


def _limit_range(x: float, low: float, high: float) -> float:
    """
    数値のはみ出しを防ぐ（sigmoid の exp オーバーフロー対策用）
    下限値と上限値から外れた値は、いずれかに変更する
    """
    if x < low:
        return low
    if x > high:
        return high
    return x


def sigmoid(x: Vector) -> Vector:
    """
    sigmoid: 1 / (1 + exp(-x))
    xの要素vに対してsigmoidの計算をして新しいリストを返す
    """
    result: Vector = []
    for v in x:
        v2 = _limit_range(v, -50.0, 50.0)
        result.append(1.0 / (1.0 + math.exp(-v2)))
    return result


def relu(x: Vector) -> Vector:
    """
    ReLU: max(0, x)
    x の各要素vが0以下なら0にし、正ならそのままにする
    """
    result: Vector = []
    for v in x:
        result.append(v if v > 0.0 else 0.0)
    return result


def pass_through(x: Vector) -> Vector:
    """
    pass_through: そのまま返す
    floatはイミュータブルなのでshallow copyとして中間値を保存
    """
    return x[:]


def apply_activation(x: Vector, name: str) -> Vector:
    """
    文字列で指定された活性化関数を適用する。
    対応していない名前が呼び出されたらエラーを返す。
    """
    key = name
    if key == "sigmoid":
        return sigmoid(x)
    if key == "relu":
        return relu(x)
    if key == "pass_through":
        return pass_through(x)
    raise ValueError(f"このような関数名は存在しません: {name}; relu or sigmoid or pass_through ")

def calc_activation_slope(u: Vector, name: str) -> Vector:
    """活性化関数の傾き（u に対する微分）を返す。"""
    key = name
    if key == "pass_through":
        return [1.0 for _ in u]

    if key == "relu":
        return [1.0 if v > 0.0 else 0.0 for v in u]

    if key == "sigmoid":
        s = sigmoid(u)
        return [sv * (1.0 - sv) for sv in s]

    raise ValueError(f"このような関数名は存在しません：{name}; relu or sigmoid or pass_through")




