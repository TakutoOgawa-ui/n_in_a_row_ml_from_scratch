#!/usr/bin/env python3


"""
nn_matrix.py（ベクトル・行列演算モジュール）

目的
- 線形層の u = xW + b を計算するための部品を用意する。
- 逆伝播で使う重み勾配や前に戻す勾配の計算を行う

前提
- ベクトル：list[float]
- 行列：list[list[float]]
- 入力は「1サンプル（1ベクトル）」
- 次元が合わない場合は ValueError を表示する

使用する関数
1) mat_shape(W)         : 行列の形（行数, 列数）
2) vec_mat_mul(x, W)    : y = xW
3) vec_add(a, b)        : ベクトル加算
4) weight_layer(x, W, b)      : u = xW + b（線形層の計算）
"""

from __future__ import annotations

from typing import List, Tuple
import math

Vector = List[float]
Matrix = List[List[float]]


def mat_shape(W: Matrix) -> Tuple[int, int]:
    """
    行列 W の形（行数, 列数）を返す。
    行の長さが不揃いならエラーにする。
    """
    if len(W) == 0:
        return (0, 0)

    #各列の長さ測定
    cols = len(W[0])
    for row in W:
        if len(row) != cols:
            raise ValueError("Matrix rows must have the same length.")
    return (len(W), cols)


def vec_mat_mul(x: Vector, W: Matrix) -> Vector:
    """
    y = xW を計算する。

    形 ベクトルは長さ、行列はn行m列を表す。
    - x : (n,)
    - W : (n, m)
    - y : (m,)

    ****実装****
    - y[j] = sum_i x[i] * W[i][j]
    """
    n = len(x)
    rows, cols = mat_shape(W)
    
    # 行列計算の定義より、ベクトルの長さと行列の列数は一致する必要がある
    if rows != n:
        raise ValueError("xとWの次元が合いません")
    
    #0.0で埋めたリストを上書きしていく
    y: Vector = [0.0] * cols
    for j in range(cols):
        s = 0.0
        for i in range(n):
            s += x[i] * W[i][j]
        y[j] = s
    return y


def vec_add(a: Vector, b: Vector) -> Vector:
    """同じ長さのベクトル同士を加算する。"""
    if len(a) != len(b):
        raise ValueError("次元が合いません(vec_add)")

    c: Vector = []# c は結果を入れるリスト
    for ai, bi in zip(a, b):#zip関数を用いてリストを束ねて処理
        c.append(ai + bi)
    return c


def apply_weight_and_bias(x: Vector, W: Matrix, b: Vector) -> Vector:
    """
    線形層の計算：u = xW + b
    """
    y = vec_mat_mul(x, W)
    if len(y) != len(b):
        raise ValueError("バイアスの長さが異なるため計算不能")
    return vec_add(y, b)

def element_mul(a: Vector, b: Vector) -> Vector:
    """
    要素ごとの掛け算(同じ長さで)
    """
    if len(a) != len(b):
        raise ValueError("次元が合いません(element_mul)")
    c: Vector = []
    for ai, bi in zip(a, b):
        c.append(ai * bi)
    return c

def calc_weight_grad(x: Vector, du: Vector) -> Matrix:
    """
    損失を減らすための重みの勾配(grad)を作る(dW[i][j] = x[i] * du[j])
    """
    dW: Matrix = []
    for xi in x:
        row: Vector = []
        for duj in du:
            row.append(xi * duj)
        dW.append(row)
    return dW

def calc_prev_grad(du:Vector, W:Matrix) -> Vector:
    """前の層へ戻す勾配dx = du * W^T"""
    rows, cols = mat_shape(W)
    if cols != len(du):
        raise ValueError("次元が合いません(calc_prev_grad)")
    
    dx: Vector = [0.0 for _ in range(rows)]
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += du[j] * W[i][j]
        dx[i] = s
    return dx

def check_finite_vector(x: Vector, name: str) -> None:
    """NaN/Inf があれば例外。"""
    for v in x:
        if not math.isfinite(v):
            raise ValueError(f"{name} に NaN/Inf が含まれています")


def check_finite_matrix(W: Matrix, name: str) -> None:
    """NaN/Inf があれば例外。"""
    for row in W:
        check_finite_vector(row, name)

