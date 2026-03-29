#!/usr/bin/env python3

"""
nn_layer.py

目的
- 1つの層で「線形変換」を行う。
  u = xW + b
- 逆伝播でdW,db,dxを計算する

前提
- x はベクトル（list[float]）
- W は行列（list[list[float]]）
- b はベクトル（list[float]）
"""

from __future__ import annotations

from typing import List, Optional
import random

from .nn_matrix import mat_shape, apply_weight_and_bias, calc_prev_grad, calc_weight_grad,check_finite_matrix,check_finite_vector


Vector = List[float]
Matrix = List[List[float]]


class WeightBiasLayer:
    """
    u = xW + b を計算する層（順伝播用）
    手計算用にW,bを省略した時に乱数で作れるように実装する
    """

    def __init__(
    self,
    in_diment: int, #入力の次元
    out_diment: int, #出力の次元
    W: Matrix | None = None,
    b: Vector | None = None,
    weight_scale: float = 0.1, #self.W = Noneの時に重みを作る際の大きさの範囲
    weight_seed: int | None = None, #scaleを用いる時、毎回同じ乱数列にして再現性を保つ
) -> None:
        self.in_diment = int(in_diment)
        self.out_diment = int(out_diment)

        # パラメータ（未指定なら乱数で用意）
        self.W = W if W is not None else self._make_random_W(self.in_diment, self.out_diment, weight_scale, weight_seed)
        self.b = b if b is not None else [0.0 for _ in range(self.out_diment)]

        # 修正容易性の観点からWとbの形が合っているか確認。
        self._check_params()
  
        #順伝播で使う(直前の入力)
        self._last_x:Vector | None = None

        # rmspropで使う
        self._nu_W:Matrix = [[0.0 for _ in range(self.out_diment)] for _ in range(self.in_diment)]
        self._nu_b: Vector = [0.0 for _ in range(self.out_diment)]
    
    # self.W = Noneの時に重み行列Wを乱数で作る。
    def _make_random_W(self, in_diment: int, out_diment: int, weight_scale: float, weight_seed: int | None = None) -> Matrix:
        """
        weight_scale は重みの大きさの範囲。
        再現性のために weight_seed を固定し、毎回同じ乱数列にする。 
        """
        # 乱数生成機generater
        generator = random.Random(weight_seed)

        # Wを「行ごと」に作る
        # まず(in_diment)行を作り、各行に(out_dim)個の要素を入れる。
        W: Matrix = []
        for _ in range(in_diment):
            row: Vector = []
            for _ in range(out_diment):
                row.append(generator.uniform(-weight_scale, weight_scale)) #乱数機械から乱数を出す
            W.append(row)
        return W

    # Wとbの形のチェック
    def _check_params(self) -> None:
        rows, cols = mat_shape(self.W)

        if rows != self.in_diment or cols != self.out_diment:
            raise ValueError("Wの形が異なっています.")
        if len(self.b) != self.out_diment:
            raise ValueError("bの形状が異なっています.") 


    @classmethod
    # params.jsonに保存されているW,bを読み込む。
    def from_params(cls, W: Matrix, b: Vector) -> "WeightBiasLayer":
        """学習済みパラメータ（W, b）から層を作る。"""
        in_diment, out_diment = mat_shape(W)
        return cls(in_diment=in_diment, out_diment=out_diment, W=W, b=b)

    def forward(self, x: Vector) -> Vector:
        """
        順伝播を行う。
        u = xW + bを計算として返す。
        計算は matrix ファイルのapply_weight_and_bias 関数に任せる
        """
        if len(x) != self.in_diment:
            raise ValueError("入力の長さが異なっています.")
        
        # 順伝播で dW を作るために入力を保存
        self._last_x = x[:]
        return apply_weight_and_bias(x, self.W, self.b)
    
    def backward(self, du: Vector) -> tuple[Vector, Matrix, Vector]:
        """
        逆伝播（線形層）
        入力：du = dL/du
        出力：dx, dW, db
        """
        if self._last_x is None:
            raise ValueError("backward の前に forward を呼んでください")
        if len(du) != self.out_diment:
            raise ValueError("du の長さが層の出力次元と一致しません")

        x = self._last_x
        dW = calc_weight_grad(x, du)
        db = du[:]          # b の勾配はそのまま
        dx = calc_prev_grad(du, self.W)

        return dx, dW, db

    def update(
        self,
        dW: Matrix,
        db: Vector,
        lr: float,
        mode: str = "sgd",
        rho: float = 0.9,
        eps: float = 1e-8,
        check_finite: bool = True,
    ) -> None:
        """W,b を更新する（SGD / RMSProp）。"""
        if check_finite:
            check_finite_matrix(dW, "dW")
            check_finite_vector(db, "db")

        if mode == "sgd":
            for i in range(self.in_diment):
                for j in range(self.out_diment):
                    self.W[i][j] -= lr * dW[i][j]
            for j in range(self.out_diment):
                self.b[j] -= lr * db[j]

        elif mode == "rmsprop":
            for i in range(self.in_diment):
                for j in range(self.out_diment):
                    g = dW[i][j]
                    self._nu_W[i][j] = rho * self._nu_W[i][j] + (1.0 - rho) * (g * g)
                    self.W[i][j] -= lr * g / ((self._nu_W[i][j] + eps) ** 0.5)

            for j in range(self.out_diment):
                g = db[j]
                self._nu_b[j] = rho * self._nu_b[j] + (1.0 - rho) * (g * g)
                self.b[j] -= lr * g / ((self._nu_b[j] + eps) ** 0.5)

        else:
            raise ValueError("mode は sgd / rmsprop のどちらかです")

        if check_finite:
            check_finite_matrix(self.W, "W")
            check_finite_vector(self.b, "b")
