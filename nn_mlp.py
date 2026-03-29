#!/usr/bin/env python3

"""
nn_mlp.py（ニューラルネットワーク）

前提
- 入力は「ベクトル（list[float]）」を 1 サンプルだけ扱う。
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import sys

# 活性化関数を適用する関数（順伝播）と、傾きを返す関数（逆伝播）
from .nn_activation import apply_activation, calc_activation_slope
# 一層分を計算するクラス
from .nn_layer import WeightBiasLayer

from .nn_matrix import element_mul, check_finite_vector
from .nn_loss import calc_loss, calc_loss_grad


Vector = List[float]
Matrix = List[List[float]]


class NeuralNetwork:
    """
    線形層 + 活性化 を積み重ねたネットワーク
    """

    def __init__(
        self,
        sizes: List[int],
        hidden_activation: str = "relu",
        output_activation: str = "pass_through",
        seed: int | None = None,
    ) -> None:
        # sizesは各層の次元の並び
        if len(sizes) < 2:
            raise ValueError("サイズは2要素以上でなければならない.")

        # 念の為コピー
        self.sizes = sizes[:]

        # 線形層(u = xW + b)を順番に入れていくリスト
        self.layers: List[WeightBiasLayer] = []

        # 各層に対応する活性化関数のリスト
        self.activations: List[str] = []

        # 層の数はsizeの区切り数で、len(sizes)-1
        num_layers = len(sizes) - 1

        for i in range(num_layers):
            # i層目の入力次元と出力次元
            in_diment = sizes[i]
            out_diment = sizes[i + 1]

            # 乱数生成の再現性
            # i層目はseed+1のようにずらして層毎に違う乱数を用意する
            # 層の乱数を固定する番号を渡す
            layer_seed = None if seed is None else (seed + i)

            # ここは代入（=）で渡す（== だと True/False になって壊れる）
            self.layers.append(
                WeightBiasLayer(in_diment=in_diment, out_diment=out_diment, weight_seed=layer_seed)
            )

            if i == num_layers - 1:
                self.activations.append(output_activation)
            else:
                self.activations.append(hidden_activation)

        # 逆伝播で使う保存（順伝播の途中結果）
        self._last_us: List[Vector] = []
        self._last_ys: List[Vector] = []

    def forward(self, x: Vector) -> Vector:
        """
        順伝播：x -> layer -> activation -> ...
        層を通るごとに入力xを更新する. x->hへ
        """
        # 逆伝播用に、毎回リセットして保存し直す
        self._last_us = []
        self._last_ys = []

        h = x
        for layer, act_name in zip(self.layers, self.activations):
            # 1)線形層で u=hW+b を計算
            u = layer.forward(h)
            # 2)活性化関数を適用して次の層に渡す
            h = apply_activation(u, act_name)

            # 逆伝播で必要なので保存（u と、活性化後h）
            self._last_us.append(u)
            self._last_ys.append(h)

        return h
    

    def train_one(
        self,
        x: Vector,
        t: Vector,
        lr: float,
        loss_name: str = "mse",
        huber_delta: float = 1.0,
        optimizer: str = "sgd",
        rho: float = 0.9,
        eps: float = 1e-8,
        check_finite: bool = True,
    ) -> float:
        """
        1サンプルで学習（順伝播→損失→逆伝播→更新）
        返り値：損失
        """
        # 1) 順伝播して出力を得る
        y = self.forward(x)

        # 2) 損失（誤差）を計算する
        L = calc_loss(y, t, name=loss_name, delta=huber_delta)

        # 3) 逆伝播の入り口：dL/dy（出力yに対する勾配）
        dh = calc_loss_grad(y, t, name=loss_name, delta=huber_delta)

        if check_finite:
            check_finite_vector(dh, "dL/dy")

        # 4) 後ろの層から順に、勾配を戻していく
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            act_name = self.activations[i]
            u = self._last_us[i]

            # 4-1) 活性化の手前へ戻す：du = dh * f'(u)
            slope = calc_activation_slope(u, act_name)
            du = element_mul(dh, slope)

            # 4-2) 線形層の逆伝播：dx, dW, db を作る
            dx, dW, db = layer.backward(du)

            # 4-3) 重みとバイアスを更新する（sgd / rmsprop）
            layer.update(
                dW=dW,
                db=db,
                lr=lr,
                mode=optimizer,
                rho=rho,
                eps=eps,
                check_finite=check_finite,
            )

            # 4-4) さらに前の層へ勾配を渡す
            dh = dx

            if check_finite:
                check_finite_vector(dh, "backward dh")

        return L

    @classmethod
    def from_pretrained_json(cls, path: str) -> "NeuralNetwork":
        """JSON（params.json）から学習済みパラメータを読み込んでネットワークを作る。"""
        # pathのファイルを開いて、Jsonをpythonの辞書に変換してdataを入れる
        with open(path, "r") as f:
            data: Dict[str, Any] = json.load(f)

        # layersが存在するか、リストか、からじゃないかを確認
        layers = data.get("layers")
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError("Invalid JSON: layers not found.")

        # 組み立て用の箱を用意
        sizes: List[int] = []
        Matrix_layers: List[WeightBiasLayer] = []
        activations: List[str] = []

        # 各層を読み込むループ
        for idx, layer in enumerate(layers):
            # W,bを取り出す
            W = layer.get("W")
            b = layer.get("b")
            # activationがなければ、pass_throughにしておく
            act = layer.get("activation", "pass_through")

            # Wとbがリストか
            if not isinstance(W, list) or not isinstance(b, list):
                raise ValueError("Invalid layer format.")
            # W は list[list[float]] を想定しているため、行列であるかを確認
            if len(W) == 0 or not isinstance(W[0], list):
                raise ValueError("Invalid W format.")

            # sizes を組み立て（最初だけ input_diment を入れる）
            in_diment = len(W)
            out_diment = len(W[0])
            if idx == 0:
                sizes.append(in_diment)
            sizes.append(out_diment)

            # JsonのW,bをそのまま使う層を作る
            # Matrix_layersに入れる
            Matrix_layers.append(WeightBiasLayer.from_params(W=W, b=b))

            # activationを層ごとに保存する
            activations.append(str(act))

        # 仮にclsで箱を作る
        net = cls(sizes=sizes)

        # 箱の中身をJson由来のものに
        net.layers = Matrix_layers
        net.activations = activations
        net.sizes = sizes
        return net

"""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 nn_mlp.py params.json")
        sys.exit(1)

    params_path = sys.argv[1]
    net = NeuralNetwork.from_pretrained_json(params_path)

    x = [2.0, 1.0, 3.0]
    y = net.forward(x)  

    print("x:", x)
    print("y:", y)
"""


    