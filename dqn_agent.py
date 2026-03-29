#!/usr/bin/env python3

"""
dqn_agent.py

TicTacToe用のDQNエージェント

最終課題
状態を入力として、各行動のQ値をまとめて出力する全結合NNを持つ
ベルマン方程式で行動を選ぶ
目標値 y = r + γ * max_a' Q_target(s', a')を用いて、onlineネットワークを更新
デルタクリッピング(y - Q)を[-clip, clip]に制限する
ターゲットネットワークを一定間隔で更新

注意
DQNでは「選択された行動 a に対応するQ(s,a)のみ」を学習対象にする
出力ベクトルを全部正解に合わせるわけではない。
"""

from __future__ import annotations 

import copy
import random
from typing import List, Sequence, Tuple

from .agent_base import TicTacToeAgent
from env.tictactoe import Cell, Player
from nn.nn_mlp import NeuralNetwork

Vector = List[float]

def encode_state_from_player_view(game, my_player: Player) -> Vector:
    """
    盤面を-1,0,1に変換
    my_player: 1(先手) or 2(後手)
    """
    n = game._size
    my_cell = Cell.FIRST if my_player == Player.FIRST else Cell.SECOND
    opp_cell = Cell.SECOND if my_player == Player.FIRST else Cell.FIRST

    output: Vector = []
    for r in range(n):
        for c in range(n):
            cell = game._board[r][c]
            if cell == Cell.EMPTY:
                output.append(0.0)
            elif cell == my_cell:
                output.append(1.0)
            elif cell == opp_cell:
                output.append(-1.0)
            else:
                output.append(0.0)
    return output

def pick_best_action_index(values: Sequence[float], legal_indices: Sequence[int]) -> int:
    """合法手の中でQ値が最大の行動indexを返す"""
    best_index = legal_indices[0]
    best_value = values[best_index]

    for idx in legal_indices[1:]:
        value = values[idx]
        if value > best_value:
            best_value = value
            best_index = idx
    return best_index


def _all_legal_action_indices(game) -> List[int]:
    """空いているマスの一覧を返す、indexは0~n^2-1"""
    n = game._size
    legal: List[int] = []
    for r in range(n):
        for c in range(n):
            if game._board[r][c] == Cell.EMPTY:
                legal.append(r * n + c)
    return legal

def _index_to_row_col(index: int, size: int) -> Tuple[int, int]:
    """番号のindexを盤面の(行,列)に変換するための関数"""
    return index// size, index % size
    


# DQN agent

class DQNAgent(TicTacToeAgent):
    """TicTacToeのDQNエージェント"""

    def __init__(
            self,
            n: int = 3,
            name: str = "DQN",
            hidden_sizes: Sequence[int] = (64,64),
            gamma: float = 0.95,
            learning_rate: float = 1e-3,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.1,
            epsilon_reduce_count: int = 10000,
            target_update_interval: int = 500,
            delta_clip: float = 1.0,
            optimizer: str = "rmsprop",
            loss_name: str = 'mse',
            huber_delta: float = 1.0,
            seed: int | None = None,
    ) -> None:
        super().__init__(n=n, trainable=True, name=name)

        self._rnd = random.Random(seed)
    
        self.gamma = float(gamma)
        self.lr = float(learning_rate)
        self.delta_clip = float(delta_clip)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_reduce_count = int(epsilon_reduce_count)

        self.target_update_interval = int(target_update_interval)
        self.optimizer = str(optimizer)
        self.loss_name = str(loss_name)
        self.huber_delta = float(huber_delta)

        self.train_steps = 0

        state_diment = n * n
        action_diment = n * n
        sizes = [state_diment] + list(hidden_sizes) + [action_diment]

        self.online_net = NeuralNetwork(
            sizes = sizes,
            hidden_activation="relu",
            output_activation="pass_through",
            seed=seed,
        )
        self.target_net = copy.deepcopy(self.online_net)

        #誤ってfpropとforwardで実装したため、以下で相互性を担保する
        if not hasattr(self.online_net, "fprop"):
            self.online_net.fprop = self.online_net.forward
            self.target_net.fprop = self.target_net.forward

    def epsilon(self) -> float:
        t = self.train_steps
        if t >= self.epsilon_reduce_count:
            return self.epsilon_end
        ratio = t / float(self.epsilon_reduce_count)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def select_action(self, game):
        legal = _all_legal_action_indices(game)
        if not legal:
            return None
        
        if self.player is None:
            raise RuntimeError("DQNAgent.playerがセットされていません。 \n" \
            "ゲームを行う前にagent.playerをセットして下さい。")

        
        if self._rnd.random() < self.epsilon():
            a = self._rnd.choice(legal)
        else:
            s = encode_state_from_player_view(game, self.player)
            q = self.online_net.fprop(s)
            a = pick_best_action_index(q, legal)

        return _index_to_row_col(a, game._size)
    
    def train_from_transition(
            self,
            state: Vector,
            action_index: int,
            reward: float,
            next_state: Vector,
            done: bool,
            next_legal_actions: Sequence[int],
    ) -> float:
        # 目標値
        if done or not next_legal_actions:
            y = float(reward)
        else:
            q_next = self.target_net.fprop(next_state) # ignoreのお話
            best_next = pick_best_action_index(q_next,next_legal_actions)
            y = float(reward) + self.gamma * float(q_next[best_next])

        q_now = self.online_net.fprop(state)
        q_sa = float(q_now[action_index])

        delta = y - q_sa
        if delta > self.delta_clip:
            delta = self.delta_clip
        elif delta < -self.delta_clip:
            delta = -self.delta_clip

        #選んだ行動だけが教師を変える
        target = q_now[:]
        target[action_index] = q_sa + delta

        #nn_mlp.train_one が loss/updateを内部でやる前提
        self.online_net.train_one(
            x=state,
            t=target,
            lr=self.lr,
            loss_name=self.loss_name,
            huber_delta=self.huber_delta,
            optimizer=self.optimizer,
        )

        self.train_steps += 1
        if self.target_update_interval > 0 and (self.train_steps % self.target_update_interval == 0):
            self.target_net = copy.deepcopy(self.online_net)
            if not hasattr(self.target_net,"fprop"):
                self.target_net.fprop = self.target_net.forward

        return float(delta * delta)
    









