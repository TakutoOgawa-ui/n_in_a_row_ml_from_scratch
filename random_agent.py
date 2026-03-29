#!/usr/bin/env python3


"""
Random_Agent.py

TicTacToeの盤面から’打てるマス’を見つけ、
その中からランダムに１つ選ぶエージェント
"""

import random

from .agent_base import TicTacToeAgent
from env.tictactoe import Cell


class RandomAgent(TicTacToeAgent):
    """
    ランダムに手を選ぶエージェント
    TicTacToeAgentのselect_actionをオーバーライドする必要がある
    """

    def __init__(self, name= "Random"):
        """
        name : エージェントの名前を指定できる
        """
        super(). __init__(name= name)

    def select_action (self, game):  #ランダムエージェントの頭脳に当たる部分
        size = game._size
        """
        盤面の空きマスから一つ選んで返す

        Parameter
        game : TicTacToe
            game._size : 盤面の一辺の長さ(3*3なら3)
            game._board : 現在の盤面の二次元リスト
                        0: 空
                        1: 先手
                        2: 後手
        Returns
        (row, col):tuple[int,int] or None
        """

        empty_cells = []
        #全盤面を総当たりでチェックする
        for row in range(size):
            for col in range(size):
                #game_board[row][col]が0なら空きマス
                if game._board[row][col] == Cell.EMPTY:
                    empty_cells.append((row, col)) #空きマスに追加

        #打てるマスがないならNoneを返す
        if not empty_cells:
            return None

        #ランダムに１マス選ぶ
        row, col = random.choice(empty_cells)

        #タプルで返す
        return (row, col)







                 

