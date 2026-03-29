#!/usr/bin/env python3


from abc import ABCMeta, abstractmethod


class TicTacToeAgent(metaclass = ABCMeta):
    """
    TicTacToeのエージェントの基底クラスを実装する

    今後、ランダムエージェント、人間エージェントなどの
    共通のインターフェースとして用いる。
    """
    def __init__(self, n: int = 3, trainable: bool = False,  name: str ="agent"):
        """
        Parameters
        n : int
            盤面サイズ
        name : str
            エージェントの名前
        trainable : bool
            学習可能エージェントかどうか
        """
        
        self.n: int = n  #盤面サイズを保持
        self._name: str = name #エージェント名
        self._trainable: bool = trainable 
        self._player: int | None = None

    @property
    def name(self):
        #エージェントの名前を返す
        return self._name
    
    @property
    def player(self):
        #プレイヤー番号を返す
        return self._player

    @player.setter
    def player(self, player: int):
        #プレイヤー番号を設定する
        self._player = player

    @abstractmethod
    def select_action(self, game):
        """次の一手を選択して返すメソッド
        
        Parameters
        geme : TicTacToe

        Returns
        (x, y): tuple[int, int]
        エージェントが選択した手
        xは行(0 ~ size-1)、yは列(0 ~ size-1)を想定するものとする。
        """
        raise NotImplementedError('select_action is not implemented.')
    
    def notify_result(self, result):
        """
        対戦結果を通知するためのメソッド
        デフォルト実装では何もしない
        Result: int など
            ゲームの結果を表す値
            1:勝ち, 2:引き分け, 3:負け

        """
        #学習しないエージェント（ランダム、人間）などは何もしなくていい
        pass

    def reset(self):
        """
        対局開始に呼び出されるメソッド
        学習型エージェントが内部状況をリセットするためのもの
        """
        pass

    def game_over(self, winner: int):
        """
        試合が終了した時のことを想定したメソッド
        """
        pass

    