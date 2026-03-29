#!/usr/bin/env python3


from enum import Enum


# === Enumの定義 ===
class Player(Enum):
    FIRST = 1   #先手
    SECOND = 2  #後手

class Cell(Enum):
    EMPTY = 0
    FIRST = 1
    SECOND = 2

class State(Enum):
    PLAYING = 0
    DRAW = 1
    WIN = 2

# === 例外処理 ===
class GameError(Exception):
    """ゲーム進行に関する例外(不正手・状態不整合など)"""

class InvalidMoveError(GameError):
    """不正な手が指示された"""

# ==============

# === tictactoeの実装 ===
class TicTacToe:
    """
    n目並べ(n×n盤)
    - 環境(ゲームルール)としての責務に集中
    - 入力(input)は外でやる(責務分離)
    """

    
    def __init__(self, size: int = 3):
        # nは自由であり、sizeを保持
        if not isinstance(size, int):
            raise TypeError("size must be int")
        if size < 2:
            raise ValueError("size must be >= 2")
        
        self._size: int = size  #盤面サイズ
        self._next: Player = Player.FIRST  #次に打つプレイヤー先手が１後手が２
        self._state: State = State.PLAYING  #ゲームの状態(0:進行中、1:引き分け、2:勝敗あり)
        self._winner = None  

        self._board: list[list[Cell]] = []  #盤面をはじめ０で表す
        for _ in range(size):
            row = [Cell.EMPTY for _ in range(size)]
            self._board.append(row)
        
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def next_player(self):
        return self._next  
    
    @property
    def state(self):      
        return self._state
    
    @property
    def winner(self):
        return self._winner

    def print_board(self) -> None:
        #現在の盤面を表示
        marks = {Cell.EMPTY: ".", Cell.FIRST: "○", Cell.SECOND: "×"}  #○が1 ×が2
        for row in self._board:
            print(" ".join(marks[cell] for cell in row))
        print()

    def play(self, player: Player, x: int, y: int | None = None) -> None:
        #playerがx,yの位置に打つ
        #x,yの指定が二通りあるので分けて考える。

        #ゲームが終わっていたらもう置けない
        if self._state != State.PLAYING:
            raise GameError('ゲームは終わっています')

        #どちらのターンかを表示
        if player != self._next:
            raise InvalidMoveError('今はプレイヤーの番ではありません')  

        #位置の処理
        if y is None:  #インデックス指定
            index = x
            if not isinstance(index, int):
                raise TypeError("index must be int")
            if index < 0 or index >= self._size * self._size:
                raise InvalidMoveError("Index out of range.")
            row = index // self._size
            col = index % self._size
        else:  #座標指定
            row, col = x, y
            if not isinstance(row, int) or not isinstance(col, int):
                raise TypeError("row/col must be int")
            if row < 0 or row >= self._size or col < 0 or col >= self._size:
                raise InvalidMoveError("Row/col out of range.")

        #空きマスチェック
        if self._board[row][col] != Cell.EMPTY:
            raise InvalidMoveError('そのマスには置けません')

        #配置
        if player == Player.FIRST:
            self._board[row][col] = Cell.FIRST
        elif player == Player.SECOND:
            self._board[row][col] = Cell.SECOND

        #勝敗
        if self._check_winner(player, row, col):
            self._state = State.WIN
            self._winner = player
            self._next = None
            return

        #引き分け
        if self._is_draw():
            self._state = State.DRAW
            self._winner = None
            self._next = None
            return

        #続行
        self._next = Player.SECOND if player == Player.FIRST else Player.FIRST

    def _is_draw(self) -> bool:
        for row in self._board:
            for cell in row:
                if cell == Cell.EMPTY:
                    return False
        return True

    def _check_winner(self, player: Player, r: int, c: int) -> bool:
        #勝敗を判定するメソッド
        n = self._size

        if player == Player.FIRST:
            target = Cell.FIRST
        else:
            target = Cell.SECOND

        #列
        if all(self._board[row][c] == target for row in range(n)):
            return True

        #行
        if all(self._board[r][col] == target for col in range(n)):
            return True

        #主体角栓チェック
        if r == c and all(self._board[i][i] == target for i in range(n)):
            return True

        #服対角線チェック
        if r + c == n - 1 and all(self._board[i][n - 1 - i] == target for i in range(n)):
            return True
        
        #揃ってない
        else:
            return False
        
def ask_board_size() -> int:
        """
        ユーザに盤の大きさを入力させる
        """
        while True:
            try:
                n = int(input("盤の大きさを入力してください(例:3 ->3*3) : "))
                if n >= 2:
                    print(f"{n}*{n}で試合を開始します。\n")
                    return n
                else:
                    print("2以上の整数を入力してください\n")
            except ValueError:
                print("整数を入力してください。\n")