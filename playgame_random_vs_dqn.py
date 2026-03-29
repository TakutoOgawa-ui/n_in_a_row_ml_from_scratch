#!/usr/bin/env python3


"""
課題（DQN版）
RandomAgent vs DQNAgent の対戦スクリプト

要求：
- DQNが未学習から開始
- 下記ルールで対戦させ、勝ち/引き分け等の割合の推移を調べる

ルール1: 先手 = Random, 後手 = DQN
ルール2: 先手 = DQN, 後手 = Random
ルール3: 勝者が次の対戦で後手になる

学習（Q学習版と同じ思想）：
- 学習は「DQNが打った行動」を評価して更新したい
- 途中報酬は基本 0
- 終局した場合は
  - DQN勝ち: +1
  - DQN負け: -1
  - 引き分け: 0
- DQNが打った後に相手が打って終局したなら、直前のDQN行動に -1 or 0 を反映して学習
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

from env.tictactoe import TicTacToe, Player, State
from agent.random_agent import RandomAgent
from agent.dqn_agent import DQNAgent, encode_state_from_player_view, _all_legal_action_indices


# -------------------------
# 入力（ask_board_size が無い環境でも動くように）
# -------------------------
def ask_board_size_fallback() -> int:
    while True:
        try:
            n = int(input("盤の大きさを入力してください: "))
            if n >= 2:
                return n
            print("2以上の整数を入力してください。")
        except ValueError:
            print("整数を入力してください。")


try:
    from tictactoe import ask_board_size as ask_board_size  # type: ignore
except Exception:
    ask_board_size = ask_board_size_fallback


# -------------------------
# 便利関数
# -------------------------
def _row_col_to_index(r: int, c: int, n: int) -> int:
    """(row, col) -> action index (0..n^2-1)"""
    return r * n + c


# -------------------------
# 集計用の箱
# -------------------------
@dataclass
class Stats:
    draws: int = 0
    rnd_wins: int = 0
    dqn_wins: int = 0

    def add(self, state: State, winner_agent, rnd_agent, dqn_agent) -> None:
        if state == State.DRAW:
            self.draws += 1
        elif winner_agent is rnd_agent:
            self.rnd_wins += 1
        else:
            self.dqn_wins += 1

    def total(self) -> int:
        return self.draws + self.rnd_wins + self.dqn_wins

    def rates(self) -> Tuple[float, float, float]:
        t = self.total()
        if t == 0:
            return 0.0, 0.0, 0.0
        return self.draws / t, self.rnd_wins / t, self.dqn_wins / t


# -------------------------
# 表示（進捗は1行上書き、intervalはログとして残す）
# -------------------------
import sys

class ResultShowProgress:
    def __init__(self) -> None:
        self._overwrite = sys.stdout.isatty()

    def progress(self, line: str) -> None:
        if self._overwrite:
            print(line, end="\r", flush=True)
        else:
            pass

    def log(self, line: str) -> None:
        # intervalはログとして残す
        if self._overwrite:
            print()
        print(line)

    def finalize(self, line: str) -> None:
        # 最後は改行して確定
        if self._overwrite:
            print()
        print(line)
        print()


# -------------------------
# 1ゲーム
# -------------------------
def play_one_game(
    size: int,
    first_agent,
    second_agent,
    dqn_agent: DQNAgent,
    train: bool = True,
    verbose: bool = False,
):
    """
    1ゲームを進めて結果を返す
    戻り値:
      state: State.WIN / State.DRAW
      winner_agent: 勝ったエージェント(引き分けならNone)
    """
    game = TicTacToe(size)

    agents = {
        Player.FIRST: first_agent,
        Player.SECOND: second_agent,
    }

    # DQNの視点（どっちの手番か）を合わせる
    dqn_agent.player = Player.FIRST if first_agent is dqn_agent else Player.SECOND

    # DQNが打った直前の (state, action_index) を覚えておく（次の相手手番後に評価する）
    prev_state: Optional[List[float]] = None
    prev_action_index: Optional[int] = None

    while game._state == State.PLAYING:
        player = game._next
        agent = agents[player]

        # DQN手番なら「打つ前の state」を先に取る
        if agent is dqn_agent:
            state_before = encode_state_from_player_view(game, dqn_agent.player)

        action = agent.select_action(game)
        if action is None:
            break

        r, c = action
        game.play(player, r, c)

        if verbose:
            who = "DQN" if agent is dqn_agent else "RND"
            print(f"turn player={player} agent={who} state={game._state}")

        # 終局判定
        if game._state != State.PLAYING:
            if game._state == State.DRAW:
                reward = 0.0
            else:
                winner_is_dqn = (agents[game._winner] is dqn_agent)
                reward = 1.0 if winner_is_dqn else -1.0

            if train:
                next_state = encode_state_from_player_view(game, dqn_agent.player)
                next_legal = _all_legal_action_indices(game)

                if agent is dqn_agent:
                    # DQNが打って終局：その手を即評価
                    a_index = _row_col_to_index(r, c, size)
                    dqn_agent.train_from_transition(
                        state=state_before,
                        action_index=a_index,
                        reward=reward,
                        next_state=next_state,
                        done=True,
                        next_legal_actions=next_legal,
                    )
                else:
                    # Randomが打って終局：直前のDQN手（prev）に報酬を返す
                    if prev_state is not None and prev_action_index is not None:
                        dqn_agent.train_from_transition(
                            state=prev_state,
                            action_index=prev_action_index,
                            reward=reward,
                            next_state=next_state,
                            done=True,
                            next_legal_actions=next_legal,
                        )

            break  # 終局

        # 終局してない時
        if agent is dqn_agent:
            # DQNが打ったら prev に保存（相手が打った後に reward=0 で更新するため）
            prev_state = state_before
            prev_action_index = _row_col_to_index(r, c, size)
        else:
            # Randomが打った後：prev があれば reward=0 で更新（終局は上で処理済み）
            if train and prev_state is not None and prev_action_index is not None:
                next_state = encode_state_from_player_view(game, dqn_agent.player)
                next_legal = _all_legal_action_indices(game)

                dqn_agent.train_from_transition(
                    state=prev_state,
                    action_index=prev_action_index,
                    reward=0.0,
                    next_state=next_state,
                    done=False,
                    next_legal_actions=next_legal,
                )

                prev_state = None
                prev_action_index = None

    # 勝者
    if game._state == State.WIN:
        winner_agent = agents[game._winner]
    else:
        winner_agent = None

    return game._state, winner_agent


# -------------------------
# ルール差分：次の先手後手をどうするか
# -------------------------
NextPairFunc = Callable[[Optional[object], object, object], Tuple[object, object]]
# 引数: winner_agent, first_agent, second_agent -> (next_first, next_second)


def next_pair_fixed(_: Optional[object], first_agent, second_agent):
    """先手後手固定"""
    return first_agent, second_agent


def next_pair_winner_goes_second(winner_agent: Optional[object], first_agent, second_agent):
    """
    勝者が次局で後手になる
    - 先手が勝ったら入れ替える
    - 後手が勝ったら維持
    - 引き分けは維持
    """
    if winner_agent is first_agent:
        return second_agent, first_agent
    return first_agent, second_agent


# -------------------------
# 共通の実行ループ
# -------------------------
def run_rule(
    rule_title: str,
    size: int,
    num_games: int,
    interval: int,
    first_agent,
    second_agent,
    rnd: RandomAgent,
    dqn: DQNAgent,
    next_pair_func: NextPairFunc,
) -> None:
    printer = ResultShowProgress()

    total_stats = Stats()
    interval_stats = Stats()
    start = 0

    last_interval_line: Optional[str] = None

    for i in range(num_games):
        state, winner_agent = play_one_game(size, first_agent, second_agent, dqn, train=True, verbose=False)

        total_stats.add(state, winner_agent, rnd, dqn)
        interval_stats.add(state, winner_agent, rnd, dqn)

        first_agent, second_agent = next_pair_func(winner_agent, first_agent, second_agent)

        total = i + 1
        d_rate, r_rate, dqn_rate = total_stats.rates()
        progress_line = (
            f"[{rule_title}]{size}×{size} game={total:7d} | "
            f"draw={d_rate:6.3f} rnd={r_rate:6.3f} dqn={dqn_rate:6.3f}"
        )
        printer.progress(progress_line)

        if (i + 1) % interval == 0:
            end = i + 1
            id_rate, ir_rate, idqn_rate = interval_stats.rates()
            last_interval_line = (
                f"[{rule_title}]{size}×{size} last interval({start+1}-{end}) | "
                f"draw={interval_stats.draws}({id_rate:.3f}) "
                f"rnd={interval_stats.rnd_wins}({ir_rate:.3f}) "
                f"dqn={interval_stats.dqn_wins}({idqn_rate:.3f})"
            )
            printer.log(last_interval_line)

            start = end
            interval_stats = Stats()

    # 最後は last interval を確定表示する
    if last_interval_line is None:
        # interval > num_games のときだけここに来る（その場合は累計をlast intervalとして出す）
        d_rate, r_rate, dqn_rate = total_stats.rates()
        last_interval_line = (
            f"[{rule_title}]{size}×{size} last interval(1-{num_games}) | "
            f"draw={total_stats.draws}({d_rate:.3f}) "
            f"rnd={total_stats.rnd_wins}({r_rate:.3f}) "
            f"dqn={total_stats.dqn_wins}({dqn_rate:.3f})"
        )

    printer.finalize(last_interval_line)


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    size = ask_board_size()
    num_games = 30000
    interval = 3000

    print("=== Random vs DQN ===")
    print(f"size={size}, num_games={num_games}, interval={interval}")
    print()

    rnd = RandomAgent("Random")

    # ルール1: 先手Random, 後手DQN
    dqn1 = DQNAgent(n=size, seed=0)
    run_rule(
        rule_title="Rule1 (Random first)",
        size=size,
        num_games=num_games,
        interval=interval,
        first_agent=rnd,
        second_agent=dqn1,
        rnd=rnd,
        dqn=dqn1,
        next_pair_func=next_pair_fixed,
    )

    # ルール2: 先手DQN, 後手Random
    dqn2 = DQNAgent(n=size, seed=0)
    run_rule(
        rule_title="Rule2 (DQN first)",
        size=size,
        num_games=num_games,
        interval=interval,
        first_agent=dqn2,
        second_agent=rnd,
        rnd=rnd,
        dqn=dqn2,
        next_pair_func=next_pair_fixed,
    )

    # ルール3: 勝者が次局で後手
    dqn3 = DQNAgent(n=size, seed=0)
    run_rule(
        rule_title="Rule3 (winner becomes SECOND)",
        size=size,
        num_games=num_games,
        interval=interval,
        first_agent=dqn3,
        second_agent=rnd,
        rnd=rnd,
        dqn=dqn3,
        next_pair_func=next_pair_winner_goes_second,
    )



# -----------------------
# [DQN ハイパーパラメータ]
# -----------------------
# [学習ハイパーパラメータ]
# - gamma: 0.95
# - learning_rate: 1e-3
# - epsilon: start=1.0, end=0.1, reduce_count=10000
# target_update_interval = 500  # 学習ステップ = train_from_transition呼び出し回数
# log_interval_games = 10000    # 試合数基準（表示）

# [モデル設定（ネットワーク構成）]
# - input_dim: n*n（盤面をベクトル化）
# - hidden: (64, 64)
# - output_dim: n*n（各マスを行動として出力）
# - activation: hidden=relu, output=pass_through
# - optimizer: rmsprop

# -------------------------------
# 試行回数の拡大により掴めたこと
# -------------------------------
# 1,試行が2万~3万回までは顕著に勝率が上がるような学習をしていたが、
# 学習回数が増えるにつれて、勝率が下がるようなある種の学習の劣化のようなものが見られた。
# 2,勝率がわずかに減少して見えるのは、学習中の評価であり、
# ε-greedy探索が常に混入するため、勝率が区間ごとに揺れることが主な要因と考えられる。

# -------------------------------
# 試行回数について
# -------------------------------
# 10万回を採用した理由：
# 2〜3万回までは勝率が顕著に上昇したが、5万回以降は勝率が一定範囲で推移し、
# 10万回まで回しても区間ごとの勝率が大きく改善しないことを確認できた。
# よって学習の立ち上がり→横ばい（揺らぎ）まで観察できる最小の十分量として10万回を採用した。

# -------------------------------
# [対戦結果]（Rule1〜Rule3）
# 結果は (draw, random_win, dqn_win) の比率 or 件数で記録
# -------------------------------

# --- Rule1: Random first（先手=Random, 後手=DQN） ---
# n=3, games=100,000
# total: draw= 0.050 rnd= 0.222 dqn= 0.729
# last_interval(90001-100000) | draw=549(0.055) rnd=1935(0.194) dqn=7516(0.752)

# --- Rule2: DQN first（先手=DQN, 後手=Random） ---
# n=3, games=100,000
# total: draw= 0.027 rnd= 0.083 dqn= 0.890
# last_interval(90001-100000) | draw=321(0.032) rnd=1153(0.115) dqn=8526(0.853)

# --- Rule3: Winner becomes second（勝者が次の対戦で後手になる） ---
# n=3, games=100,000
# total: draw= 0.036 rnd= 0.220 dqn= 0.744
# last_interval(90001-100000) | draw=354(0.035) rnd=2342(0.234) dqn=7304(0.730)