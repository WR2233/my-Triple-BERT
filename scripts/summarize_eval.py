#!/usr/bin/env python3
"""eval.py の出力 (eval_result.txt) を集計し、論文 Table 4 (Multiple Periods) の Triple-BERT の行と比べる。

eval.pbs の REPEAT で、時間帯のリストを繰り返して評価したログを想定している。
(1行 = 1エピソード。HOURS を REPEAT 回並べた順に出力される)

使い方:
  python3 scripts/summarize_eval.py runs/eval_<ジョブID>/eval_result.txt
  python3 scripts/summarize_eval.py runs/eval_<ジョブID>   # ディレクトリでもよい
  python3 scripts/summarize_eval.py a/eval_result.txt b/eval_result.txt --labels best latest
  python3 scripts/summarize_eval.py x/eval_result.txt --hours 19    # HOURS を変えた場合
"""
import argparse
import re
import statistics as st
from pathlib import Path

METRICS = ["Reward", "Served Rate", "Delivery Time", "Detour Time", "Pickup Time", "Confirmation Time"]
# 論文 Table 4 (Average Performance under Multiple Periods) の Triple-BERT の行
PAPER = {"Reward": 14730.48, "Served Rate": 0.98, "Delivery Time": 11.53,
         "Detour Time": 2.52, "Pickup Time": 5.73, "Confirmation Time": 0.13}
PAIR_RE = re.compile(r"([A-Za-z][A-Za-z ]*?):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def load(path: Path):
    p = path / "eval_result.txt" if path.is_dir() else path
    rows = []
    for line in p.read_text().splitlines():
        if "|" not in line:
            continue
        vals = {k.strip(): float(v) for k, v in PAIR_RE.findall(line.split("|", 1)[1])}
        if all(m in vals for m in METRICS):
            rows.append(vals)
    return rows


def fmt(m, s, key):
    return f"{m:,.1f} ± {s:,.1f}" if key == "Reward" else f"{m:.3f} ± {s:.3f}"


def summarize(label, rows, hours, skip):
    rows = rows[skip:]
    n_h = len(hours)
    n_rep = len(rows) // n_h
    if n_rep == 0:
        print(f"[{label}] 行数が足りません ({len(rows)} 行 < 時間帯 {n_h} 個)")
        return None
    if len(rows) % n_h:
        print(f"[{label}] 注意: {len(rows)} 行は {n_h} の倍数ではないので、最初の {n_rep * n_h} 行だけを使います")
    rows = rows[: n_rep * n_h]
    grid = [rows[r * n_h:(r + 1) * n_h] for r in range(n_rep)]  # grid[繰り返し][時間帯]

    print(f"\n## {label}: {n_h} 時間帯 x {n_rep} 回 ({len(rows)} エピソード)")
    print("\n### 時間帯ごと (繰り返しにわたる 平均 ± 標準偏差)")
    print("| hour | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for j, h in enumerate(hours):
        cells = []
        for key in METRICS:
            v = [grid[r][j][key] for r in range(n_rep)]
            cells.append(fmt(st.mean(v), st.stdev(v) if n_rep > 1 else 0.0, key))
        print(f"| {h} | " + " | ".join(cells) + " |")

    print("\n### 全時間帯の平均 (1回分ごとに時間帯平均を取り、繰り返しにわたる 平均 ± 標準偏差) vs 論文 Table 4")
    print("| 指標 | 今回 | 論文 | 差 |")
    print("|---|---|---|---|")
    overall = {}
    for key in METRICS:
        per_rep = [st.mean(grid[r][j][key] for j in range(n_h)) for r in range(n_rep)]
        m, s = st.mean(per_rep), (st.stdev(per_rep) if n_rep > 1 else 0.0)
        overall[key] = m
        diff = (m - PAPER[key]) / PAPER[key] * 100
        print(f"| {key} | {fmt(m, s, key)} | {PAPER[key]:g} | {diff:+.1f}% |")
    return overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="eval_result.txt、またはそれを含むディレクトリ")
    ap.add_argument("--labels", nargs="+", help="表示名 (paths と同じ数)")
    ap.add_argument("--hours", nargs="+", default=["14", "17", "20", "21", "22"], help="1回分の時間帯 (eval.pbs の HOURS)")
    ap.add_argument("--skip", type=int, default=0, help="先頭の無視する行数 (別の実行が混ざっている場合)")
    ap.add_argument("--table4", action="store_true", help="論文 Table 4 と同じ形式 (小数2桁) の行も出力する")
    args = ap.parse_args()
    if args.labels and len(args.labels) != len(args.paths):
        raise SystemExit("--labels の数は paths と同じにしてください")
    results = []
    for i, p in enumerate(args.paths):
        label = args.labels[i] if args.labels else str(p)
        res = summarize(label, load(Path(p).expanduser()), args.hours, args.skip)
        if res:
            results.append((label, res))

    if args.table4 and results:  # 論文 Table 4 と同じ並びで、小数2桁の行だけを出す
        print("\n## Table 4 形式 (全時間帯の平均。Service Rate は小数2桁)")
        print("| Method | " + " | ".join(METRICS) + " |")
        print("|---|" + "---|" * len(METRICS))
        for label, res in results:
            print(f"| {label} | " + " | ".join(f"{res[k]:.2f}" for k in METRICS) + " |")
        print("| (論文) Triple-BERT | " + " | ".join(f"{PAPER[k]:.2f}" for k in METRICS) + " |")


if __name__ == "__main__":
    main()
