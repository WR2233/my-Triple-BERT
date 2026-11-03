# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""学習ログ(eval.txt / train.txt)から、6指標の推移グラフ(2x3)を描く。

使い方 (uv が依存を自動で用意する):
  # 1回分の学習 (実行ディレクトリ、または eval.txt / train.txt のパス)
  uv run scripts/plot_curves.py ~/tb/runs/stage2_3794500 -o curves.png

  # 複数回を重ねて比較 (最大3本。色は固定順で割り当てる)
  uv run scripts/plot_curves.py runA runB --labels "seed A" "seed B" -o compare.png

  # 複数回の平均 ± 標準偏差の帯 (論文 Fig.3 のような図。2回以上が必要)
  uv run scripts/plot_curves.py run1 run2 run3 --aggregate -o mean_std.png

  # 学習時の値 (探索ノイズあり) を、移動平均で平滑化して描く
  uv run scripts/plot_curves.py runA --source train --smooth 10 -o train.png

  # 軸の範囲を論文 Fig. 3 に合わせる (論文の曲線と見比べるとき)
  uv run scripts/plot_curves.py runA --source eval --paper-axes -o curves_paper_axes.png

eval.txt は 10 エピソードごとの評価 (探索ノイズなし)、train.txt は毎エピソードの学習時の値。
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

METRICS = [
    ("Reward", "Reward"),
    ("Served Order", "Served Order"),
    ("Delivery Time", "Delivery Time (min)"),
    ("Detour Time", "Detour Time (min)"),
    ("Pickup Time", "Pickup Time (min)"),
    ("Confirmation Time", "Confirmation Time (min)"),
]

# カテゴリ色: 固定順で割り当てる (validate_palette.js --pairs all で3色まで検証済み)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"

# 論文 Fig. 3 の軸範囲 (スクリーンショットの目盛りから読み取った概算値。誤差は数 % 程度)
PAPER_XLIM = (-50, 1050)
PAPER_YLIM = {
    "Reward": (10720, 15580),
    "Served Order": (3104, 3762),
    "Delivery Time": (10.36, 13.35),
    "Detour Time": (2.05, 3.74),
    "Pickup Time": (5.41, 9.35),
    "Confirmation Time": (-0.056, 1.11),
}

PAIR_RE =re.compile(r"([A-Za-z][A-Za-z ]*?):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf)")


def resolve(path: str, source: str) -> Path:
    p = Path(path).expanduser()
    return p / f"{source}.txt" if p.is_dir() else p


def load_log(path: Path):
    """各行 '<Eval|Epoch> N | Key: value , Key: value ...' を {指標: 配列} に変換する。"""
    episodes, rows = [], []
    for line in path.read_text().splitlines():
        head, sep, rest = line.partition("|")
        m = re.search(r"(\d+)", head)
        if not sep or not m:
            continue
        values = {k.strip(): float(v) for k, v in PAIR_RE.findall(rest)}
        episodes.append(int(m.group(1)))
        rows.append(values)
    if not rows:
        raise SystemExit(f"データ行が見つかりません: {path}")
    data = {}
    for key, _ in METRICS:
        data[key] = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    return np.array(episodes), data


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """直近 window 点の移動平均 (序盤は、あるだけの点で平均する)。"""
    if window <= 1:
        return y
    csum = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="実行ディレクトリ、またはログファイルのパス")
    ap.add_argument("--source", choices=["eval", "train"], default="eval", help="ディレクトリ指定時に読むログ (既定: eval)")
    ap.add_argument("--labels", nargs="+", help="凡例の名前 (runs と同じ数)")
    ap.add_argument("--aggregate", action="store_true", help="複数回の平均 ± 標準偏差の帯で描く")
    ap.add_argument("--smooth", type=int, default=1, help="移動平均の窓 (点数)。既定 1 = 平滑化なし")
    ap.add_argument("--paper-axes", action="store_true", help="軸の範囲を論文 Fig. 3 に合わせる (x: 0〜1000 付近)")
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--out", default="curves.png")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise SystemExit("--labels の数は runs と同じにしてください")
    if args.aggregate and len(args.runs) < 2:
        raise SystemExit("--aggregate には 2 回以上の実行が必要です")
    if not args.aggregate and len(args.runs) > len(SERIES_COLORS):
        raise SystemExit(f"重ねて描けるのは {len(SERIES_COLORS)} 本までです。--aggregate を使ってください")

    labels = args.labels or [Path(r).expanduser().name or r for r in args.runs]
    runs = [load_log(resolve(r, args.source)) for r in args.runs]

    # 系列を作る: [(ラベル, 色, x, {指標: 平均}, {指標: 標準偏差 or None})]
    series = []
    if args.aggregate:
        common = sorted(set.intersection(*[set(ep.tolist()) for ep, _ in runs]))
        if not common:
            raise SystemExit("全ての実行に共通するエピソードがありません")
        x = np.array(common)
        mean, std = {}, {}
        for key, _ in METRICS:
            stack = np.stack([smooth(d[key], args.smooth)[np.isin(ep, x)] for ep, d in runs])
            mean[key], std[key] = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        series.append((f"mean ± std (n={len(runs)})", SERIES_COLORS[0], x, mean, std))
    else:
        for i, (ep, d) in enumerate(runs):
            series.append((labels[i], SERIES_COLORS[i], ep, {k: smooth(d[k], args.smooth) for k, _ in METRICS}, None))

    plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.ravel(), METRICS):
        ax.set_facecolor(SURFACE)
        for label, color, x, mean, std in series:
            y = mean[key]
            if std is not None:
                ax.fill_between(x, y - std[key], y + std[key], color=color, alpha=0.18, linewidth=0)
            ax.plot(x, y, color=color, linewidth=1.6, label=label, solid_capstyle="round")
            if len(x) <= 12:  # 点が少ないときだけ、点を見せる
                ax.plot(x, y, "o", color=color, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:g}"))
        ax.set_xlabel("Episode", fontsize=9, color=INK_SECONDARY)
        if args.paper_axes:
            ax.set_xlim(*PAPER_XLIM)
            ax.set_ylim(*PAPER_YLIM[key])

    if len(series) >= 2:
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, fontsize=10,
                   labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.965))
    top = 0.90 if len(series) >= 2 else 0.955
    if len(series) == 1:  # 1系列は凡例を出さないので、タイトルで何の図かを示す
        default_title = f"{series[0][0]} ({args.source})"
    else:
        default_title = None
    suptitle = args.title or default_title
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color=INK, x=0.07, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
