学習1episode: 約37秒 (Stage2, 評価込みの平均。1000episode ≒ 10.3時間)
推論1episode: 約25秒 (1時間帯 = 30ステップ。eval.pyの既定は5時間帯で約2〜3分)



学習曲線検証プロット用
```
rsync -av --exclude '*.pth' --exclude 'osrm.log' miyabi:~/tb/runs/timing_3794517/ ./runs/timing_3794517/
uv run scripts/plot_curves.py runs/timing_3794517 --source eval --smooth 5 -o timing_train_eval.png
uv run scripts/plot_curves.py runs/timing_3794517 --source eval --smooth 5 --paper-axes -o timing_train_paper_axes.png 
```
- `--source eval`: 10episodeごとの評価 (探索ノイズなし。論文の曲線はこちら) / `--source train`: 毎episodeの学習時の値 (ノイズあり)
- 複数回の平均±標準偏差の帯: `uv run scripts/plot_curves.py run1 run2 run3 --aggregate -o x.png`
- 読むのは train.txt / eval.txt だけ (osrm.log, timing.log は使わない)

## Miyabi 接続・ジョブ
```
ssh miyabi                  # 設定済み(OTP要)。共有接続は2時間有効
ssh -fN miyabi              # 接続だけ保持 (以後 ssh/rsync はOTP不要)
ssh -O check miyabi         # 接続が生きているか
ssh -O exit miyabi          # 接続終了
qstat                       # 実行中・待機中 (括弧付きの日時は開始予定)
qstat -H                    # 終了済み
qdel <ジョブID>
```
- ログ: `~/tb/my-Triple-BERT/logs/<ジョブID>.opbs.OU` (PBS標準出力) / 学習の出力: `~/tb/runs/<RUN_NAME>_<ジョブID>/` (train.txt, eval.txt, latest.pth, best.pth, ckpt_100..., progress.log, run_info.log)
- 投入前: `cd ~/tb/my-Triple-BERT && git pull && mkdir -p logs`

キュー (公式案内より): debug-g 30分 / short-g 8時間(1〜8ノード) / regular-g→small-g 48時間(1〜16ノード)
```
# 本番(約10.3h): 24hだと開始が遠い。短くしても早まる保証はない
qsub -l walltime=14:00:00 -m abe -M <アドレス> -v RUN_NAME=stage2_w14 scripts/train_stage2.pbs
# 今すぐ寄りの学習: short-g 8時間 (約780episodeで打ち切り。1000には届かない)
qsub -q short-g -l walltime=08:00:00 -m abe -M <アドレス> -v RUN_NAME=stage2_short scripts/train_stage2.pbs
# 推論 (時間帯はHOURSで指定。19時だけ等)
qsub -M <アドレス> -v HOURS="19" scripts/eval.pbs
# 所要時間の測定 (24分)
qsub -M <アドレス> scripts/train_timing.pbs
```
- メール宛先は公開リポジトリに書かず `-M` で渡す。`-m abe` = 開始/異常終了/終了で通知 (メール通知は動作確認済み)
- トークン: Miyabi-G 1ノード×1時間 = 1トークン。Stage2は1回 約10トークン。待機中は消費しない
- システム停止予定: 2026/11/25 09:00 (メンテナンス)。それをまたぐジョブは避ける

## 投入済みジョブ (2026-11-04)
- 3794674: small-g, walltime 14h, 開始予定 11/05 21:57 (RUN_NAME=stage2_w14)
- 3794811: short-g, walltime 8h (RUN_NAME=stage2_short)
- どちらかが始まったら、もう一方を消すかは要判断 (別run = 別乱数の2回分として両方走らせる手もある)

## 検証メモ
- 重みの読み込み: Worker.load() を修正。未使用のBERT埋め込みキー(word/position_embeddings)だけ読み飛ばす
- Actor BERTの max_len: 1500→2500 (学習序盤は未処理の注文が多く 1000+503 が上限1500を超えて落ちた)
- 100episodeごとに ckpt_<N>.pth を保存 (train.py)
- 推論(19時, best.pth) 6回: 報酬 平均15035 (14970〜15096)。学習ログの評価(平均15330)よりやや低い。地図(OSM)とOSRMのバージョン違いが原因かも (未検証)
- 論文 Table 4 (Multiple Periods) との比較 (eval.pyの既定5時間帯の平均。時間帯の対応は推測):
  Reward 14172 (論文14730, -3.8%) / Service 0.974 (0.98) / Delivery 11.94 (11.53) / Detour 2.76 (2.52) / Pickup 5.96 (5.73) / Confirmation 0.14 (0.13)
- Table 2 は FHV データ。公開重みはYellow Taxi用と思われ、ベースラインの実装もないため再現は難しい
- GPUはGH200で使われている (timing.log: cuda available: True)
- timing_3794517: Stage1重み(parameters/stage1/best.pth)ベースで24分38episode。クラッシュなし

## 注意
- git commit/push/pull は自分で行う
- data/ の parquet, osm.pbf, taxi_zones, pkl は著作権・容量のためコミットしない (.gitignore済み)
- runs/ と *.png は未追跡。コミットしないよう注意
