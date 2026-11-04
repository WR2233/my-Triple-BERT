#!/bin/bash
# Miyabi の学習ジョブの完了を待ち、(1) 学習した重みを評価するジョブを自動投入し、(2) その完了を待ち、
# (3) 評価結果を Miyabi 上で summarize_eval.py で集計し、(4) ~/tb/runs 全体を Mac の ./runs に rsync する。
# (Mac側で実行する)
#
# 使い方: [MAIL_TO=<アドレス>] scripts/pull_when_done.sh <学習のジョブID> <~/tb/runs 以下の学習ディレクトリ名> [確認間隔(秒)]
#   例:   MAIL_TO=me@example.com scripts/pull_when_done.sh 3794674 stage2_w14_3794674
#
# 環境変数:
#   MAIL_TO       評価ジョブの完了メールの宛先 (省略時は、メール通知なし。公開リポジトリには書かない)
#   DEADLINE_HHMM この時刻(今日, 24時間表記 例: 0845)を過ぎたら、評価の完了を待たずに転送する (既定: 0845)
#   REPEAT        評価の繰り返し回数 (既定: 8。5時間帯 x 8回 = 約14分)
# 前提: `ssh -fN miyabi` で共有接続を開いてある (OTPなしで ssh できる)。
#       確認のたびに共有接続が使われるので、間隔を空けすぎない限り、接続は切れない (ControlPersist 2h)。
# 転送しないもの: 重み (*.pth) と osrm.log (巨大でGBを超える)。
set -u
JOB="$1"
RUN="$2"
INTERVAL="${3:-300}"
REPEAT="${REPEAT:-8}"
DEADLINE_HHMM="${DEADLINE_HHMM:-0845}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/runs"
MAIL_OPTS=""
if [ -n "${MAIL_TO:-}" ]; then MAIL_OPTS="-m abe -M $MAIL_TO"; fi

past_deadline() { [ "$((10#$(date +%H%M)))" -ge "$((10#$DEADLINE_HHMM))" ]; }

# 指定したジョブIDのうち、まだ終わっていないものの数を返す (sshに失敗したら、空文字)
unfinished() {
  local out n=0 id
  out=$(ssh -o BatchMode=yes miyabi "qstat" 2>&1) || { echo ""; return; }
  for id in "$@"; do
    if echo "$out" | grep -q "^$id "; then n=$((n + 1)); fi
  done
  echo "$n"
}

transfer() {
  echo "$(date +%T) 転送します: ~/tb/runs/ -> $DEST/"
  mkdir -p "$DEST"
  rsync -av --exclude '*.pth' --exclude 'osrm.log' -e "ssh -o BatchMode=yes" "miyabi:~/tb/runs/" "$DEST/"
  echo "$(date +%T) 転送完了: $DEST"
}

# 1. 学習ジョブの完了待ち
while true; do
  n=$(unfinished "$JOB")
  if [ -z "$n" ]; then echo "$(date +%T) ssh に失敗しました (共有接続が切れた可能性)。何もせずに終了します"; exit 1; fi
  if [ "$n" -eq 0 ]; then break; fi
  echo "$(date +%T) 学習ジョブ $JOB は実行中"
  sleep "$INTERVAL"
done
echo "$(date +%T) 学習ジョブ $JOB は終了しました"

# 2. 学習した重みがあるか確認し、評価ジョブを投入する
if ! ssh -o BatchMode=yes miyabi "test -f ~/tb/runs/$RUN/best.pth && test -f ~/tb/runs/$RUN/latest.pth"; then
  echo "$(date +%T) $RUN に best.pth / latest.pth がありません。評価は行わず、転送だけ行います"
  transfer
  exit 1
fi

submit() {  # $1=RUN_NAME $2=重みのファイル名
  ssh -o BatchMode=yes miyabi "cd ~/tb/my-Triple-BERT && mkdir -p logs && qsub $MAIL_OPTS -v REPEAT=$REPEAT,RUN_NAME=$1,MODEL=\$HOME/tb/runs/$RUN/$2 scripts/eval.pbs"
}
ID_BEST=$(submit eval_w14_best best.pth | head -1 | cut -d. -f1)
ID_LATEST=$(submit eval_w14_latest latest.pth | head -1 | cut -d. -f1)
echo "$(date +%T) 評価ジョブを投入しました: best=$ID_BEST latest=$ID_LATEST"
if [ -z "$ID_BEST" ] || [ -z "$ID_LATEST" ]; then
  echo "$(date +%T) 投入に失敗しました。転送だけ行います"; transfer; exit 1
fi

# 3. 評価ジョブの完了待ち (締め切りを過ぎたら待たずに進む)
while true; do
  n=$(unfinished "$ID_BEST" "$ID_LATEST")
  if [ -z "$n" ]; then echo "$(date +%T) ssh に失敗しました。転送せずに終了します"; exit 1; fi
  if [ "$n" -eq 0 ]; then break; fi
  if past_deadline; then echo "$(date +%T) 締め切り $DEADLINE_HHMM を過ぎたので、評価の完了を待たずに進みます"; break; fi
  echo "$(date +%T) 評価ジョブが実行中: 未完了 $n 本"
  sleep 60
done

# 4. 集計 (Miyabi 上で実行。今回の学習の評価結果だけが対象)
echo "$(date +%T) Miyabi 上で summarize_eval.py を実行します"
ssh -o BatchMode=yes miyabi "cd ~/tb && python3 my-Triple-BERT/scripts/summarize_eval.py runs/eval_w14_best_$ID_BEST runs/eval_w14_latest_$ID_LATEST --labels w14_best w14_latest > runs/summary_w14.txt 2>&1; echo \"summary_w14.txt: \$(wc -l < runs/summary_w14.txt) 行\""

# 5. 転送
transfer
echo "$(date +%T) 完了。集計結果: $DEST/summary_w14.txt"
