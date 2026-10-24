#!/bin/bash
# Small web copies of the rollout videos for the stage page. Source mp4s (groot_eval.write_video) hold one frame per
# 4 env steps at 10 fps (1 s = 40 steps); the copies keep every frame but play at 5 fps, so second t of the video =
# env step 20t. 384x192 (agentview left + wrist right, 192 px each), H.264 yuv420p, CRF 28, faststart.
# usage: transcode_videos.sh <run_dir> <out_dir>    -> <out_dir>/<Task>_s<seed>.mp4
set -u
RUN=$1; OUT=$2; mkdir -p "$OUT"
for f in "$RUN"/*/*_single.mp4; do
  b=$(basename "$f" _single.mp4); o="$OUT/$b.mp4"
  [ -s "$o" ] && [ "$o" -nt "$f" ] && continue
  nice -n 19 ffmpeg -hide_banner -loglevel error -y -i "$f" -vf "setpts=2.0*PTS,scale=384:192" -r 5 \
    -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart "$o" || echo "FAILED $f"
done
ls -l "$OUT"/*.mp4 | awk '{s+=$5; if(min==""||$5<min)min=$5; if($5>max)max=$5; n++} END{printf "%d files, %.2f-%.2f MB each, %.1f MB total\n", n, min/1e6, max/1e6, s/1e6}'
