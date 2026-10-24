#!/bin/bash
# Resumable render: 150-frame chunks into build/chunks/, then lossless concat. Re-run to resume.
cd "$(dirname "$0")"
PY=../.venv/bin/python
FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p build/chunks
for a in $(seq 0 150 1050); do
  b=$((a+150)); f=$(printf "build/chunks/c%04d.mp4" $a)
  [ -f "$f.done" ] && continue
  $PY render.py --frames $a:$b --out $f >> build/render_log.txt 2>&1 && touch "$f.done" || { echo "chunk $a failed" >> build/render_log.txt; exit 1; }
done
for f in build/chunks/c*.mp4; do echo "file '$PWD/$f'"; done > build/chunks/list.txt
$FF -y -loglevel error -f concat -safe 0 -i build/chunks/list.txt -c copy -movflags +faststart out/hydroorbit_booth_v2.mp4 && echo "ALLDONE" >> build/render_log.txt
