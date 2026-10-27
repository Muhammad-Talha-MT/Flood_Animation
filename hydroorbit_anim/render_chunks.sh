#!/bin/bash
# Resumable render: 150-frame chunks into build/chunks/ (2 at a time), then lossless concat. Re-run to resume.
cd "$(dirname "$0")"
PY=../.venv/bin/python
FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=${OUT:-out/hydroorbit_booth_v3.mp4}
JOBS=${JOBS:-1}                    # 2 parallel renderers run out of memory on a 8 GB machine
N=${N:-1410}                       # 47 s x 30 fps
mkdir -p build/chunks
chunk() {
  a=$1; b=$((a+150)); [ $b -gt $N ] && b=$N; f=$(printf "build/chunks/c%04d.mp4" $a)
  [ -f "$f.done" ] && return 0
  $PY render.py --frames $a:$b --out $f > build/chunks/log_$a.txt 2>&1 && touch "$f.done" || { echo "chunk $a failed" >> build/render_log.txt; return 1; }
}
export -f chunk; export PY N
seq 0 150 $((N-1)) | xargs -P $JOBS -I{} bash -c 'chunk {}' || exit 1
for f in build/chunks/c*.mp4; do echo "file '$PWD/$f'"; done > build/chunks/list.txt
$FF -y -loglevel error -f concat -safe 0 -i build/chunks/list.txt -c copy -movflags +faststart $OUT && echo "ALLDONE $OUT" >> build/render_log.txt
