#!/bin/bash
# Full render, resumable: finished 150-frame chunks (build/chunks/*.done) are skipped on re-run.
# Second pass retries any chunk that failed in the first. Run from anywhere.
cd "$(dirname "$0")"
export OUT=${OUT:-out/hydroorbit_booth_v5.mp4}
./render_chunks.sh || ./render_chunks.sh
