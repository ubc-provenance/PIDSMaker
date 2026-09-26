#!/bin/bash
# ============================================================================
# DOMAIN 14: MEDIA & FILE FORMAT TOOLS
# ============================================================================
source "$(dirname "$0")/prov_framework.sh"
domain_start "14 — MEDIA"
M="$PROV_WORKDIR/media"; mkdir -p "$M"

section "ImageMagick"
run "convert red png" convert -size 100x100 xc:red "$M/red.png" 2>/dev/null
run "convert blue jpg" convert -size 200x200 xc:blue "$M/blue.jpg" 2>/dev/null
run "convert gradient" convert -size 200x100 gradient:red-blue "$M/gradient.png" 2>/dev/null
run "convert plasma" convert -size 200x200 plasma: "$M/plasma.png" 2>/dev/null
run "convert checkerboard" convert -size 100x100 pattern:checkerboard "$M/checker.png" 2>/dev/null
run "convert resize" convert "$M/red.png" -resize 50x50 "$M/small.png" 2>/dev/null
run "convert rotate" convert "$M/red.png" -rotate 45 "$M/rotated.png" 2>/dev/null
run "convert flip" convert "$M/red.png" -flip "$M/flipped.png" 2>/dev/null
run "convert flop" convert "$M/red.png" -flop "$M/flopped.png" 2>/dev/null
run "convert grayscale" convert "$M/gradient.png" -colorspace Gray "$M/gray.png" 2>/dev/null
run "convert blur" convert "$M/gradient.png" -blur 0x5 "$M/blurred.png" 2>/dev/null
run "convert sharpen" convert "$M/gradient.png" -sharpen 0x3 "$M/sharp.png" 2>/dev/null
run "convert negate" convert "$M/red.png" -negate "$M/negated.png" 2>/dev/null
run "convert png->jpg" convert "$M/red.png" "$M/red.jpg" 2>/dev/null
run "convert png->bmp" convert "$M/red.png" "$M/red.bmp" 2>/dev/null
run "convert png->gif" convert "$M/red.png" "$M/red.gif" 2>/dev/null
run "convert composite" convert "$M/red.png" "$M/blue.jpg" -resize 100x100 -composite "$M/comp.png" 2>/dev/null
run "convert annotate" convert "$M/gradient.png" -annotate +10+20 "PROVENANCE" "$M/annotated.png" 2>/dev/null
run "identify" identify "$M/red.png" 2>/dev/null
run "identify -verbose" identify -verbose "$M/red.png" 2>/dev/null | head -20
run "identify all" find "$M" -name "*.png" -exec identify {} \; 2>/dev/null

section "FFmpeg"
run "ffmpeg gen sine" ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -y "$M/tone.wav" 2>/dev/null
run "ffmpeg wav->mp3" ffmpeg -i "$M/tone.wav" -y "$M/tone.mp3" 2>/dev/null
run "ffmpeg wav->ogg" ffmpeg -i "$M/tone.wav" -y "$M/tone.ogg" 2>/dev/null
run "ffmpeg wav->flac" ffmpeg -i "$M/tone.wav" -y "$M/tone.flac" 2>/dev/null
run "ffmpeg gen video" ffmpeg -f lavfi -i "testsrc=duration=1:size=320x240:rate=10" -y "$M/test.mp4" 2>/dev/null
run "ffmpeg mp4->avi" ffmpeg -i "$M/test.mp4" -y "$M/test.avi" 2>/dev/null
run "ffmpeg mp4->gif" ffmpeg -i "$M/test.mp4" -y "$M/test.gif" 2>/dev/null
run "ffprobe" ffprobe "$M/test.mp4" 2>/dev/null
run "ffprobe -show_format" ffprobe -show_format "$M/test.mp4" 2>/dev/null
run "ffprobe -show_streams" ffprobe -show_streams "$M/test.mp4" 2>/dev/null

section "File type detection"
run "file all media" find "$M" -type f -exec file {} \; 2>/dev/null

rm -rf "$M"
domain_end
