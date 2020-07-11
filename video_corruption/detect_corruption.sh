#! /bin/bash

video_dir="../../raw_videos/google-drive"

for i in $(echo $video_dir/*); do 
    b=$(ffprobe "$i" 2>&1 | tail -1 | grep -c "kb/s")
    echo "$b     $(du -hs $i)"
done
