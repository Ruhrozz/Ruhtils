# FFMPEG Guide

#### How to concat 2 videos horisontally:
```
ffmpeg -i video1.mp4  -i video2.mp4 -filter_complex hstack=inputs=2 output.mp4
```
TODO: Answer which video on left which on right

#### How to compress video
```
ffmpeg -i video.mp4 -vcodec libx265 -crf 28 compressed.mp4
```

#### How to make video from frames.jpg
```
ffmpeg -framerate 30 -pattern_type glob -i '*.jpg'   -c:v libx264 -qp 0 -pix_fmt yuv420p -video_size 1920x1080 out.mp4
```
-qp 0 это сжатие
