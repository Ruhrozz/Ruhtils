# FFMPEG Guide

#### How to concat 2 videos horisontally:
```
ffmpeg -i video1.mp4  -i video2.mp4 -filter_complex hstack=inputs=2 output.avi
```
TODO: Answer which video on left which on right

#### How to compress video
ffmpeg -i video.mp4 -vcodec libx265 -crf 28 compressed.avi
