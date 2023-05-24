ffmpeg -i volosnitsa_rural.mp4  -i kay_rural.mp4 -filter_complex hstack=inputs=2 output
ffmpeg -i zombie.mp4 -vcodec libx265 -crf 28 compressed.avi
