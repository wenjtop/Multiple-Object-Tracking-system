import time

import cv2
import subprocess as sp

input_file = 'video.mp4'

# 打开视频文件
cap = cv2.VideoCapture(input_file)

# 获取视频文件的基本信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定义推流命令
command = ['ffmpeg',
           '-y', '-an',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', '{}x{}'.format(width, height),
           '-r', str(fps),
           '-i', '-',
           # '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           # '-preset', 'ultrafast',
           '-f', 'flv',
           'rtmp://localhost:1935/live/stream_name']

# 启动 FFmpeg 子进程
pipe = sp.Popen(command, stdin=sp.PIPE)

# 逐帧读取视频
while True:
    ret, frame = cap.read()
    time.time(1)
    if not ret:
        break

    # 在这里处理帧，例如应用滤镜、调整亮度等
    # 示例：将帧转换为灰度
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # 将处理后的帧推送到 FFmpeg
    pipe.stdin.write(colored_frame.tobytes())

# 释放资源
cap.release()
pipe.stdin.close()
pipe.wait()
