import cv2
import subprocess as sp
import imageio_ffmpeg as ffmpeg

def main():
    # 配置 RTMP 服务器地址和流名称
    rtmp_url = "rtmp://192.168.31.176:1935/live/stream_name"

    # 获取摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 获取摄像头分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 配置 ffmpeg 命令
    command = [
        ffmpeg.get_ffmpeg_exe(),
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}",
        '-r', '25',
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        rtmp_url
    ]

    # 启动子进程，使用 ffmpeg 进行视频推流
    proc = sp.Popen(command, stdin=sp.PIPE, bufsize=10**8)

    while True:
        # 从摄像头捕获帧
        ret, frame = cap.read()

        # 如果未能正确读取帧，则退出循环
        if not ret:
            break

        # 将帧写入 ffmpeg 子进程的标准输入
        proc.stdin.write(frame.tobytes())

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
