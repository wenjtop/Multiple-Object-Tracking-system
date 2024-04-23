import cv2
import time
import subprocess as sp

def get_video_stream(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def main():
    # # 获取摄像头的分辨率
    # cap = cv2.VideoCapture(0)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.release()


    input_file = 'video.mp4'
    cap = cv2.VideoCapture(input_file)

    # 获取视频文件的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # 设置 FFmpeg 命令
    cmd = (
        "ffmpeg -f rawvideo -vcodec rawvideo -s {width}x{height} -pix_fmt bgr24 -i - "
        "-c:v libx264 -preset veryfast -c:a aac -hls_time 2 -hls_list_size 10 -hls_flags delete_segments "
        "-f hls output/playlist.m3u8"
    ).format(width=width, height=height)

    ffmpeg_process = sp.Popen(cmd, shell=True, stdin=sp.PIPE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ffmpeg_process.stdin.write(frame.tobytes())
    # for frame in get_video_stream():
    #     # 将视频帧推流给 FFmpeg
    #     ffmpeg_process.stdin.write(frame.tobytes())

if __name__ == "__main__":
    main()
