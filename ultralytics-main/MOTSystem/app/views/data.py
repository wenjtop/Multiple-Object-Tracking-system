from django.shortcuts import render, HttpResponse, redirect
from app.utils.bootstrap import BootStrapModelForm
from app.utils.pagination import Pagination
from django.http import JsonResponse
from app import models
import time
import json
from trackManage import *
import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = np.asarray(boxes)
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.copy(boxes)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = clip_boxes(bboxes, image.shape)
    patches = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return patches


def management(request):
    queryset = models.video.objects.all()
    page_object = Pagination(request, queryset, page_size=5)
    context = {
        "queryset": page_object.page_queryset,
        "page_string": page_object.html(),
    }
    return render(request, 'management.html', context)

def report(request, nid):
    queryset = models.obj.objects.filter(videoID=nid)
    page_object = Pagination(request, queryset, page_size=5)
    context = {
        "queryset": page_object.page_queryset,
        "page_string": page_object.html(),
    }
    return render(request, 'report.html', context)

class VideoModelForm(BootStrapModelForm):
    bootstrap_exclude_fields = ['video']

    class Meta:
        model = models.video
        fields = ["videoname", "video"]

def addVideo(request):
    title = "上传视频"
    if request.method == "GET":
        form = VideoModelForm()
        return render(request, 'addVideo.html', {"form": form, 'title': title})

    # row_obj = models.UserInfo.objects.filter(pk=id).first()
    form = VideoModelForm(data=request.POST, files=request.FILES)
    if form.is_valid():
        # 对于文件：自动保存；
        # 字段 + 上传路径写入到数据库
        form.instance.username = request.session["info"]["name"]
        form.save()
        return redirect("/data/")
    return render(request, 'addVideo.html', {"form": form, 'title': title})

def video_delete(request, nid):
    if  models.video.objects.filter(username=request.session["info"]["name"]):
        models.video.objects.filter(id=nid).delete()
    return redirect('/data/')

from MOTSystem import settings
def videoTracking(request, nid):
    tg = models.video.objects.filter(username=request.session["info"]["name"], id=nid)
    if request.method == "GET":
        settings.count = 0
        settings.objs = {}
        settings.frame_id = 0
        settings.video_writer_sign = 0
        settings.cap = cv2.VideoCapture(tg[0].video.path)
        # 获取视频的宽度（单位：像素）
        settings.width = int(settings.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        obj = models.obj.objects.filter(videoID=nid)
        for i in obj:
            if os.path.exists(i.obj1):
                os.remove(i.obj1)
            if os.path.exists(i.obj2):
                os.remove(i.obj2)
            if os.path.exists(i.obj3):
                os.remove(i.obj3)
            if os.path.exists(i.obj4):
                os.remove(i.obj4)
            if os.path.exists(i.obj5):
                os.remove(i.obj5)
        models.obj.objects.filter(videoID=nid).delete()
        # 获取视频的高度（单位：像素）
        settings.height = int(settings.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = 30
        settings.video_name = tg[0].video.path.split('video/')[0]+'video/'+str(nid)+'_MoT.mp4'

        if os.path.exists(settings.video_name):
            os.remove(settings.video_name)
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            settings.video_writer = cv2.VideoWriter(settings.video_name, fourcc, fps, (settings.width, settings.height))
        except Exception as e:
            print("打开视频失败！！！")
        settings.trackID = str(time.time())
        settings.trackers[settings.trackID] = gettrackers(cfg='ultralytics/tracker/cfg/enhancesort.yaml')
        if not settings.cap.isOpened():
            print("Error opening video file")
        return render(request, 'videoTracking.html', {'videoname': tg[0].videoname if tg else '出错啦'})

    if tg and request.method == 'POST':
        data = json.loads(request.body)
        if "YES" == data.get('end'):
            if settings.video_writer_sign == 0:
                settings.video_writer_sign = 1
                settings.video_writer.release()
                print("release：OK")
            models.obj.objects.filter(videoID=nid, obj3='').delete()  # 删除少于2个对截图

            models.video.objects.filter(username=request.session["info"]["name"], id=nid).update(videoMOT=settings.video_name)
            return JsonResponse({'status': 'success', 'message': str('e'), 'end':'YES'})
        elif settings.cap.isOpened():
            ret, frame = settings.cap.read()
            if not ret:
                return JsonResponse({'status': 'success', 'message': str('e')})
            im, im0 = dataprocess(frame, imgsz=[640, 640], device=settings.model.device, iscv2=True)
            preds = settings.model(im)
            results = postprocess(preds, im, im0, settings.model)
            det = results[0].boxes.cpu().numpy()
            tracks = settings.trackers[settings.trackID].update(det, im0)

            settings.frame_id += 1
            for item in tracks:
                if int(item[4]) > settings.count:
                    settings.count = int(item[4])
                obj_frame_id = settings.objs.get(int(item[4]), 0)
                if obj_frame_id==0 or settings.frame_id - obj_frame_id > 24:
                    settings.objs[int(item[4])] = settings.frame_id
                    patches = extract_image_patches(im0, [item])
                    objPath = 'video/'+str(nid)+'_'+str(int(item[4]))+'_'+str(settings.frame_id)+'.png'
                    try:
                        cv2.imwrite(settings.video_name.split('video/')[0]+objPath, patches[0])
                    except Exception as e:
                        print("写入错误！！！")

                    obj = models.obj.objects.filter(videoID=nid,trackID=int(item[4])).first()
                    if obj:
                        models.obj.objects.filter(videoID=nid,trackID=int(item[4])).update(obj1=obj.obj2,obj2=obj.obj3,obj3=obj.obj4,obj4=obj.obj5, obj5=objPath)
                    else:
                        models.obj.objects.create(videoID=tg[0], trackID=int(item[4]), obj5=objPath)

            if len(tracks) != 0:
                results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))
                write_results(results, im0, settings.model)

            img_resized = cv2.resize(im0, (settings.width, settings.height))
            settings.video_writer.write(img_resized)
            im0 = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            output_buffer = BytesIO()
            im0.save(output_buffer, format="JPEG")
            im0 = output_buffer.getvalue()
            frame_data = "data:image/jpeg;base64," + base64.b64encode(im0).decode("utf-8")
            return JsonResponse({'status': 'success', 'processed_frame': frame_data, 'time':time.time(),'count': settings.count})
    return JsonResponse({'status': 'error', 'message': str('e')})

import os
from django.http import FileResponse

def download_video(request, nid):
    # 视频文件所在的路径
    tg = models.video.objects.filter(username=request.session["info"]["name"], id=nid)
    if tg:
        # file_path = tg[0].video.path.split('.')[0]+'_MOT.mp4'
        file_path = tg[0].videoMOT
        print(file_path)
        # 确保文件存在
        if os.path.exists(file_path):
            print('OK')
            # 创建一个FileResponse对象来处理文件下载
            response = FileResponse(open(file_path, 'rb'), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response

    # 如果文件不存在，返回一个404 Not Found响应
    raise Http404

