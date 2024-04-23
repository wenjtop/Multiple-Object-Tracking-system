import time
import json
import base64
from PIL import Image
from io import BytesIO
from django.shortcuts import render, HttpResponse

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from trackManage import *

# model = getmodel(weight="../pretrained/best.pt")
# trackers = gettrackers(cfg='ultralytics/tracker/cfg/enhancesort.yaml')
from MOTSystem import settings

# @csrf_exempt
def Tracking(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_data = data.get('frame')
        frame_time = data.get('time')
        if settings.frame_time_tmp > frame_time or time.time()-frame_time>100 :
            settings.frame_time_tmp = frame_time
            return JsonResponse({'status': 'error', 'message': str('e')})
        settings.frame_time_tmp = frame_time

        if not frame_data:
            raise ValueError("No frame data provided")
        image_data = frame_data.replace('data:image/jpeg;base64,', '')
        image_data = bytes(image_data, encoding='utf-8')
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        im, im0 = dataprocess(image, imgsz=[288, 512], device=settings.model.device)

        preds = settings.model(im)
        results = postprocess(preds, im, im0, settings.model)
        det = results[0].boxes.cpu().numpy()
        tracks = settings.trackers[settings.trackID].update(det, im0)
        if len(tracks) != 0:
            results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))
            write_results(results, im0, settings.model)
        # cv2.imwrite('1233.png', im0)
        for item in tracks:
            if int(item[4]) > settings.count:
                settings.count = int(item[4])
        im0 = Image.fromarray(im0)
        output_buffer = BytesIO()
        im0.save(output_buffer, format="JPEG")
        im0 = output_buffer.getvalue()
        # im0.save('12413.png')
        frame_data = "data:image/jpeg;base64," + base64.b64encode(im0).decode("utf-8")

        # frame_data是一个base64编码的JPEG图像
        # 如果处理成功，返回成功状态和处理过的帧数据
        return JsonResponse({'status': 'success', 'processed_frame': frame_data, 'time': frame_time, 'count': settings.count})
    else:
        settings.count = 0
        settings.trackID = str(time.time())
        settings.trackers[settings.trackID] = gettrackers(cfg='ultralytics/tracker/cfg/enhancesort.yaml')
        return render(request, 'timeTracking.html')
