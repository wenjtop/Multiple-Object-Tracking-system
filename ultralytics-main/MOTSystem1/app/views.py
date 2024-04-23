import time
import json
import base64
from PIL import Image
from io import BytesIO
from django.shortcuts import render, HttpResponse



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from trackManage import *

# Create your views here.
def login(requset):
    return render(requset, 'login.html')

# model = getmodel(weight="../pretrained/best.pt")
# trackers = gettrackers(cfg='ultralytics/tracker/cfg/enhancesort.yaml')
from MOTSystem import settings
@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        frame_data = data.get('frame')
        frame_time = data.get('time')

        if settings.frame_time_tmp > frame_time or time.time()-frame_time>100 :
            settings.frame_time_tmp = frame_time
            print(settings.frame_time_tmp)
            return JsonResponse({'status': 'error', 'message': str('e')})
        settings.frame_time_tmp = frame_time

        if not frame_data:
            raise ValueError("No frame data provided")
        image_data = frame_data.replace('data:image/jpeg;base64,', '')
        image_data = bytes(image_data, encoding='utf-8')
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        im, im0 = dataprocess(image, imgsz=[288, 512], device=model.device)

        preds = model(im)
        results = postprocess(preds, im, im0, model)
        det = results[0].boxes.cpu().numpy()
        tracks = trackers.update(det, im0)
        if len(tracks) != 0:
            results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))
            write_results(results, im0, model)
        # cv2.imwrite('1233.png', im0)
        im0 = Image.fromarray(im0)
        output_buffer = BytesIO()
        im0.save(output_buffer, format="JPEG")
        im0 = output_buffer.getvalue()
        # im0.save('12413.png')
        frame_data = "data:image/jpeg;base64," + base64.b64encode(im0).decode("utf-8")
        # 在这里处理帧数据
        # frame_data是一个base64编码的JPEG图像

        # 如果处理成功，返回成功状态和处理过的帧数据（如果需要）
        return JsonResponse({'status': 'success', 'processed_frame': frame_data, 'time': frame_time})
    else:
        return render(request, 'video2.html')
      #  return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
