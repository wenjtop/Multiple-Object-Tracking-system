from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import base64
from io import BytesIO

from PIL import Image

# Create your views here.
def login(requset):
    return render(requset, 'login.html')

def save_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('image', '')
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = bytes(image_data, encoding='utf-8')
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        # 在这里处理图像数据，例如使用OpenCV库进行处理
        # ...

        output_buffer = BytesIO()
        image.save(output_buffer, format='PNG')
        image_data = output_buffer.getvalue()
        image_data = base64.b64encode(image_data).decode('utf-8')
        return JsonResponse({'image': 'data:image/png;base64,' + image_data})
    else:
        return render(request, 'video.html')
        # return JsonResponse({'message': 'Invalid request method.'})



def process_image(request):
    if request.method == 'POST':
        image_data = request.POST.get('image', '')
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = bytes(image_data, encoding='utf-8')
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        # 在这里处理图像数据，例如使用OpenCV库进行处理
        # ...

        output_buffer = io.BytesIO()
        image.save(output_buffer, format='PNG')
        image_data = output_buffer.getvalue()
        image_data = base64.b64encode(image_data).decode('utf-8')
        return JsonResponse({'image': 'data:image/png;base64,' + image_data})
    else:
        return JsonResponse({'message': 'Invalid request method.'})

import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

class StreamConsumer(AsyncWebsocketConsumer):
	async def connect(self):
		await self.accept()

	async def disconnect(self, close_code):
		pass

	async def receive(self, text_data):
		await self.send(text_data=json.dumps({'message': 'Text data is not supported.'}))

	async def receive_bytes(self, bytes_data):
		await self.send(bytes_data=await self.process_video(bytes_data))

	@sync_to_async
	def process_video(self, bytes_data):
		file_path = os.path.join(settings.MEDIA_ROOT, 'input.webm')
		with open(file_path, 'wb') as f:
			f.write(bytes_data)
		ffmpeg = FFmpeg(inputs={'input.webm': None}, outputs={'output.mp4': '-c:v libx264 -crf 22 -c:a aac -b:a 128k -f mp4 -y'})
		ffmpeg.run()
		file_path = os.path.join(settings.MEDIA_ROOT, 'output.mp4')
		with open(file_path, 'rb') as f:
			return f.read()

import json
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ../../ultralytics-main.trackManage import *
getmodel()

@csrf_exempt
def process_frame(request):

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            frame_data = data.get('frame')
            if not frame_data:
                raise ValueError("No frame data provided")
            image_data = frame_data.replace('data:image/jpeg;base64,', '')
            image_data = bytes(image_data, encoding='utf-8')
            image = Image.open(BytesIO(base64.b64decode(image_data)))

            image.save('234.png')

            # 在这里处理帧数据
            # frame_data是一个base64编码的JPEG图像

            # 如果处理成功，返回成功状态和处理过的帧数据（如果需要）
            return JsonResponse({'status': 'success', 'processed_frame': frame_data})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    else:
        return render(request, 'video2.html')
      #  return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
