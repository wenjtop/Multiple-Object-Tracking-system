from django.shortcuts import render, HttpResponse, redirect
import json
from MOTSystem import settings
from trackManage import *
from django.http import JsonResponse
def changeModel(request):
    data = json.loads(request.body)
    frame_data = data.get('model')
    print(frame_data)
    if frame_data == 'getName':
        print(settings.modelName)
        return JsonResponse({'status': 'success', 'modelName': settings.modelName})
    if frame_data == 'Person':
        settings.modelName = '行人跟踪'
        settings.model = getmodel(weight="../pretrained/best.pt")
    if frame_data == 'Car':
        settings.modelName = '汽车跟踪'
        settings.model = getmodel(weight="../pretrained/yolov8l.pt")
    if frame_data == 'MultiObj':
        settings.modelName = '多目标跟踪'
        settings.model = getmodel(weight="../pretrained/yolov8l.pt")

    return JsonResponse({'status': 'success'})