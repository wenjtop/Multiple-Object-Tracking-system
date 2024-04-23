import os
import torch
from ultralytics import YOLO
model = YOLO("./pretrained/yolov8l.pt")  # or a segmentation model .i.e yolov8n-seg.pt
# # # Load a model
# model = YOLO("yolov8s.yaml")  # build a new model from scratch
# model = YOLO("./pretrained/best.pt")  # load a pretrained model (recommended for training)
# # #
# # # # Use the model
# # # model.train(data="coco128.yaml", batch=8, epochs=10, imgsz=1440)  # train the model
# # # metrics = model.val()  # evaluate model performance on the validation set
# # # r = model.model(torch.rand(2,3,640,640))
# #
# # # results = model("000229.jpg", save=True)  # predict on an image
# # # # success = model.export(format="onnx")  # export the model to ONNX format
# # # model = YOLO("best.pt")
# # # 验证
# #
# model.val(data="coco128.yaml", batch=1, imgsz=1440)
#
#
# results = model.byte_track(source='./datasets/yolo_mot_ch/annotations/val_half.json', stream=False, imgsz=[800,1440], save=True)

# # 测试

# results = model.predict(source='./img/MOT17-11-FRCNN_img1_000855.jpg', stream=False, imgsz=[832,1440], save=True)


# # 追踪
model.track(
    # source="./video/test_person.mp4",
    source="./video/car.mp4",
    tracker="ultralytics/tracker/cfg/enhancesort.yaml",
    save=True,
    imgsz=[832,1440],
)

# MOT17-02-FRCNN
# MOT17-04-FRCNN
# MOT17-05-FRCNN
# MOT17-09-FRCNN
# MOT17-10-FRCNN
# MOT17-11-FRCNN
# MOT17-13-FRCNN",

# imglist = ['./datasets/yolo_mot_ch/val_half/MOT17-02-FRCNN/img1',
#            './datasets/yolo_mot_ch/val_half/MOT17-04-FRCNN/img1',
#            './datasets/yolo_mot_ch/val_half/MOT17-05-FRCNN/img1',
#            './datasets/yolo_mot_ch/val_half/MOT17-09-FRCNN/img1',
#            './datasets/yolo_mot_ch/val_half/MOT17-10-FRCNN/img1',
#            './datasets/yolo_mot_ch/val_half/MOT17-11-FRCNN/img1',
#            './datasets/yolo_mot_ch/val_half/MOT17-13-FRCNN/img1']

# imglist = ['./datasets/yolo_mot_ch/val_half/MOT17-10-FRCNN/img1',]

# for file in os.listdir('./runs/track_results'):
#     os.remove(os.path.join('./runs/track_results', file))
#
# for dataPath in imglist:
#     model = YOLO("./pretrained/best.pt")  # or a segmentation model .i.e yolov8n-seg.pt
#     model.track(
#         # source="./video/test_person.mp4",
#         source=dataPath,
#         tracker="ultralytics/tracker/cfg/enhancesort.yaml",
#         save=False,
#         save_txt=True,
#     )
