import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.tracker import BOTSORT, BYTETracker, EnhanceTracker
from ultralytics.yolo.utils.checks import check_requirements, check_yaml
from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load

def getmodel(weight="./pretrained/best.pt"):
    model = YOLO(weight)
    device = select_device(None)
    model = AutoBackend(model.model, device=device)
    model.eval()
    return model

def dataprocess(source, imgsz, device):
    im0 = cv2.imread(source)
    im = LetterBox(imgsz, True, stride=32)(image=im0)
    im = im.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = preprocess(im, device=device)
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im, im0

def gettrackers(cfg ='ultralytics/tracker/cfg/enhancesort.yaml'):
    tracker = check_yaml(cfg)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    trackers = EnhanceTracker(args=cfg, frame_rate=30)
    return trackers

def preprocess(img, device):
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    return img

def postprocess(preds, img, orig_img, model, conf_thres=0.1, iou=0.7):
    preds = ops.non_max_suppression(preds,
                                    conf_thres=conf_thres,
                                    iou_thres=iou,
                                    agnostic=False,
                                    max_det=300,
                                    classes=None)
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
        shape = orig_img.shape
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        results.append(Results(boxes=pred, orig_img=orig_img, names=model.names))
    return results


def write_results(results, im0, model):
    annotator = Annotator(im0, line_width=3, example=str(model.names))
    det = results[0].boxes  # TODO: make boxes inherit from tensors
    for d in reversed(det):
        # Add bbox to image
        cls, conf = d.cls.squeeze(), d.conf.squeeze()
        c = int(cls)  # integer class
        name = f'id:{int(d.id.item())} {model.names[c]}' if d.id is not None else model.names[c]
        label = f'{name} {conf:.2f}'
        annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))


def write_results_track(results, im0):
    annotator = Annotator(im0, line_width=3, example=str(model.names))
    det = results[:-1]  # TODO: make boxes inherit from tensors
    for d in reversed(det):
        # Add bbox to image
        cls, conf = d.cls.squeeze(), d.conf.squeeze()
        c = int(cls)  # integer class
        name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else model.names[c]
        label = f'{name} {conf:.2f}'
        annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))


def predYolov8(source='./img/MOT17-11-FRCNN_img1_000855.jpg', imgsz=[832, 1440]):
    model = getmodel()
    im, im0 = dataprocess(source, imgsz, model.device)
    print(im.shape)
    preds = model(im)
    results = postprocess(preds, im, im0, model)
#    write_results(results, im0, model)
    cv2.imwrite('33.png', im0)
    #########################
    trackers = gettrackers(cfg='ultralytics/tracker/cfg/enhancesort.yaml')
    det = results[0].boxes.cpu().numpy()
    tracks = trackers.update(det, im0)
    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))
    write_results(results, im0, model)
    cv2.imwrite('1233.png', im0)
    return results

# results = predYolov8()

#print(results)
