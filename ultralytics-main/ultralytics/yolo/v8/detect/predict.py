# Ultralytics YOLO 🚀, GPL-3.0 license
import numpy
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_img=orig_img, names=self.model.names))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                ###############
                save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
                filename = self.batch[0].split('/')[-3]
                boxe = d.boxes.view(-1).tolist()
                with open('runs/track_results/'+filename+'.txt', 'a') as f:
                    x1 = boxe[0]
                    y1 = boxe[1]
                    w = boxe[2]-boxe[0]
                    h = boxe[3]-boxe[1]
                    track_id = int(boxe[4])
                    score = boxe[5]
                    line = save_format.format(frame=self.seen, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                              w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    f.write(line)
                ################
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

    def write_results1(self, idx, results, batch):
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        filename = batch[0].name
        print(filename)
        with open(filename, 'w') as f:
            for boxes in results[0].boxes:
                for boxe in boxes.boxes:
                    # boxe = numpy.array(boxe)
                    print(boxe)
                    # x1 = boxe[0]
                    # y1 = boxe[1]
                    # w = boxe[2]-boxe[0]
                    # h = boxe[3]-boxe[1]
                    # track_id = int(boxe[4])
                    # score = boxe[5]
                    # line = save_format.format(frame=idx, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                    #                           w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    # f.write(line)
        print(11213)
def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
