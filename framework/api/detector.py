"""

Author: Chen Ji <https://github.com/jichen1226>
Created at 2020/11/9
"""
import os
from math import floor
from dataclasses import dataclass
from typing import List
from timeit import default_timer as timer
from PIL import Image
from PIL import ImageFile

from framework.yolo.evaluate import init_yolo


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class ResultBox:
    predicted_class: str
    score: float
    extents: (int, int, int, int)  # x0, y0, x1, y1

    def to_string(self):
        s = '{"class": "'+self.predicted_class+'", "score": '+str(self.score) + \
            ', "extents": ['\
            + str(self.extents[0])+','+str(self.extents[1])+','+str(self.extents[2])+','+str(self.extents[3])+']}'
        return s


@dataclass
class Result:
    image_path: str
    image_size: (int, int)
    boxes: List[ResultBox]

    def format_str(self):
        s = ''+self.image_path+'\t'
        s = s+str(self.image_size)+'\t'
        has_box = True if len(self.boxes) > 0 else False
        s = s+str(has_box)+'\t'
        if has_box:
            bs = '['
            for box in self.boxes:
                bs = bs + box.to_string() + ','
            bs = bs[:-1]
            bs = bs + ']'
            s = s + bs
        return s


def image(yolo, img_path, direct=True, scale=None, step=None) -> Result:
    # in_img = Image.open(img_path)
    with Image.open(img_path) as in_img:

        rs_d = yolo.detect_image(in_img, True)
        boxes_d = [ResultBox(*r) for r in rs_d]

        if direct:
            return Result(img_path, in_img.size, boxes_d)

        # down scale detect
        iw, ih = in_img.size
        iw = round(iw/256)
        ih = round(ih/256)

        if not scale:
            scale = min(iw, ih)
            scale = floor(scale/2) - 1
        if not step:
            step = floor(scale/4)

        if iw < scale:
            scale = iw
        if ih < scale:
            scale = ih

        detect_results = []

        for x in range(0, iw-scale+1, step):
            for y in range(0, ih-scale+1, step):
                left = x*256
                upper = y*256
                right = (x+scale) * 256
                lower = (y+scale) * 256

                # avoid the sub-window extents exceeds the image extents
                if right > iw:
                    right = iw * 256
                if lower > ih:
                    lower = ih * 256

                sub_box = (left, upper, right, lower)
                sub_img = in_img.crop(sub_box)
                rs = yolo.detect_image(sub_img, True)
                if len(rs) > 0:
                    for r in rs:
                        predicted_class, r_score, extents = r
                        x0, y0, x1, y1 = extents
                        extents = (x0+left, y0+upper, x1+left, y1+upper)
                        detect_results.append(ResultBox(predicted_class, r_score, extents))

        detect_results.extend(boxes_d)
        return Result(img_path, in_img.size, detect_results)


def images(yolo, root_dir, out_file, direct=True, scale=None, step=None):
    with open(os.path.join(root_dir, out_file), 'a') as results_file:
        _n = len(os.listdir(root_dir))
        for i, file in enumerate(os.listdir(root_dir)):
            start = timer()
            file_path = os.path.join(root_dir, file)
            if os.path.isdir(file_path):
                continue
            if os.path.splitext(file_path)[1] == '.jpg':
                r = image(yolo, file_path, direct, scale, step)
                results_file.write(r.format_str() + '\n')
                end = timer()
                print('[{}/{} use time:{}s]'.format(i+1, _n, end-start), r.format_str())


def detect_dir(model_path, anchors_path, classes_path, root_path, result_fp, direct=True, scale=None, step=None):

    yolo = init_yolo(model_path, anchors_path, classes_path)
    try:
        images(yolo, root_path, result_fp, direct, scale, step)
    finally:
        yolo.close_session()


if __name__ == '__main__':
    _model_path = r'D:\Workspace\python\GlobalAirportsValidation\workspace\detection_model\out\trained_weights_final.h5'
    _anchor_path = r'D:\Workspace\python\GlobalAirportsValidation\workspace\detection_model\model_data\yolo_anchors.txt'
    _class_path = r'D:\Workspace\python\GlobalAirportsValidation\workspace\detection_model\model_data\object_classes.txt'

    _image_dir = r'D:\Data\airport\all17\test'
    detect_dir(_model_path, _anchor_path, _class_path,
               _image_dir, 'detect_results.txt', False)
