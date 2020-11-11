"""
evaluate the trained yolov3 model performance
display result and annotation boxes
"""
import json
import os
from timeit import default_timer as timer

import pandas as pd
from PIL import Image, ImageDraw

from framework.yolo import predict
from framework.sample.model import Box


def init_yolo(model_path, anchors_path, classes_path, score=0.3, iou=0.3, input_size=(416, 416), gpu_num=2):
    config = {
        "model_path": model_path,
        "anchors_path": anchors_path,
        "classes_path": classes_path,
        "score": score,
        "iou": iou,
        "model_image_size": input_size,
        "gpu_num": gpu_num,
        "font_path": "../../workspace/detection_model/model_data/font/FiraMono-Medium.otf"
    }
    yolo = predict.YOLO(**config)
    return yolo


def detect_images(img_dir, results_path, yolo):
    with open(results_path, 'a') as results_file:
        N = len(os.listdir(img_dir))
        for i, file in enumerate(os.listdir(img_dir)):
            _start = timer()
            img_path = os.path.join(img_dir, file)
            if os.path.isdir(img_path):
                continue
            if os.path.splitext(img_path)[1] == '.jpg':
                image = Image.open(img_path)
                rs = yolo.detect_image(image, True)
                s = '['
                if rs:
                    for r in rs:
                        box = Box('0', r[1], r[2][0], r[2][1], r[2][2], r[2][3])
                        box_str = box.to_str()
                        s = s + box_str + ','
                    s = s[:-1]
                    s += ']'
                else:
                    s = '[]'
                s = str(img_path) + '\t' + s + '\n'
                results_file.write(s)
                _end = timer()
                print('[{}/{} use time:{}s]'.format(i + 1, N, _end - _start))


# Precision, Recall, F1
def metric(rp, ap):
    p, r, ar, rn, an = 0, 0, 0, 0, 0

    df = load_result_annotation(rp, ap)

    airport_num = len(df)

    for i, row in df.iterrows():
        result = row['results']
        annotation = row['annotations']
        r_boxes = format_boxes(result)
        a_boxes = format_boxes(annotation)

        _p, _r, _rn, _an = 0, 0, len(r_boxes), len(a_boxes)

        if len(r_boxes) == 0:
            an += _an
            continue

        for rb in r_boxes:
            for ab in a_boxes:
                _iou = iou(rb, ab)
                if _iou > 0.3:
                    _p += 1
                    break
        for ab in a_boxes:
            for rb in r_boxes:
                _iou = iou(rb, ab)
                if _iou > 0.3:
                    _r += 1
                    break

        p += _p
        r += _r
        rn += _rn
        an += _an

        if p > 0:
            ar += 1

    print(p, r, ar, rn, an)
    precision = p/rn
    recall = r/an
    f1 = 2 * precision*recall / (precision+recall)
    airport_recall = ar / airport_num
    return precision, recall, f1, airport_recall


def load_result_annotation(rp, ap):
    df_r = pd.read_csv(rp, sep='\t', names=['path', 'results'])
    df_a = pd.read_csv(ap, sep='\t', names=['path', 'annotations'])

    if len(df_r) != len(df_a):
        raise ValueError('file list not match')

    df = pd.merge(df_r, df_a, on='path', how='outer')

    if len(df_r) != len(df):
        raise ValueError('file list not match')

    return df


def load_annotation(ap):
    df_a = pd.read_csv(ap, sep='\t', names=['path', 'annotations'])
    return df_a


def format_boxes(boxes_str: str):
    boxes = json.loads(boxes_str)
    box_list = []
    for box in boxes:
        extents = tuple(box['extents'])
        box_list.append(extents)
    return box_list


def iou(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    width1, height1 = right1 - left1, bottom1 - top1
    width2, height2 = right2 - left2, bottom2 - top2

    end_x = max(right1, right2)
    start_x = min(left1, left2)
    width = width1 + width2 - (end_x - start_x)

    end_y = max(bottom1, bottom2)
    start_y = min(top1, top2)
    height = height1 + height2 - (end_y - start_y)

    if width <= 0 or height <= 0:
        return 0.0

    area1 = width1 * height1
    area2 = width2 * height2
    area = width * height2

    return area * 1. / (area1 + area2 - area)


def show_boxes(row):
    img_path = row['path']
    result = row['results']
    annotation = row['annotations']

    image = Image.open(img_path)

    r_boxes = format_boxes(result)
    a_boxes = format_boxes(annotation)

    draw = ImageDraw.Draw(image)

    print('There are {} results in this image.'.format(len(r_boxes)))
    for rb in r_boxes:
        draw.rectangle(rb, outline='red', width=5)

    print('There are {} annotations in this image.'.format(len(a_boxes)))
    for ab in a_boxes:
        draw.rectangle(ab, outline='blue', width=5)
    image.show()


def show_annotation(row):
    img_path = row['path']
    annotation = row['annotations']

    image = Image.open(img_path)

    a_boxes = format_boxes(annotation)

    draw = ImageDraw.Draw(image)

    print('There are {} annotations in this image.'.format(len(a_boxes)))
    for ab in a_boxes:
        draw.rectangle(ab, outline='blue', width=5)
    image.show()


if __name__ == '__main__':
    # ---- model performance ----
    result_path = r'D:\Data\test_results\test_results_9.txt'
    annotation_path = r'D:\Data\val_sample_enhance\test.txt'

    _model_path = '../../workspace/detection_model/out/trained_weights_final.h5'
    _anchors_path = '../../workspace/detection_model/model_data/yolo_anchors.txt'
    _classes_path = '../../workspace/detection_model/model_data/object_classes.txt'

    _images_dir = ''

    _yolo = init_yolo(_model_path, _anchors_path, _classes_path, 0.3, 0.3)

    start = timer()
    detect_images(_images_dir, result_path, _yolo)
    print('detection use time: {}s'.format(timer()-start))

    start_m = timer()
    p, r, f, ar = metric(result_path, annotation_path)
    print('metric use time: {}s'.format(timer()-start_m))
    print('precision:{}, recall:{}, f1:{}, airport_recall:{}'.format(p, r, f, ar))
    # ---------------------------

    # ----show boxes----
    # df = load_result_annotation(result_path, annotation_path)
    # index = 2222
    #
    # sr = df.iloc[index]
    # show_boxes(sr)
    # ------------------

    # ----show annotation----
    # a_path = r'D:\Data\pt\train_enhance2\test.txt'
    # df = load_annotation(a_path)
    # index = 1002

    # sr = df.iloc[index]
    # show_annotation(sr)
    # -----------------------
