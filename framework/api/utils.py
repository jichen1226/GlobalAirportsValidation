"""

Author: Chen Ji <https://github.com/jichen1226>
Created at 2020/11/9
"""
import json
from collections import namedtuple
from dataclasses import dataclass
from typing import List

import pandas as pd
from PIL import Image, ImageDraw


TilesRange = namedtuple('TilesRange', ['zoom', 'x0', 'y0', 'x1', 'y1'])
Extents = namedtuple('Extents', ['left', 'top', 'right', 'bottom'])


@dataclass
class Potential:
    n: int
    score: float
    left: int
    top: int
    right: int
    bottom: int

    def get_extents(self):
        return self.left, self.top, self.right, self.bottom

    def to_str(self):
        ss = '{"score": ' + str(self.score) + \
             ', "extents": [' + str(self.left) + ', ' + str(self.top) + ', ' + str(self.right) + ', ' + str(
            self.bottom) + ']}'
        return ss


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


def load_result_file(result_file, only_detected=True):
    file_format = ['path', 'size', 'detected', 'boxes']
    df = pd.read_csv(result_file, names=file_format, sep='\t')
    if only_detected:
        df = df[df['detected']==True]
    return df


def format_boxes(boxes_str: str):
    boxes = json.loads(boxes_str)
    box_list = []
    for box in boxes:
        extents = tuple(box['extents'])
        box_list.append(extents)
    return box_list


def draw_tile_lines(image, tile_size=256):
    w, h = image.size
    x, y = int(w/tile_size), int(h/tile_size)
    draw = ImageDraw.Draw(image)
    for xx in range(x):
        for yy in range(y):
            if xx != 0:
                draw.line([(xx*tile_size, 0), (xx*tile_size, h)], fill='blue', width=4)
            if yy != 0:
                draw.line([(0, yy*tile_size), (w, yy*tile_size)], fill='blue', width=4)
    return image


def show_boxes(ds, show_tiles=True, duplicate=True):
    img_path = ds['path']
    boxes_str = ds['boxes']
    image = Image.open(img_path)

    if show_tiles:
        image = draw_tile_lines(image)

    box_extents_list = format_boxes(boxes_str)
    draw = ImageDraw.Draw(image)
    print('There are {} detected boxes in this image.'.format(len(box_extents_list)))
    for extents in box_extents_list:
        draw.rectangle(extents, outline='red', width=4)

    if duplicate:
        potentials = filter_box(boxes_str)
        print('There are {} potential regions in this image.'.format(len(potentials)))
        for potential in potentials:
            print(potential.get_extents(), potential.n, potential.score)
            draw.rectangle(potential.get_extents(), outline='yellow', width=8)
    return image


def filter_box(boxes_str: str, threshold_iou: float = 0.3, threshold_n: int = 2,
               other_boxes_str: str = None) -> List[Potential]:
    """
    filter boxes by nms and num of overlap
    :param boxes_str:
    :param threshold_iou:
    :param threshold_n:
    :param other_boxes_str:
    :return:
    """
    boxes = json.loads(boxes_str)

    if other_boxes_str:
        boxes2 = json.loads(other_boxes_str)
        boxes.extend(boxes2)

    boxes.sort(key=lambda _: _['score'], reverse=True)

    keep = []

    while len(boxes) > 0:
        keep_box = boxes[0]
        intersect = 0

        boxes = boxes[1:]
        _boxes = []

        for b in boxes:
            if iou(tuple(keep_box['extents']), tuple(b['extents'])) >= threshold_iou:
                intersect += 1
            else:
                _boxes.append(b)

        if intersect >= threshold_n:
            keep.append(Potential(intersect, keep_box['score'], *tuple(keep_box['extents'])))

        boxes = _boxes

    return keep


if __name__ == '__main__':
    rp = r'D:\Data\airport\all17\test\detect_results.txt'
    df_r = load_result_file(rp)

    index = 3

    image = show_boxes(df_r.iloc[index])
    image.show()
