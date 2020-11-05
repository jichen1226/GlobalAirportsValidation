import functools
import os
import random
import xml.etree.ElementTree as ET
import json
from math import floor, sin, cos, pi, atan
from typing import List

from code.sample.model import Sample, Annotation, Label, Point, Box

from PIL import Image


def load_annotation(file_path: str, a_type: str, auxiliary_name: str):
    """

    :param file_path: annotation file path
    :param a_type: VOC or labelme
    :param auxiliary_name: if have auxiliary field, name it
    :return:
    """
    if a_type == 'VOC':
        if not file_path.endswith('.xml'):
            file_path = file_path + '.xml'

        annotation_file = ET.parse(file_path)
        root = annotation_file.getroot()
        img_name = os.path.splitext(root.find('filename').text)[0]
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        boxes = root.findall('object')
        labels = []
        for box in boxes:
            name = box.find('name').text
            auxiliary = True if name == auxiliary_name else False
            shapes = [Point(round(box.find('bndbox').find('xmin').text),
                            round(box.find('bndbox').find('ymin').text)),
                      Point(round(box.find('bndbox').find('xmax').text),
                            round(box.find('bndbox').find('ymax').text))]
            labels.append(Label(name, shapes, auxiliary))
        return Annotation(img_name, img_width, img_height, labels)

    if a_type == 'labelme':
        if not file_path.endswith('.json'):
            file_path = file_path + '.json'

        with open(file_path, 'r') as load_f:
            af = json.load(load_f)
            img_name = os.path.splitext(af['imagePath'])[0]
            img_width = int(af['imageWidth'])
            img_height = int(af['imageHeight'])
            labels_shapes = af['shapes']
            labels = []
            for label_shapes in labels_shapes:
                name = label_shapes['label']
                auxiliary = True if name == auxiliary_name else False
                shapes = [Point(int(p[0]), int(p[1])) for p in label_shapes['points']]
                labels.append(Label(name, shapes, auxiliary))
            return Annotation(img_name, img_width, img_height, labels)


def transform_annotation(old_annotation: Annotation, transform_type, **kwargs):
    """
    转换样本标签
    :param old_annotation:
    :param transform_type:
    :param kwargs:
    :return:
    """
    if transform_type == 'clip':
        c_xmin, c_ymin, c_xmax, c_ymax = kwargs['clip_img_box']
        new_labels = []
        for old_label in old_annotation.labels:
            new_shapes = []
            for old_point in old_label.shapes:
                x = old_point.x - c_xmin
                y = old_point.y - c_ymin
                if x < 0:
                    x = 0
                if x > c_xmax - c_xmin:
                    x = c_xmax - c_xmin - 1
                if y < 0:
                    y = 0
                if y > c_ymax - c_ymin:
                    y = c_ymax - c_ymin - 1
                new_shapes.append(Point(x, y))
            new_label = Label(old_label.name, new_shapes, old_label.auxiliary)
            n_x_min, n_y_min, n_x_max, n_y_max = new_label.get_extents()
            # 如果标签在新图片外，则不添加
            if abs(n_x_max-n_x_min) == 0 or abs(n_y_max-n_y_min) == 0:
                continue
            new_labels.append(new_label)
        return Annotation(old_annotation.image_name, c_xmax - c_xmin, c_ymax - c_ymin, new_labels)

    if transform_type == 'rotate':
        image_height, image_width = old_annotation.image_width, old_annotation.image_height
        new_labels = [Label(old_label.name,
                            [Point(old_point.y, image_height-old_point.x-1) for old_point in old_label.shapes],
                            old_label.auxiliary)
                      for old_label in old_annotation.labels]
        return Annotation(old_annotation.image_name, image_width, image_height, new_labels)

    if transform_type == 'rotate2':
        a = kwargs['a']
        w = kwargs['w']
        h = kwargs['h']
        nw = kwargs['nw']
        nh = kwargs['nh']
        new_labels = [Label(old_label.name,
                            [Point(*transform_point(w, h, a, old_point)) for old_point in old_label.shapes],
                            old_label.auxiliary)
                      for old_label in old_annotation.labels]
        return Annotation(old_annotation.image_name, nw, nh, new_labels)

    if transform_type == 'mirror':
        image_width, image_height = old_annotation.image_width, old_annotation.image_height
        new_labels = [Label(old_label.name,
                            [Point(image_width-old_point.x-1, old_point.y) for old_point in old_label.shapes],
                            old_label.auxiliary)
                      for old_label in old_annotation.labels]
        return Annotation(old_annotation.image_name, image_width, image_height, new_labels)

    if transform_type == 'resize':
        new_img_width, new_img_height = kwargs['new_size']
        width_per = new_img_width/old_annotation.image_width
        height_per = new_img_height/old_annotation.image_height
        new_labels = [Label(old_label.name,
                            [Point(floor(old_point.x * width_per), floor(old_point.y * height_per))
                             for old_point in old_label.shapes],
                            old_label.auxiliary)
                      for old_label in old_annotation.labels]
        return Annotation(old_annotation.image_name, new_img_width, new_img_height, new_labels)


def transform_point(w, h, a, point):
    x0, y0 = point
    rad_a = a / 180 * pi
    nh = w*sin(rad_a) + h*cos(rad_a)
    x = x0*cos(rad_a) - (h-y0)*sin(rad_a)
    y = x0*sin(rad_a) + (h-y0)*cos(rad_a)
    return round(h*sin(rad_a) + x), round(nh-y)


def load_sample(fp: str, name: str, img_type: str = '.jpg', an_type: str = 'labelme') -> Sample:
    image = Image.open(os.path.join(fp, name+img_type))
    annotation = load_annotation(os.path.join(fp, name), an_type, 'airport')
    return Sample(image, annotation)


def zoom_clip_image(sample: Sample, ratio: float, random_shift=False) -> Sample:
    """
    zoom, then clip by ratio
    :param sample:
    :param ratio:
    :param random_shift:
    :return:
    """
    if ratio > 1:
        ratio = 1
    if ratio < 0:
        raise ValueError('ratio must be greater than 0')

    image = sample.image
    annotation = sample.annotation

    labels = annotation.labels
    object_extents = None
    for label in labels:
        if label.auxiliary:
            object_extents = label.get_extents()
    if object_extents is None:
        raise ValueError('can not find auxiliary label')

    img_width, img_height = image.size
    ob_width = abs(object_extents.x_max - object_extents.x_min)
    ob_height = abs(object_extents.y_max - object_extents.y_min)

    if ratio == 1:
        # 如果目标占满整个子图片，则裁剪成机场范围
        clip_img_box = (object_extents.x_min, object_extents.y_min, object_extents.x_max, object_extents.y_max)
        return Sample(image.crop(clip_img_box), transform_annotation(annotation, 'clip', clip_img_box=clip_img_box))

    x_mid = object_extents.x_min + floor(ob_width/2)
    y_mid = object_extents.y_min + floor(ob_height/2)

    size = min(img_width, img_height)

    if max(ob_width/ratio, ob_height/ratio) < size:
        # 符合条件说明大小合适，选择合适的范围即可
        size = max(ob_width/ratio, ob_height/ratio)
        c_xmin = x_mid-floor(size/2)
        c_ymin = y_mid-floor(size/2)
        c_xmax = x_mid+floor(size/2)
        c_ymax = y_mid+floor(size/2)

        if random_shift is True:
            shift_x = random.randint(-floor(ob_width/2), floor(ob_width/2))
            shift_y = random.randint(-floor(ob_height / 2), floor(ob_height / 2))
            c_xmin = c_xmin + shift_x
            c_ymin = c_ymin + shift_y
            c_xmax = c_xmax + shift_x
            c_ymax = c_ymax + shift_y

        if c_xmin < 0:
            c_xmin = 0
            c_xmax = 2 * floor(size/2)
        if c_ymin < 0:
            c_ymin = 0
            c_ymax = 2 * floor(size/2)
        if c_xmax > img_width-1:
            c_xmax = img_width-1
            c_xmin = c_xmax - 2 * floor(size/2)
        if c_ymax > img_height-1:
            c_ymax = img_height-1
            c_ymin = c_ymax - 2 * floor(size/2)

        clip_box = (c_xmin, c_ymin, c_xmax, c_ymax)
        return Sample(image.crop(clip_box), transform_annotation(annotation, 'clip', clip_img_box=clip_box))

    return zoom_clip_image(sample, ratio+0.05, random_shift)


def rotate_image(sample: Sample) -> Sample:
    """
    逆时针旋转90度
    :param sample:
    :return:
    """
    # TODO: bug，旋转y轴坐标向下偏移
    image = sample.image
    annotation = sample.annotation
    rotate = Image.ROTATE_90
    return Sample(image.transpose(rotate), transform_annotation(annotation, 'rotate'))


def rotate_image2(sample: Sample, angle=90) -> Sample:
    """
    逆时针旋转
    :param sample:
    :param angle: 旋转角度
    :return:
    """
    image = sample.image
    w, h = image.size
    annotation = sample.annotation
    if angle > 90 or angle < 0:
        raise ValueError('angle must in range [0, 90]')
    new_image = image.rotate(angle, expand=True)
    nw, nh = new_image.size
    return Sample(new_image, transform_annotation(annotation, 'rotate2', a=angle, w=w, h=h, nw=nw, nh=nh))


def mirror_image(sample: Sample) -> Sample:
    """
    左右镜像
    :param sample:
    :return:
    """
    image = sample.image
    annotation = sample.annotation
    return Sample(image.transpose(Image.FLIP_LEFT_RIGHT), transform_annotation(annotation, 'mirror'))


def resize_image(sample: Sample, percent: float) -> Sample:
    """
    百分比重采样
    :param sample:
    :param percent:
    :return:
    """
    image = sample.image
    annotation = sample.annotation
    if percent > 1 or percent <= 0:
        raise ValueError('percent must in the range: (0,1]')
    image_width, image_height = image.size
    new_size = (floor(image_width*percent), floor(image_height*percent))
    return Sample(image.resize(new_size), transform_annotation(annotation, 'resize', new_size=new_size))


def square_clip(sample: Sample) -> Sample:
    image = sample.image
    image_width, image_height = image.size
    if image_width == image_height:
        return sample
    annotation = sample.annotation
    x, y = floor(image_width/2), floor(image_height/2)
    new_size = min(image_width, image_height)
    x0, y0 = x-floor(new_size/2), y-floor(new_size/2)
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    clip_box = (x0, y0, x0+new_size, y0+new_size)
    return Sample(image.crop(clip_box), transform_annotation(annotation, 'clip', clip_img_box=clip_box))


def label_format(annotation: Annotation, classes: List[str], format_type: str, skip_aux=True, **kwargs) -> str:
    """
    转换标注格式
    :param annotation:
    :param classes:
    :param format_type:
    :param skip_aux:
    :return:
    """
    if format_type == 'keras-yolo3':
        img_path = kwargs['img_path']
        boxes_str = ''
        for label in annotation.labels:
            if skip_aux and label.auxiliary:
                continue
            class_str = str(classes.index(label.name))
            box = label.get_extents()
            box_str = ' '+str(box.x_min)+','+str(box.y_min)+','+str(box.x_max)+','+str(box.y_max)+','+class_str
            boxes_str = boxes_str+box_str
        return img_path+boxes_str+'\n'

    if format_type == 'darknet':
        dw = 1./annotation.image_width
        dh = 1./annotation.image_height
        boxes_str = ''
        for label in annotation.labels:
            if skip_aux and label.auxiliary:
                continue
            class_str = str(classes.index(label.name))
            box = label.get_extents()
            x = ((box.x_min+box.x_max)/2.0 - 1)*dw
            y = ((box.y_min+box.y_max)/2.0 - 1)*dh
            w = (box.x_max-box.x_min)*dw
            h = (box.y_max-box.y_min)*dh
            box_str = class_str+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n'
            boxes_str = boxes_str+box_str
        return boxes_str

    if format_type == 'test':
        img_path = kwargs['img_path']
        boxes_str = '['
        for label in annotation.labels:
            if skip_aux and label.auxiliary:
                continue
            class_str = str(classes.index(label.name))
            box = label.get_extents()
            bbox = Box(class_str, 1, box.x_min, box.y_min, box.x_max, box.y_max)
            box_str = bbox.to_str()

            boxes_str = boxes_str + box_str + ','

        boxes_str = boxes_str[:-1]
        boxes_str += ']'
        return img_path + '\t' + boxes_str + '\n'

    raise ValueError('format type is not supported')


def save_result(result: Sample, classes, out_path, out_name) -> Sample:
    image = result.image
    annotation = result.annotation
    new_name = annotation.image_name + '_' + out_name
    annotation.image_name = new_name
    image.save(os.path.join(out_path, new_name + '.jpg'))  # 保存图片
    darknet = label_format(annotation, classes, 'darknet')
    kyolo3 = label_format(annotation, classes, 'keras-yolo3',
                          img_path=os.path.join(out_path, new_name + '.jpg'))
    test = label_format(annotation, classes, 'test',
                          img_path=os.path.join(out_path, new_name + '.jpg'))
    with open(os.path.join(out_path, new_name + '.txt'), 'w') as darknet_annotation_file:
        darknet_annotation_file.write(darknet)
    with open(os.path.join(out_path, 'ky_train.txt'), 'a') as yolo_annotation_file:
        yolo_annotation_file.write(kyolo3)
    with open(os.path.join(out_path, 'test.txt'), 'a') as yolo_annotation_file:
        yolo_annotation_file.write(test)
    return result


def enhance_flow(sample_path, name, classes, outpath, img_type: str = '.jpg', an_type: str = 'labelme', square=False):
    origin_sample = load_sample(sample_path, name, img_type, an_type)
    if square:
        origin_sample = square_clip(origin_sample)
    # rotate(8)
    result_r90 = rotate_image2(origin_sample)
    result_r90.annotation.image_name = origin_sample.annotation.image_name+'_r90'

    result_r180 = rotate_image2(result_r90)
    result_r180.annotation.image_name = origin_sample.annotation.image_name + '_r180'

    result_r270 = rotate_image2(result_r180)
    result_r270.annotation.image_name = origin_sample.annotation.image_name + '_r270'

    result_r45 = rotate_image2(origin_sample, 45)
    result_r45.annotation.image_name = origin_sample.annotation.image_name + '_r45'

    result_r135 = rotate_image2(result_r45)
    result_r135.annotation.image_name = origin_sample.annotation.image_name + '_r135'

    result_r225 = rotate_image2(result_r135)
    result_r225.annotation.image_name = origin_sample.annotation.image_name + '_r225'

    result_r315 = rotate_image2(result_r225)
    result_r315.annotation.image_name = origin_sample.annotation.image_name + '_r315'

    inputs = [origin_sample, result_r90, result_r180, result_r270,
              result_r45, result_r135, result_r225, result_r315]

    outputs = []
    # zoom clip by 80%,60%,40%,20% (4)
    outputs_c80 = enhance_unit(inputs,
                               functools.partial(zoom_clip_image, ratio=0.8),
                               classes, outpath, 'c80', False)
    outputs_c60 = enhance_unit(inputs,
                               functools.partial(zoom_clip_image, ratio=0.6, random_shift=True),
                               classes, outpath, 'c60', False)
    outputs_c40 = enhance_unit(inputs,
                               functools.partial(zoom_clip_image, ratio=0.4, random_shift=True),
                               classes, outpath, 'c40', False)
    outputs_c20 = enhance_unit(inputs,
                               functools.partial(zoom_clip_image, ratio=0.2, random_shift=True),
                               classes, outpath, 'c20', False)
    outputs.extend(outputs_c80)
    outputs.extend(outputs_c60)
    outputs.extend(outputs_c40)
    outputs.extend(outputs_c20)

    # mirror(2)
    outputs = enhance_unit(outputs,
                           mirror_image,
                           classes, outpath, 'm')

    # resize(3)
    # outputs_rs1 = enhance_unit(outputs,
    #                            functools.partial(resize_image, percent=0.5),
    #                            classes, outpath, 'rs50', False)
    # outputs_rs2 = enhance_unit(outputs,
    #                            functools.partial(resize_image, percent=0.25),
    #                            classes, outpath, 'rs25', False)
    # outputs.extend(outputs_rs1)
    # outputs.extend(outputs_rs2)
    return outputs


def enhance_unit(inputs, method, classes, out_path, out_name, contain_inputs=True):
    results = [save_result(method(input_), classes, out_path, out_name)
               for input_ in inputs]
    if not contain_inputs:
        return results
    inputs.extend(results)
    return inputs


class EnhanceTask:
    def __init__(self, input_path, output_path, annotation_classes):
        self._input_path = input_path
        self._sample_names = []
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            if os.path.isdir(file_path):
                continue
            if os.path.splitext(file_path)[1] == '.jpg':
                self._sample_names.append(os.path.splitext(file)[0])
        self._output_path = output_path
        self._annotation_classes = annotation_classes

    def run(self, clip=False):
        """
        :param clip: if true, the result images will be clip as square
        :return:
        """
        n = len(self._sample_names)
        for i in range(n):
            sample_name = self._sample_names[i]
            enhance_flow(self._input_path, sample_name, self._annotation_classes, self._output_path, square=clip)
            print('{}/{} finished'.format(i+1, n))


if __name__ == '__main__':
    object_class = ['runway', 'airport']
    _origin_samples_dir = r'D:\Data\sample2\val_sample'
    _out_dir = r'D:\Data\val_sample_enhance'
    t = EnhanceTask(_origin_samples_dir, _out_dir, object_class)
    t.run(True)

    # ----- test -----
    # ts = load_sample(r'D:\Data\sample2\sample_2', '1887')
    # rs = clip_image(ts, 0.6, random_shift=True)
    # rs = rotate_image2(ts, 32)
    # print(rs.image.size)
    # rs = mirror_image(rs)
    # rs = resize_image(rs, 0.5)
    # draw = ImageDraw.Draw(rs.image)
    # for label in rs.annotation.labels:
    #     color = 'blue'
    #     if label.auxiliary:
    #         color = 'red'
    #     print(label.get_extents())
    #     draw.rectangle(label.get_extents(), outline=color, width=10)
    # rs.image.show()
    # ----------------
