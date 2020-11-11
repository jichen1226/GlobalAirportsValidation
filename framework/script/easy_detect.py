import os
from timeit import default_timer as timer

from PIL import Image, ImageDraw

from framework.yolo import predict
from framework.yolo.evaluate import init_yolo


def image(image, yolo, out_path=None):
    r = yolo.detect_image(image)
    if out_path:
        r.save(out_path)
    return r


def images(image_dir, yolo, out_dir, image_ext='.jpg'):
    for i, file in enumerate(os.listdir(image_dir)):
        start = timer()

        file_path = os.path.join(image_dir, file)
        if os.path.isdir(file_path):
            continue
        if os.path.splitext(file_path)[1] == image_ext:
            img = Image.open(file_path)
            image(img, yolo, os.path.join(out_dir, file))

        end = timer()
        print('[{} use time:{}s]'.format(i + 1, end - start))


if __name__ == '__main__':
    _iamge_path = r'D:\Data\test\speed_test\oap_5528.jpg'

    _model_path = r'D:\Workspace\python\GlobalAirportsValidation\workspace\detection_model\out\trained_weights_final.h5'
    _anchor_path = r'D:\Workspace\python\GlobalAirportsValidation\workspace\detection_model\model_data\yolo_anchors.txt'
    _class_path = r'D:\Workspace\python\GlobalAirportsValidation\workspace\detection_model\model_data\object_classes.txt'
    _yolo = init_yolo(_model_path, _anchor_path, _class_path)

    _r_img = image(Image.open(_iamge_path), _yolo)
    _r_img.show()
