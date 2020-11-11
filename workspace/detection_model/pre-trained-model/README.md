## Pre-trained Model
1. Download YOLOv3 weights from [YOLOv3 website](https://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
  
    wget https://pjreddie.com/media/files/yolov3.weights  
    python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5  
  
More information [here](https://github.com/qqwweee/keras-yolo3).