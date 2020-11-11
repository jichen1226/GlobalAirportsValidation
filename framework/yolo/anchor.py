"""
Calculate anchor boxes
"""
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

from framework.yolo.kmeans import kmeans, avg_iou


def load_voc_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) / width
			ymin = int(obj.findtext("bndbox/ymin")) / height
			xmax = int(obj.findtext("bndbox/xmax")) / width
			ymax = int(obj.findtext("bndbox/ymax")) / height

			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)


def load_ky_dataset(path):
	dataset = []
	df = ky_to_csv(path)

	# df.to_csv(path, index=False)
	# df = pd.read_csv(path)

	for i, row in df.iterrows():
		width = row['width']
		height = row['height']
		xmax = int(row['xmax']) / width
		xmin = int(row['xmin']) / width
		ymax = int(row['ymax']) / height
		ymin = int(row['ymin']) / height
		dataset.append([xmax - xmin, ymax - ymin])

	print('success load dataset')
	return np.array(dataset)


def ky_to_csv(ky_path):
	ky_list = []
	with open(ky_path, 'r') as ky_file:
		for line in ky_file.readlines():
			_line = line.strip()
			content = _line.split(' ')
			image_path = content[0]

			for i in range(1, len(content)):
				xmin, ymin, xmax, ymax, class_id = content[i].split(',')

				image = Image.open(image_path)
				width, height = image.size

				value = (image_path, int(class_id), int(xmin), int(ymin), int(xmax), int(ymax), int(width), int(height))
				ky_list.append(value)

	column_name = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']
	out_df = pd.DataFrame(ky_list, columns=column_name)
	print('success to csv')
	return out_df


if __name__ == '__main__':
	# train samples for keras-yolo
	_annotations_path = '../../workspace/detection_model/annotation/ky_train.txt'
	data = load_ky_dataset(_annotations_path)

	_k = 9
	out = kmeans(data, k=_k)

	print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
	print("Boxes:\n {}".format(out))

	ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
	print("Ratios:\n {}".format(sorted(ratios)))
