from dataclasses import dataclass
from collections import namedtuple
from typing import List

from PIL import Image


Point = namedtuple('Point', ['x', 'y'])
Extents = namedtuple('Extents', ['x_min', 'y_min', 'x_max', 'y_max'])


@dataclass
class Label:
    name: str
    shapes: List[Point]
    auxiliary: bool = False

    def get_extents(self):
        x = [p.x for p in self.shapes]
        y = [p.y for p in self.shapes]
        x.sort()
        y.sort()
        x_min, x_max, y_min, y_max = x[0], x[-1], y[0], y[-1]
        return Extents(x_min, y_min, x_max, y_max)


@dataclass
class Annotation:
    image_name: str
    image_width: int
    image_height: int
    labels: List[Label]


@dataclass
class Sample:
    image: Image
    annotation: Annotation


@dataclass
class Box:
    type: str
    a_score: float
    left: int
    top: int
    right: int
    bottom: int

    def get_extents(self):
        return self.left, self.top, self.right, self.bottom

    def to_str(self):
        ss = '{"type": "'+self.type+'", "score": '+str(self.a_score) + \
             ', "extents": ['+str(self.left)+', '+str(self.top)+', '+str(self.right)+', '+str(self.bottom)+']}'
        return ss
