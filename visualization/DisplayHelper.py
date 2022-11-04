import csv

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.patches as patches


import s2geometry.pywraps2 as s2

from grid_builder.LabelBuilder import LabelBuilder
from grid_builder.GridBuilder import Point, Grid

# bounding box of the image switzerland_2.png
# Not the same as the bounding box from the GridBuild class !!!#
# downloaded from https://www.openstreetmap.org/export with the following bpinding box
min_longitude = 5.4380
max_longitude = 10.6000
min_latitude = 45.6000
max_latitude = 47.6430

LOWER_BOUND_IMAGE = Point(min_latitude, min_longitude)
UPPER_BOUND_IMAGE = Point(max_latitude, max_longitude)
IMAGE_FILE_NAME = 'switzerland_2.png'


class Display:
    def __init__(self, lower_bound: Point, upper_bound: Point, image_file_name: str, labelBuilder: LabelBuilder):
        self.bbox = (lower_bound.lng, upper_bound.lng, lower_bound.lat, upper_bound.lat)
        self.image = mpimg.imread(image_file_name)
        self.color_map = mpl.colormaps['viridis']
        self.labelBuilder = labelBuilder

    def create_heatmap(self, probabilities: list=None, ground_truth: str=None, file_name :str=None):

        fig, ax = plt.subplots(figsize=(30, 20))

        # City of Bern 46.9480° N, 7.4474° E
        ax.scatter(7.4474, 46.9480, zorder=1, c='b', s=20, alpha=0.5)
        # City of Geneva 46.2044° N, 6.1432° E
        ax.scatter(6.1432, 46.2044, zorder=1, c='b', s=20, alpha=0.5)
        # City of Lugano 46.0037° N, 8.9511° E
        ax.scatter(8.9511, 46.0037, zorder=1, c='b', s=20, alpha=0.5)

        if probabilities is None or len(probabilities) != len(self.cellIds):
            no_colors = True
        else:
            no_colors = False

        alpha = 0.5
        for idx, (label, cell) in enumerate(self.labelBuilder):
            x, y = self.get_cell_points(cell)
            id = str(cell.id())

            edge_color = 'red' if id is ground_truth else 'black'
            face_color = self.get_cell_color(probabilities[idx]) if not no_colors else None

            self.draw_cell_box(id, x, y, alpha, face_color, edge_color, ax)


        ax.set_title('Plotting Spatial Data on Switzerland Map')
        ax.set_xlim(self.bbox[0], self.bbox[1])
        ax.set_ylim(self.bbox[2], self.bbox[3])

        # x = [8, 9, 9, 8]
        # y = [46, 46, 47, 47]
        # self.draw_cell_box('test box', x, y, ax)

        ax.imshow(self.image, zorder=0, extent=self.bbox, aspect='equal')

        if file_name is None:
            plt.show()
        else:
            print(f'Saved image inb {file_name}')
            plt.savefig(file_name)


    def get_cell_color(self, value: float):
        assert 0.0 <= value <= 0.1
        idx = int( (self.color_map.N - 1) * value)
        return self.color_map.colors[idx]


    def get_cell_points(self,cellId: s2.S2CellId):
        x_list = []
        y_list = []

        cell = s2.S2Cell(cellId)
        # order: lower left, lower right, upper right, upper left
        # x, y
        for i in range(4):
            vertex = cell.GetVertex(i)
            latlng = s2.S2LatLng(vertex)
            x, y = latlng.lng().degrees(), latlng.lat().degrees()

            # assert self.bbox[0] <= x <= self.bbox[1]
            # assert self.bbox[2] <= y <= self.bbox[3]
            x_list.append(x)
            y_list.append(y)

        return x_list, y_list


    def draw_cell_box(self, id: str, x: list, y: list, alpha: float, facecolor, edgecolor, axes: plt.Axes):
        # order: lower left, lower right, upper right, upper left

        array = np.column_stack((x, y))
        polygon1 = Polygon(array, closed=True, fill=False, edgecolor=edgecolor)
        axes.patches.append(polygon1)
        if facecolor:
            polygon2 = Polygon(array, closed=True, fill=True, alpha=alpha, facecolor=facecolor, edgecolor=None)
            axes.patches.append(polygon2)

        #print(f'Plotted cell {id} box: x={x} y={y}, alpha={alpha} color={color}')




def show_grid():
    labelBuilder = LabelBuilder('../grid_builder/output/grid_cellIds.csv')
    display = Display(LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, IMAGE_FILE_NAME, labelBuilder)
    display.create_heatmap(probabilities=None, ground_truth=None, file_name=None)


if __name__ == '__main__':

    show_grid()

