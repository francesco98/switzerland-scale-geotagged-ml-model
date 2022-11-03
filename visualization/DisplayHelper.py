import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.patches as patches


import s2geometry.pywraps2 as s2


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
    def __init__(self, lower_bound: Point, upper_bound: Point, image_file_name: str, grid_file_name: str):
        self.bbox = (lower_bound.lng, upper_bound.lng, lower_bound.lat, upper_bound.lat)
        self.image = mpimg.imread(image_file_name)
        self.cellIds = []

        ids = set()
        with open(grid_file_name, 'r', encoding='UTF8', newline='') as file:
            reader = csv.reader(file)

            for tokens in reader:

                if len(tokens) == 0 or tokens[0].startswith('#'):
                    continue

                label = tokens[0]
                id = tokens[1]

                if id in ids:
                    raise RuntimeError(f'Duplicated cellId {id}')
                ids.add(id)

                cellId = s2.S2CellId(int(id))

                self.cellIds.append(cellId)


    def create_heatmap(self, probabilities: list=None):

        fig, ax = plt.subplots(figsize=(30, 20))

        # City of Bern 46.9480° N, 7.4474° E
        ax.scatter(7.4474, 46.9480, zorder=1, c='b', s=10, alpha=0.5)
        # City of Geneva 46.2044° N, 6.1432° E
        ax.scatter(6.1432, 46.2044, zorder=1, c='b', s=10, alpha=0.5)
        # City of Lugano 46.0037° N, 8.9511° E
        ax.scatter(8.9511, 46.0037, zorder=1, c='b', s=10, alpha=0.5)

        if probabilities is None or len(probabilities) != len(self.cellIds):
            no_colors = True
        else:
            no_colors = False

        alpha = 0.5
        for idx, cell in enumerate(self.cellIds):
            color = self.get_color(probabilities[idx]) if not no_colors else None
            x, y = self.get_cell_points(cell)
            self.draw_cell_box(cell.id(), x, y, alpha, color,  ax)


        ax.set_title('Plotting Spatial Data on Switzerland Map')
        ax.set_xlim(self.bbox[0], self.bbox[1])
        ax.set_ylim(self.bbox[2], self.bbox[3])

        # x = [8, 9, 9, 8]
        # y = [46, 46, 47, 47]
        # self.draw_cell_box('test box', x, y, ax)

        ax.imshow(self.image, zorder=0, extent=self.bbox, aspect='equal')

        plt.show()


    def get_cell_points(self,cellId: s2.S2CellId):
        x_list = []
        y_list = []

        cell = s2.S2Cell(cellId)
        for i in range(4):
            vertex = cell.GetVertex(i)
            latlng = s2.S2LatLng(vertex)
            x, y = latlng.lng().degrees(), latlng.lat().degrees()

            # assert self.bbox[0] <= x <= self.bbox[1]
            # assert self.bbox[2] <= y <= self.bbox[3]
            x_list.append(x)
            y_list.append(y)

        return x_list, y_list


    def draw_cell_box(self, id: str, x: list, y: list, alpha: float, color, axes: plt.Axes):
        # order: lower left, lower right, upper right, upper left
        # x, y

        #axes.scatter(x, y, zorder=1, c='r', s=10, alpha=0.5)
        array = np.column_stack((x, y))
        polygon1 = Polygon(array, closed=True, fill=False)
        axes.patches.append(polygon1)
        if color:
            polygon2 = Polygon(array, closed=True, fill=True, alpha=alpha, facecolor=color)
            axes.patches.append(polygon2)

        #print(f'Plotted cell {id} box: x={x} y={y}, alpha={alpha} color={color}')




def show_grid():

    display = Display(LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, IMAGE_FILE_NAME, '../grid_builder/output/grid_cellIds.csv')
    display.create_heatmap()


if __name__ == '__main__':

    show_grid()

