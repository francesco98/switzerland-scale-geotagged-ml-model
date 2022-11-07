import csv
from io import BytesIO

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle, Wedge, Polygon


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
        self.color_map = mpl.colormaps['Reds']
        self.labelBuilder = labelBuilder
        self.image = Image.open(image_file_name)
        self.dpi = 100

    def create_heatmap(self, probabilities: list=None, ground_truth: str=None, plot=False):

        # What size does the figure need to be in inches to fit the image?
        figsize = (self.image.size[0] / float(self.dpi), self.image.size[1] / float(self.dpi))

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])


        # City of Bern 46.9480° N, 7.4474° E
        ax.scatter(7.4474, 46.9480, zorder=1, c='b', s=20, alpha=0.5)
        # City of Geneva 46.2044° N, 6.1432° E
        ax.scatter(6.1432, 46.2044, zorder=1, c='b', s=20, alpha=0.5)
        # City of Lugano 46.0037° N, 8.9511° E
        ax.scatter(8.9511, 46.0037, zorder=1, c='b', s=20, alpha=0.5)

        if probabilities is None or len(probabilities) != self.labelBuilder.get_num_labels():
            no_colors = True
        else:
            no_colors = False

        alpha = 0.5

        for idx, (label, cell) in enumerate(self.labelBuilder):
            x, y = self.get_cell_points(cell)
            id = cell.id()

            if id == ground_truth:
                edge_color = 'red'
                linewidth = 4.0
            else:
                edge_color = 'black'
                linewidth = None

            if no_colors or probabilities[idx] < 0.05:
                face_color = None
            else:
                val = min(probabilities[idx] + 0.2, 1.0)
                face_color = self.get_cell_color(val)
                #face_color = None



            self.draw_cell_box(id, x, y, alpha, face_color, edge_color, linewidth, ax)


        ax.set_title('Plotting Spatial Data on Switzerland Map')
        ax.set_xlim(self.bbox[0], self.bbox[1])
        ax.set_ylim(self.bbox[2], self.bbox[3])

        # x = [8, 9, 9, 8]
        # y = [46, 46, 47, 47]
        # self.draw_cell_box('test box', x, y, ax)

        ax.imshow(self.image, zorder=0, extent=self.bbox, aspect='equal')

        if plot:
            plt.show()
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi)
            plt.close(fig)
            return buf

    def read_data_image(self, filename):

        data_image = Image.open(filename)

        # What size does the figure need to be in inches to fit the image?
        figsize = (data_image.size[0] / float(self.dpi), data_image.size[1] / float(self.dpi))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(data_image)
        ax.axis('off')
        ax.set_title('Data point')

        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")

        return buf


    def get_cell_color(self, value: float):
        assert 0.0 <= value <= 1.0
        rgb = self.color_map(value)
        return rgb


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


    def draw_cell_box(self, id: str, x: list, y: list, alpha: float, facecolor: str, edgecolor: str, linewidth: float, axes: plt.Axes):
        # order: lower left, lower right, upper right, upper left

        array = np.column_stack((x, y))
        polygon1 = Polygon(array, closed=True, fill=False, edgecolor=edgecolor, linewidth=linewidth)
        axes.patches.append(polygon1)
        if facecolor:
            polygon2 = Polygon(array, closed=True, fill=True, alpha=alpha, facecolor=facecolor, edgecolor=None)
            axes.patches.append(polygon2)

        #print(f'Plotted cell {id} box: x={x} y={y}, alpha={alpha} color={color}')




def show_grid():
    labelBuilder = LabelBuilder('../grid_builder/output/grid_cellIds.csv')
    display = Display(LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, IMAGE_FILE_NAME, labelBuilder)
    probas = [0.0] * labelBuilder.get_num_labels()
    probas[0] = 0.4
    probas[1] = 0.3
    probas[2] = 0.2
    display.create_heatmap(probabilities=probas, ground_truth=None, plot=True)


if __name__ == '__main__':

    show_grid()

