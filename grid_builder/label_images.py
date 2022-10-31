# On Mac, run it with DYLD_LIBRARY_PATH=/usr/local/lib python3 label_images.py
import csv
import os

import s2geometry.pywraps2 as s2
from GridBuilder import Point, Image, Grid, convert_images_for_grid

from grid_builder.flickr_search_images import read_cvs_file, read_excluded_file, read_validated_file


class LabelHelper:

    def __init__(self, file_name: str=None):

        self.__Labels = []
        self.__all_labels = set()
        self.__all_cellIds = set()

        if file_name is None:
            file_name = os.path.dirname(os.path.realpath(__file__))+'/output/grid_cellIds.csv'

        with open(file_name, 'r', encoding='UTF8', newline='') as file:
            reader = csv.reader(file)

            for tokens in reader:

                if len(tokens) == 0 or tokens[0].startswith('#'):
                    continue

                label = tokens[0]
                id = tokens[1]

                if label in self.__all_labels:
                    raise RuntimeError(f'Duplicated label {label}')

                if id in self.__all_cellIds:
                    raise RuntimeError(f'Duplicated cellId {id}')

                self.__all_labels.add(label)
                self.__all_cellIds.add(id)

                cellId = s2.S2CellId(int(id))

                self.__Labels.append((label, cellId))

    def get_label(self, image: Image) -> str:

        for label, cell in self.__Labels:

            if cell.contains(image.coords.cellId):
                return label

        return None

    def get_cellId(self, data_label: str) -> s2.S2CellId:

        for label, cell in self.__Labels:

            if label == data_label:
                return cell

        return None


def create_labels(dataset_name):

    # read label files
    label_helper = LabelHelper()

    # read and convert datapoints
    data_points = convert_images_for_grid(dataset_name)

    # get labels fro the datapoints
    for image in data_points:
        label = label_helper.get_label(image)
        image.label = label

    output_label_file = 'output/' + dataset_name + "_label.csv"
    header = ['# ID', 'Label']
    labels = []
    labels_map = {}

    for data in data_points:
        if data.label is None:
            continue

        if data.label not in labels_map:
            labels_map[data.label] = []

        labels.append([data.id, data.label])
        labels_map[data.label].append(data.id)

    print(f'Labeled {len(labels)} of {len(data_points)} datapoints ({len(data_points) - len(labels)} missed)')

    for label in labels_map:
        cellId = label_helper.get_cellId(label)
        id = cellId.id()
        face = cellId.face()
        level = cellId.level()
        datas = labels_map[label]
        print(f'Label: {label} (id: {id} level: {level} face: {face}) has \t\t {len(datas)} datapoints')

    with open(output_label_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(labels)


if __name__ == '__main__':
    create_labels('flickr_images')
    #create_labels('geotags_reconstructed')
    #create_labels('geotags_185K')
