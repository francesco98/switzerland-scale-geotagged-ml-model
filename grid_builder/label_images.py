# On Mac, run it with DYLD_LIBRARY_PATH=/usr/local/lib python3 label_images.py
import csv
import os

from GridBuilder import Point, Image, Grid

# Coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
from grid_builder.flickr_search_images import read_cvs_file, read_excluded_file, read_validated_file

LOWER_BOUND = Point(45.6755, 5.7349)
UPPER_BOUND = Point(47.9163, 10.6677)

MIN_POINTS = 100
MAX_POINTS = 1000



def create_labels(dataset_name):

    images = read_cvs_file(f'input/{dataset_name}.csv')
    excluded = read_excluded_file(f'input/{dataset_name}_excluded.csv')
    validated = read_validated_file(f'input/{dataset_name}_validated.csv')

    output_grid_file = 'output/' + dataset_name + "_grid.csv"
    output_label_file = 'output/' + dataset_name + "_label.csv"

    data_points = []

    for id in images:

        if id in excluded:
            continue

        if id not in validated:
            continue

        elem = images[id]
        url = elem['url']
        longitude = float(elem['longitude'])
        latitude = float(elem['latitude'])

        point = Point(latitude, longitude)
        data_points.append(Image(id, url, point))


    grid = Grid(data_points, LOWER_BOUND, UPPER_BOUND, MIN_POINTS, MAX_POINTS)
    grid.buildGrid()
    grid.toWkt(output_grid_file)

    header = ['# ID', 'Label']
    labels = [[data.id, data.label] for data in data_points if data.label != None]

    with open(output_label_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(labels)


if __name__ == '__main__':
    create_labels('flickr_images')
    #create_labels('geotags_reconstructed')
    #create_labels('geotags_185K')
