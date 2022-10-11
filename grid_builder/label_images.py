# On Mac, run it with DYLD_LIBRARY_PATH=/usr/local/lib python3 label_images.py
import csv
import os

from GridBuilder import Point, Image, Grid

# Coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
LOWER_BOUND = Point(45.6755, 5.7349)
UPPER_BOUND = Point(47.9163, 10.6677)

MIN_POINTS = 100
MAX_POINTS = 1000



def create_labels(input_filename):
    base_name = os.path.splitext(input_filename)[0]

    excluded_ids = []
    excluded_imgs = 'input/' + base_name + '_excluded.csv'

    if(os.path.exists(excluded_imgs)):
        with open(excluded_imgs) as f:
            csv_reader = csv.reader(f, delimiter=',')
            for index, line in enumerate(csv_reader):
                if index > 0:
                    excluded_ids.append(line[0])

    output_grid_file = 'output/' + base_name + "_grid.csv"
    output_label_file = 'output/' + base_name + "_label.csv"

    images = []

    with open('input/' + input_filename) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for index, line in enumerate(csv_reader):
            if index > 0 and not line[0] in excluded_ids:
                point = Point(float(line[3]), float(line[2]))
                images.append(Image(line[0], line[1], point))
            else:
                print(f'Line {index} excluded')


    grid = Grid(images, LOWER_BOUND, UPPER_BOUND, MIN_POINTS, MAX_POINTS)
    grid.buildGrid()
    grid.toWkt(output_grid_file)

    header = ['# ID', 'Label']
    labels = [[image.id, image.label] for image in images if image.label != None]

    with open(output_label_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(labels)


if __name__ == '__main__':

    create_labels('flickr_images.csv')
    #create_labels('geotags_185K.csv')
