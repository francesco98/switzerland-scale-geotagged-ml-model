# RUN USING DYLD_LIBRARY_PATH=/usr/local/lib

import csv
from GridBuilder import Point, Image, Grid

# Coordinates bounding boy switzerland according https://giswiki.hsr.ch/Bounding_Box
LOWER_BOUND = Point(45.6755, 5.7349)
UPPER_BOUND = Point(47.9163, 10.6677)

MIN_POINTS = 100
MAX_POINTS = 1000

INPUT_FILE = "./input/geotags_233K.csv"

OUTPUT_GRID_FILE = "./output/geotags_233K_grid.csv"
OUTPUT_LABEL_FILE = "./output/geotags_233K_label.csv"

images = []

with open(INPUT_FILE) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for index, line in enumerate(csv_reader):
        if index > 0:
            point = Point(float(line[3]), float(line[2]))
            images.append(Image(line[0], line[1], point))


grid = Grid(images, LOWER_BOUND, UPPER_BOUND, MIN_POINTS, MAX_POINTS)
grid.buildGrid()
grid.toWkt(OUTPUT_GRID_FILE)

header = ['ID', 'Label']
labels = [[image.id, image.label] for image in images if image.label != None]

with open(OUTPUT_LABEL_FILE, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(labels)