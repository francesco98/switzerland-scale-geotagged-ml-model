import csv

with open('flickr_images.txt') as f:
    data = f.read().splitlines()[1:]

lines = [line.split(',') for line in data]

header = ['id', 'url', 'longitude', 'latitude']
data = []
for line in lines:
    data.append([line[0], line[1], line[2], line[3]])

with open('flickr_images.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)