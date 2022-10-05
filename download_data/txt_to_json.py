import json

with open('flickr_images.txt') as f:
    data = f.read().splitlines()[1:]

lines = [line.split(',') for line in data]

json_data = [{'id': line[0], 'url': line[1], 'longitude': float(line[2]), 'latitude': float(line[3])} for line in lines]

with open('flickr_images.json', 'w', encoding='UTF8') as f:
        json.dump(json_data, f)