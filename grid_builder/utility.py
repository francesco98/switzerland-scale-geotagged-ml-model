import os
import csv

CSV_HEADER = ['# id', 'url', 'longitude', 'latitude']

def read_cvs_file(file_name: str):

    images = {}

    # open file for reading
    if not os.path.isfile(file_name):
        with open(file_name, 'x', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)

    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:

            if len(tokens) == 0 or tokens[0].startswith('#'):
                continue

            key = tokens[0]

            if key in images:
                msg = f'ERROR: duplicated image id {key}'
                print(msg)
                raise RuntimeError(msg)

            images[key] = {CSV_HEADER[1]: tokens[1], CSV_HEADER[2]: tokens[2], CSV_HEADER[3]: tokens[3]}

    return images

def read_excluded_file(file_name: str):
    return read_ids_from_file(file_name)


def read_validated_file(file_name: str):
    return read_ids_from_file(file_name)


def read_ids_from_file(file_name: str):
    ids = {}

    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:
            if len(tokens) == 0 or tokens[0].startswith('#'):
                continue

            key = tokens[0]
            if key in ids:
                msg = f'ERROR: duplicated image id {key} in {file_name}'
                print(msg)
                raise RuntimeError(msg)

            ids[key] = tokens[1]

        return ids


def read_labels_file(file_name: str):

    labels = {}
    with open(file_name, 'r', encoding='UTF8', newline='') as file:
        reader = csv.reader(file)

        for tokens in reader:

            if len(tokens) == 0 or tokens[0].startswith('#'):
                continue

            key = tokens[0]

            if key in labels:
                msg = f'ERROR: duplicated image id {key}'
                print(msg)
                raise RuntimeError(msg)

            labels[key] = int(tokens[2])

    return labels
