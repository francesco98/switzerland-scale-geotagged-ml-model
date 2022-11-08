import argparse
import base64
import io
import json
import os
import time
from functools import wraps
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from flask import render_template, redirect, url_for, session, flash, Flask, request

import matplotlib
matplotlib.use('Agg')

from datetime import date

from grid_builder.LabelBuilder import LabelBuilder
from models_pytorch.dataset import ImageGeolocationDataset, DataHelper
from models_pytorch.testing_resnet import predict_dataset, Predictor
from models_pytorch.utils import get_model, create_datahelper


# globals
from visualization.DisplayHelper import Display, LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, IMAGE_FILE_NAME

app = Flask(__name__)

device = torch.device("cpu")
data_helper: DataHelper=None
labelBuilder: LabelBuilder=None
model = None
test_dataset = None
test_dataset_results = None
sorted_test_dataset = None
display: Display=None
predictor: Predictor=None
empty_heatmap=None


# local helpers
def convert_prediction_result(elem, idx):
    id = elem['id']
    probabilities = elem['probabilities']
    cellId = elem['cellId']
    label = elem['label']
    sortedIdx = np.argsort(probabilities)[::-1]  # sorted and reversed oorder
    top3labels = sortedIdx[:3]
    top3Probs = [round(probabilities[i], 2) for i in top3labels]
    entry = []
    entry.append('OK' if top3labels[0] == label else 'NOK') # 0
    entry.append(str(-1)) # 1
    entry.append(id) # 2
    entry.append(label) # 3
    entry.append(cellId.id()) # 4
    entry.append(top3labels) # 5
    entry.append(top3Probs) # 6
    entry.append(idx) # 7
    return entry


def convert_and_sort_results(sort: str, results):

    if sort == 'status':
        reverse = True
        def sort_func(elem):
            return elem[0]

    elif sort == 'target':
        reverse = False
        def sort_func(elem):
            return elem[3]

    elif sort == 'probabilities':
        reverse = True
        def sort_func(elem):
            return elem[6][0] * 100 + elem[6][1] * 10 + elem[6][2]

    else:
        raise RuntimeError(f'Unsupported sorting order {sort}')

    predictions = []
    for idx in range(len(results)):
        elem = results[idx]
        entry = convert_prediction_result(elem,idx)

        predictions.append(entry)

    predictions.sort(reverse=reverse, key=sort_func)

    for idx, entry in enumerate(predictions):
        entry[1] = str(idx)


    return predictions


@app.route('/')
def index():
    return render_template('index.html', num_cells=labelBuilder.get_num_labels(),image_heatmap=empty_heatmap)


@app.route('/validation')
def validation():
    global sorted_test_dataset

    sorting = request.args.get('sort', type=str)
    sorted_test_dataset = convert_and_sort_results(sorting, test_dataset_results)

    return render_template('validation.html', predictions=sorted_test_dataset, total=len(sorted_test_dataset))




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global predictor

    heatmap_data = None
    image_data = None
    elem = None

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' in request.files:
            file = request.files['file']
            file_content = file.read()

            # display image
            data_image = Image.open(io.BytesIO(file_content))
            image_buff = display.create_data_image(data_image)
            image_data = base64.b64encode(image_buff.getbuffer()).decode("ascii")

            # predict geolocation
            probabilities = predictor.predict_image(io.BytesIO(file_content))

            sortedIdx = np.argsort(probabilities)[::-1]  # sorted and reversed oorder
            top3labels = sortedIdx[:3]
            top3Probs = [round(probabilities[i], 2) for i in top3labels]
            elem = [top3labels, top3Probs]

            # display heatmap
            heatmap_buff = display.create_heatmap(probabilities=probabilities, ground_truth=None)
            heatmap_data = base64.b64encode(heatmap_buff.getbuffer()).decode("ascii")


    return render_template('upload.html', image_heatmap=heatmap_data, image_data=image_data, elem=elem)


@app.route('/details')
def details():
    global sorted_test_dataset, test_dataset_results

    idx = request.args.get('idx', type=int)
    entry = sorted_test_dataset[idx]
    elem = test_dataset_results[entry[7]]

    cellId = elem['cellId']
    probabilities = elem['probabilities']
    filename = elem['filename']
    heatmap_buff = display.create_heatmap(probabilities=probabilities, ground_truth=cellId.id())
    heatmap_data = base64.b64encode(heatmap_buff.getbuffer()).decode("ascii")

    data_image = Image.open(filename)
    image_buff = display.create_data_image(data_image)
    image_data = base64.b64encode(image_buff.getbuffer()).decode("ascii")

    prev_idx = idx == 0 if idx == 0 else idx - 1
    next_idx = idx if idx == len(sorted_test_dataset) - 1 else idx + 1


    return render_template('details.html', image_heatmap=heatmap_data, image_data=image_data,
                           elem=entry, next_idx=next_idx, prev_idx=prev_idx,
                           total=len(test_dataset_results))


def prepare_model_and_data(args):

    global data_helper, labelBuilder, test_dataset, model, test_dataset_results, display, predictor, empty_heatmap

    # args.modelfilename = 'geolocation_cnn_flickr_images_resnet18_04_11_2022.pt'
    print(f'Commandline args: {args}')
    print(f'Device: {device}')

    data_helper = create_datahelper(args.dataset, args.seed)
    labelBuilder = LabelBuilder()
    test_dataset = ImageGeolocationDataset(data_helper.test_data, augumentation=False)

    num_classes = labelBuilder.get_num_labels()
    data, label = test_dataset[0]
    input_shape = data.shape

    model = get_model(args.model, device, num_classes, input_shape)
    if not os.path.exists(args.modelfilename):
        raise RuntimeError(f'Modelfile {args.modelfilename} not found')

    model.load_state_dict(torch.load(args.modelfilename, map_location=device))
    model.eval()

    predictor = Predictor(model, device)

    test_dataset_results = predict_dataset(dataset=test_dataset, labelBuilder=labelBuilder, model=model, device=device, max_limit=args.max_testdata)

    display = Display(LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, '../visualization/'+IMAGE_FILE_NAME, labelBuilder)

    heatmap_buff = display.create_heatmap(probabilities=None, ground_truth=None)
    empty_heatmap = base64.b64encode(heatmap_buff.getbuffer()).decode("ascii")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Geolocation classifier')

    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='pretrained model, supported  values: resnet18, resnet101 (default: resnet18)')
    parser.add_argument('--modelfilename', type=str, metavar='N',
                        help='names of the saved model file')
    parser.add_argument('--dataset', type=str, default='flickr_images', metavar='N',
                        help='dataset, supported are geotags_185K or flickr_images (default: flickr_images)')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')

    parser.add_argument('--max-testdata', type=int, default=500, metavar='S',
                        help='maximum test data (default: 500)')

    parser.add_argument('--host', type=str, default='localhost', metavar='N',
                        help='Listener host (default: localhost)')
    parser.add_argument('--port', type=int, default=8282, metavar='N',
                        help='Listener port (default: 8282)')

    args = parser.parse_args()

    prepare_model_and_data(args)

    print(f'Starting on {args.host}:{args.port}')
    app.run(host=args.host, port=args.port, ssl_context=None)
