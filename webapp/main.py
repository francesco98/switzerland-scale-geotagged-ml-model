import argparse
import base64
import json
import os
import time
from functools import wraps
from urllib.parse import urlparse

import numpy as np
import torch
from flask import render_template, redirect, url_for, session, flash, Flask, request

from datetime import date

from grid_builder.LabelBuilder import LabelBuilder
from models_pytorch.dataset import ImageGeolocationDataset, DataHelper
from models_pytorch.testing_resnet import prediction
from models_pytorch.utils import get_model, create_datahelper


# globals
from visualization.DisplayHelper import Display, LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, IMAGE_FILE_NAME

app = Flask(__name__)

use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
data_helper: DataHelper=None
labelBuilder: LabelBuilder=None
model = None
test_dataset = None
test_dataset_results = None
display: Display=None

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# local helpers


def convert_prediction_result(elem):
    id = elem['id']
    probabilities = elem['probabilities']
    cellId = elem['cellId']
    label = elem['label']
    filename = elem['filename']
    sortedIdx = np.argsort(probabilities)[::-1]  # sorted and reversed oorder
    top3labels = sortedIdx[:3]
    top3Probs = [round(probabilities[i], 2) for i in top3labels]
    entry = []
    entry.append('OK' if top3labels[0] == label else 'NOK')
    entry.append(id)
    entry.append(str(label))
    entry.append(cellId.id())
    entry.append(str(top3labels))
    entry.append(str(top3Probs))
    return entry


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/validation')
def validation():
    predictions = []
    for id in test_dataset_results:
        elem = test_dataset_results[id]
        entry = convert_prediction_result(elem)

        predictions.append(entry)

    return render_template('validation.html', predictions=predictions)




@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/details')
def details():
    id = request.args.get('id', type=str)
    elem = test_dataset_results[id]

    entry = convert_prediction_result(elem)

    probabilities = elem['probabilities']
    cellId = elem['cellId']
    heatmap_buff = display.create_heatmap(probabilities=probabilities, ground_truth=str(cellId.id()))
    heatmap_data = base64.b64encode(heatmap_buff.getbuffer()).decode("ascii")

    image_buff = display.read_data_image(elem['filename'])
    image_data = base64.b64encode(image_buff.getbuffer()).decode("ascii")

    return render_template('details.html', image_heatmap=heatmap_data, image_data=image_data ,elem=entry)


def prepare_model_and_data(args):

    global data_helper, labelBuilder, test_dataset, model, test_dataset_results, display

    args.modelfilename = 'geolocation_cnn_flickr_images_resnet18_04_11_2022.pt'
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

    test_dataset_results = prediction(test_dataset, labelBuilder, model, device)

    display = Display(LOWER_BOUND_IMAGE, UPPER_BOUND_IMAGE, '../visualization/'+IMAGE_FILE_NAME, labelBuilder)


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

    args = parser.parse_args()

    prepare_model_and_data(args)

    app.run(host='localhost', port=8282, ssl_context=None)
