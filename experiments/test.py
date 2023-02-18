import os
import pickle
import pandas as pd
from tqdm import tqdm

from doctr.models import ocr_predictor
from doctr.io import DocumentFile

OCR_DATA_PATH  = './../../data/ocr/docbank/images/'
TXT_DATA_PATH  = './../../results/ocr/linknet_master/txt/'
image_data_dir = os.listdir(OCR_DATA_PATH)

# model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='crnn_mobilenet_v3_large', pretrained=True)
model = ocr_predictor(det_arch='linknet_resnet18', reco_arch='vitstr_base', pretrained=True)