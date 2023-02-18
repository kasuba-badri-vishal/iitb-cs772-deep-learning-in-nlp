import os

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from ocr.config import OCR_DATA_PATH



image_data_dir = os.listdir(OCR_DATA_PATH)


model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
print("loaded model")


for file in image_data_dir:
    img_doc = DocumentFile.from_images(OCR_DATA_PATH + file)
    result = model(img_doc)
    json_output = result.export()
    print(json_output)
