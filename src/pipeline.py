from doctr.models import ocr_predictor
from doctr.io import DocumentFile

from transformers import PegasusForConditionalGeneration, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np

import os
import argparse
import warnings
import pickle

warnings.filterwarnings("ignore")




def get_command():
    parser = argparse.ArgumentParser(description="Test run", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--data", type=str, dest='data', default=None, help="path to the input data file")
    parser.add_argument("--old", action='store_true', dest='new')
    parser.add_argument("--det_model", type=str, dest='det_model', default='db_resnet50', help="Detection model name")
    parser.add_argument("--rec_model", type=str, dest='rec_model', default='crnn_vgg16_bn', help="Recognition model name")
    args = parser.parse_args()
    return args


def preprocess_data(data):
    return data


def summarize(data):


    data = preprocess_data(data)
    
    
    model_name = 'google/pegasus-cnn_dailymail'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    batch = tokenizer(data, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch, max_new_tokens=64)
    summary_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return summary_text


def machine_translation(data):


    model_name = 'facebook/nllb-200-distilled-600M'
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", tgt_lang="hin_Deva")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    translator = pipeline('translation', model=model, tokenizer=tokenizer,src_lang="eng_Latn", tgt_lang="hin_Deva",device=device) 
    output = translator(data,max_length=256)

    return output[0]['translation_text']



def get_ocr_output(args):

    if (args.new==False):

        model = ocr_predictor(det_arch=args.det_model, reco_arch=args.rec_model, pretrained=True)

        if args.data.endswith('.pdf'):
            doc = DocumentFile.from_pdf(args.data)
        else:
            doc = DocumentFile.from_images(args.data)

        result = model(doc)

        with open('results/test_ocr_3.pkl', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(result, outp, pickle.HIGHEST_PROTOCOL)

    else:
         with open('results/test_ocr_3.pkl', 'rb') as f:
            result = pickle.load(f)


    print("No. of pages: ", len(result.pages))
    output = []
    for page in result.pages:
        # value = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    value = word.value
                    output.append(value)
    output = " ".join(output)
    # print(output)

    return output


if __name__ =="__main__":
    args = get_command()

    print("#### Started OCR of the input documents ####")
    output = get_ocr_output(args)
    print("#### Done with OCR, OCR output is stored in results directory for reference")
    
    print("\n\n##### Starting Summarization of OCRed Data #####")
    data = summarize(output)
    data = " ".join(data)
    print("#### Done with Summarization #####")

    print("---- Following is the summary of above document ----")
    print(data)


    print("\n\n")


    print("#### Starting Translation ####")

    output = machine_translation(data)

    print("\n")
    print(output)

    file1 = open("myfile.txt","w")
    
    file1.write(output)
    file1.close() #to


    print("\n\n")

    print("Storing output in image")

    output = output.replace("।","।\n")

    blank_image = Image.new('RGB', (1024, 512))
    font = ImageFont.truetype('convincingDirectory/font.ttf', size=40, encoding='unic')
    draw = ImageDraw.Draw(blank_image)
    draw.text((10, 10), output, font=font, fill=(255, 0, 0))
    cv2_image = cv2.cvtColor(np.array(blank_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("temp.jpg", cv2_image)

    print("Done")
    



