import json
import os
import argparse
from config import *
import subprocess
import pandas as pd
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def store_image_results(directory, predictions):
    
    src_dir = os.path.join(directory)
    dest_dir = os.path.join(args.data,'outputs/')
    
    if not (os.path.exists(dest_dir)):
        os.makedirs(dest_dir)
    
    for file in os.listdir(src_dir):
        image = cv2.imread(src_dir + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        font = ImageFont.truetype('./data/font.ttf', size=40, encoding='unic')
        draw = ImageDraw.Draw(pil_image)
        draw.text((10, 10), predictions[file], font=font, fill=(255, 0, 0))
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(dest_dir + file, cv2_image)



def get_machine_translations(directory, predictions):
    
    tokenizer = AutoTokenizer.from_pretrained("salesken/translation-hi-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("salesken/translation-hi-en")

    for file in os.listdir(directory):
        hin_snippet = predictions[file]
        inputs = tokenizer.encode(hin_snippet, return_tensors="pt",padding=True,max_length=512,truncation=True)
        outputs = model.generate(inputs, max_length=128, num_beams=None, early_stopping=True)
        translated = tokenizer.decode(outputs[0]).replace('<pad>',"").strip().lower()
        print(translated)
    # due to covid, we have reduced our debt interest




def parse_args():
    
    '''
    Input arguments
    data: Data folder path to run test images (images should be present in images folder inside the data folder)
    lang: Language of the images for which recognition model of that language would be run
    '''
    parser = argparse.ArgumentParser(description="Test run", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--data", type=str, dest='data', default='./data/', help="path to the input data file")
    parser.add_argument("--lang", type=str, dest='lang', default='telugu', help="language of images")
    parser.add_argument("--model", type=str, dest='model', default=RESUME_FILE, help="path to the trained model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    ### Dataset Structure Creation
    files = os.listdir(os.path.join(args.data, 'images'))
    predictions = {}
    for file in files:
        predictions[file] = ""
    
    ### Empty Labels json file creation
    with open(os.path.join(args.data, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
        
    ### Run the test example on sample data and get predictions
    command = ["python", "./doctr/references/recognition/train_pytorch.py", ARCH, "--train_path", TRAIN_PATH + LANGUAGE + "/", "--epochs", str(1), "--device", str(0),  "-b", str(1),  "--test-only",
               "--val_path", args.data, "--vocab", args.lang, "--resume", args.model]
    
    subprocess.run(command)
    
    
    ### Store the results in the test folder
    df = pd.read_csv('./results/' + args.lang + '_results.csv')
    index = 0
    for file in files:
        predictions[file] = df['pred'][index]
        index+=1
        
    ### Prediction Labels json file updation
    with open(os.path.join(args.data, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
        
        
    # ### Store results in images
    store_image_results(os.path.join(args.data, 'images/'), predictions)

    if(args.lang == 'devanagari'):
        translations = get_machine_translations(predictions)
    
        
    
    
    
    
                
        
    
    
    