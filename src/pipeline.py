from doctr.models import ocr_predictor
from doctr.io import DocumentFile

from transformers import PegasusForConditionalGeneration, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

import cv2
import re
import argparse
import warnings
import pickle
import shutil
import numpy as np
from pathlib import Path
from config import *
# from nltk.tokenize import  sent_tokenize


from temp.tts_infer.tts import TextToMel, MelToWav
from temp.tts_infer.transliterate import XlitEngine
from temp.tts_infer.num_to_word_on_sent import normalize_nums


from scipy.io.wavfile import write
from PIL import ImageFont, ImageDraw, Image
import IPython.display as ipd



warnings.filterwarnings("ignore")
device = 'cpu'

text_to_mel = TextToMel(glow_model_dir=GLOW_MODEL_DIR, device=device)
mel_to_wav = MelToWav(hifi_model_dir=HIFI_MODEL_DIR, device=device)


def get_command():
    parser = argparse.ArgumentParser(description="Test run", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--data", type=str, dest='data', default=None, help="path to the input data file")
    parser.add_argument("--old", action='store_true', dest='new')
    parser.add_argument("--det_model", type=str, dest='det_model', default='db_resnet50', help="Detection model name")
    parser.add_argument("--rec_model", type=str, dest='rec_model', default='crnn_vgg16_bn', help="Recognition model name")
    args = parser.parse_args()
    return args


def translit(text, lang):
    reg = re.compile(r'[a-zA-Z]')
    engine = XlitEngine(lang)
    words = [engine.translit_word(word, topk=1)[lang][0] if reg.match(word) else word for word in text.split()]
    updated_sent = ' '.join(words)
    return updated_sent
    
def run_tts(text, lang):
    text = text.replace('।', '.') # only for hindi models
    text_num_to_word = normalize_nums(text, lang) # converting numbers to words in lang
    text_num_to_word_and_transliterated = translit(text_num_to_word, lang) # transliterating english words to lang
    
    mel = text_to_mel.generate_mel(text_num_to_word_and_transliterated)
    audio, sr = mel_to_wav.generate_wav(mel)
    write(filename='temp.wav', rate=sr, data=audio) # for saving wav file, if needed
    return (sr, audio)



def preprocess_data(total_data):

    final_data = []

    for data in total_data:

        brac_text_pattern = r"[\(\[].*?[\)\]]"
        brac_pattern = r'[()\[\]{}]'
        eqn_pattern = r'\b[\w\.\-]+\s*[+\-\.]\s*[\w\.\-]+\s*=\s*[\w\.\-]+\b'
        space_pattern = r'[/\s]{2,}'

        data = data.replace("- ", "")
        data = re.sub(brac_text_pattern, "", data)
        data = re.sub(brac_pattern, '', data)
        data = re.sub(eqn_pattern, "", data, 0, re.MULTILINE)
        data = re.sub(space_pattern, ' ', data)
        final_data.append(data)

    return final_data


def get_summary(data):
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    model = PegasusForConditionalGeneration.from_pretrained(SUMMARIZATION_MODEL).to(device)
    batch = tokenizer(data, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch, max_new_tokens=MAX_VALUE)
    summary_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return summary_text


def summarize(data):


    total_data = preprocess_data(data)

    total_summary = ""
    
    for data in total_data:

        summary_text = get_summary(data)
        # print("Page Summary: ", summary_text)
        if not (isinstance(summary_text, str)):
            summary_text = " ".join(summary_text)
        summary_text = summary_text.replace("<n>", "\n")
        print
        total_summary += summary_text + "\n"

    print(total_summary)

    return total_summary



def get_speech(file, data):
    _, audio = run_tts(data, 'hi')
    ipd.Audio('temp.wav')
    shutil.move('temp.wav', './results/speech/' + file + '.wav')



def machine_translation(data):

    result = ""

    for eng_text in data:

        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL, src_lang="eng_Latn", tgt_lang="hin_Deva")
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
        translator = pipeline('translation', model=model, tokenizer=tokenizer,src_lang="eng_Latn", tgt_lang="hin_Deva",device=device) 
        output = translator(eng_text,max_length=MAX_VALUE)
        result += output[0]['translation_text'] + "\n"

    return result

def get_ocr_output(args):

    if (args.new==False):
        model = ocr_predictor(det_arch=args.det_model, reco_arch=args.rec_model, pretrained=True)

        if args.data.endswith('.pdf'):
            doc = DocumentFile.from_pdf(args.data)
        else:
            doc = DocumentFile.from_images(args.data)

        result = model(doc)

        file = Path(args.data).stem

        with open('results/int_ocr/' + file + '.pkl', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(result, outp, pickle.HIGHEST_PROTOCOL)

    else:
         with open('results/int_ocr/' + file + '.pkl', 'rb') as f:
            result = pickle.load(f)


    print("No. of pages: ", len(result.pages))
    data = []
    for page in result.pages:
        output = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    value = word.value
                    output.append(value)
        output = " ".join(output)
        data.append(output)

    return data


if __name__ =="__main__":
    args = get_command()

    print("#### Started OCR of the input documents ####")
    output = get_ocr_output(args)
    print("#### Done with OCR, OCR output is stored in results directory for reference")
    
    print("\n\n##### Starting Summarization of OCRed Data #####")
    data = summarize(output)
    # data = " ".join(data)
    print("#### Done with Summarization #####")

    print("---- Following is the summary of above document ----")
    print(data)

    file = Path(args.data).stem
    file1 = open("./results/summary/"+ file+'.txt',"w")
    
    file1.write(data)
    file1.close() 

    with open("./results/summary/"+ file+'.txt') as f:
        data = f.readlines()


    print("\n\n")


    print("#### Starting Translation ####")

    output = machine_translation(data)

    print("\n")
    print(output)

    
    file1 = open("./results/translation/"+ file+'.txt',"w")
    
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
    
    cv2.imwrite("./results/translation/"+ file+'.jpg', cv2_image)


    get_speech(file, output)

    print("Done")
    



