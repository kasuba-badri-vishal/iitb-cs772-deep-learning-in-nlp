from config import *
import shutil

import IPython.display as ipd
from pathlib import Path
from scipy.io.wavfile import write
import re

from temp.tts_infer.tts import TextToMel, MelToWav
from temp.tts_infer.transliterate import XlitEngine
from temp.tts_infer.num_to_word_on_sent import normalize_nums

import sys



device = 'cpu'

text_to_mel = TextToMel(glow_model_dir=GLOW_MODEL_DIR, device=device)
mel_to_wav = MelToWav(hifi_model_dir=HIFI_MODEL_DIR, device=device)


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


def get_speech(file, data):
    _, audio = run_tts(data, 'hi')
    ipd.Audio('temp.wav')
    shutil.move('temp.wav', './results/speech2/' + file + '.wav')



file = Path(sys.argv[1]).stem
get_speech(file, sys.argv[1])