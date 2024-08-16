# DocToSpeech [ocr-mt-tts]
Project Work of Creation of efficient software system which reads documents and translates the languages 

Project work done as part of CS772 Course


## ToDos :

1. Learning about the problem statement
2. Literature Survey 
   1. Reading Papers which are releavant and getting the ideas of how they are solved 
   2. Writing summary of appraoches that are done till now for OCR-MT-TTS   
3. Implementation of OCR and Machine Translation and TTS model independently and works for mulitple languages
4. Integration of all the separte models
5. Training or Fine tuning any of the models based on availability of Time


## Time Target:

1. Learning about problem : 01st Feb
2. Literature Survey      : 20th Feb
3. Implementation         : 05th March
4. Integration            : 15th March
5. Training and Finetuning: 25th March
6. Results, Analaysis     : 05th April


## PROPOSED IDEA :-
1. OCR+MT+(Summary in Translated language)
2. OCR+MT+(TTS in Translated language)
3. OCR+MT+(TTS in original language)
4. OCR+MT+(layout preservation in translated language) 


## Datasets :-
1. Docbank(OCR)
2. CFILT dataset(MT, 3GB, Eng-Hin) (https://www.cfilt.iitb.ac.in/iitb_parallel/)
3. Summarization,TTS,layout preservation (pretrained models)
4. dataset for CFLIT(MT(eng to hindi)) - https://drive.google.com/drive/folders/1W1VJr3uzFuFi_49ZbpfeHiIOb_z-EugH?usp=share_link
5. dataset for running current machine translation - https://drive.google.com/drive/folders/18GyLGps2smbpYAtZahdM83qfulcb6-iZ?usp=share_link

## Things we discussed

1. Agenda
2. Time Allocation based on priority
   1.30% OCR (TD - 3, TR - 2)
   2.60% MT (3-4 papers)
   3.10% TTS ( pretrained models) (2)

4. Why we are doing



If managable, we will implement Speech(if time permits)
training model we wont do in speech, if time permits implement speech


## Links to Slides, Presentation

## References

1. [Udaan](https://udaanproject.org/) - OCR to MT
2. [cfilt-ssmt](https://www.cfilt.iitb.ac.in/ssmt/speech2speech)
3. [OCR to TTS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9697030)
4. [OCR to MT](https://arxiv.org/abs/1910.05535)

# Model Details

## Models used for Text Detection
1. [DBnet](https://arxiv.org/pdf/1911.08947.pdf)
2. [mobilenet](https://arxiv.org/abs/1801.04381)

## Models used for Text Recognition

1. [VGG16_Resnet](https://arxiv.org/pdf/1507.05717.pdf)
2. [TRocr](https://arxiv.org/abs/2109.10282)

## References used to learn about OCR details

Evaluation Metrics for Text Recognition - [CER and WER](https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510)
