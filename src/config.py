TRAIN_PATH      =  "/data/BADRI/IHTR/trainset/"
VALIDATION_PATH = "/data/BADRI/IHTR/validationset_small/"
TEST_PATH       = "/data/BADRI/IHTR/testset_small/"
MODELS_PATH     = "./models/"


LANGUAGE   = "tamil"

ARCH = "crnn_vgg16_bn"
EPOCHS     = 100
BATCH_SIZE = 1024


RESUME      = True
RESUME_FILE = MODELS_PATH + ARCH + "_" + "tamil_cont" + ".pt"


GLOW_MODEL_DIR = 'src/temp/tts_infer/translit_models/hindi/glow_ckp'
HIFI_MODEL_DIR = 'src/temp/tts_infer/translit_models/hindi/hifi_ckp'

SUMMARIZATION_MODEL = 'google/pegasus-cnn_dailymail'
TRANSLATION_MODEL = 'facebook/nllb-200-distilled-600M'


# MAX_VALUE = 65536
MAX_VALUE = 8192

