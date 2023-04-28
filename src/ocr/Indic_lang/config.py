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
