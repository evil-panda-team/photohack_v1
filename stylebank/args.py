import torch
import os

device = torch.device("cuda")

batch_size = 4
lr = 0.001
T = 2
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 250000
REG_WEIGHT = 1e-5

continue_training = False

CONTENT_IMG_DIR = '../datasets/scene2archer/trainA'
STYLE_IMG_DIR = '../datasets/scene2archer/trainB'
MODEL_WEIGHT_DIR = 'weights_test_2'
BANK_WEIGHT_DIR = os.path.join(MODEL_WEIGHT_DIR, 'bank_2')
BANK_WEIGHT_PATH = os.path.join(BANK_WEIGHT_DIR, '{}_2.pth')
MODEL_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'model_2.pth')
ENCODER_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'encoder_2.pth')
DECODER_WEIGHT_PATH = os.path.join(MODEL_WEIGHT_DIR, 'decoder_2.pth')
GLOBAL_STEP_PATH = os.path.join(MODEL_WEIGHT_DIR, 'global_step_2.log')

K = 1000
MAX_ITERATION = 450 * K
ADJUST_LR_ITER = 20 * K
LOG_ITER = 5 * K
