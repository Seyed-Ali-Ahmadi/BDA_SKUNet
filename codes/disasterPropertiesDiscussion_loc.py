# Libraries ------------------------------------------------------------------------------------------
import os

import pandas as pd
import numpy as np
import pickle as pkl
import datetime as dt

import albumentations as A
from skimage.io import imread

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import keras

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
keras.backend.set_image_data_format('channels_last')
from segmentation_models import Unet
from segmentation_models import get_preprocessing

from utils import Semantic_loss_functions, weighted_categorical_crossentropy, f1_m, precision_m, recall_m

# ==================================================================================================
print('===> All packages successfully imported.')
# ==================================================================================================
ROOT = 'E:/Ahmadi/Data/xBD/'

test = 'test/'
train = 'train/'
hold = 'hold/'
tier3 = 'tier3/'

SUB = ['images', 'labels', 'targets']
# ==================================================================================================
with open(f'{ROOT}Metadata_xBD/All_Data_Props.csv', 'rb') as f:
    db = pd.read_csv(f)
# ==================================================================================================
print('===> Database successfully imported.')
# ==================================================================================================

class xBD_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path_to_jsons, batch_size=5, patch_size=256, shuffle=True, preprocessing=None):

        self.path_to_jsons = path_to_jsons
        self.pre_img_paths = []
        self.loc_target_paths = []

        for i in range(len(self.path_to_jsons)):
            self.pre_img_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(self.path_to_jsons[i].rsplit('/', 3)[1:]).replace('/labels/', '/images/').replace('_post_', '_pre_') + '.png')
            if 'tier3' not in self.path_to_jsons[i]:
              self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(self.path_to_jsons[i].rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '_target.png')
            else:
              self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(self.path_to_jsons[i].rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '.png')

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.on_epoch_end()
        self.preprocessing = preprocessing

    def __len__(self):
        return int(len(self.pre_img_paths) // self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        pre_image_batch = list(np.array(self.pre_img_paths)[batch_indices])
        loc_target_batch = list(np.array(self.loc_target_paths)[batch_indices])

        x_pre_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype='float32')
        y_target_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (1,), dtype='float32')
        for b in range(self.batch_size):
            pre_image = imread(pre_image_batch[b])
            target_image = imread(loc_target_batch[b])

            # AUGMENTATION TO BE ADDED #
            transform = A.Compose([
                A.RandomCrop(width=self.patch_size, height=self.patch_size, p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(p=0.6),
                A.Transpose(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(min_holes=4, max_holes=12, max_height=20, max_width=20, mask_fill_value=0, p=0.6),     # https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/
                A.OneOf([
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.6),
                    A.Blur(blur_limit=3, p=0.6)], p=0.7),])
            transformed = transform(image=pre_image, mask=target_image)

            if self.preprocessing is None:
                x_pre_batch[b] = transformed['image']
            else:
                x_pre_batch[b] = self.preprocessing(transformed['image'])
            y_target_batch[b, :, :, 0] = transformed['mask']

        return x_pre_batch, y_target_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==================================================================================================
print('===> Data splits: ')
# ==================================================================================================
# Testing data generation
testing_files = []
cond0 = db['Group'] == 'Test'                 # use images in Train folder
cond1 = db['buildings#'] > 0                  # ensure there are buildings
cond2 = db['Pre_Post'] == 'pre'               # choose from pre or post
testing_files = list(db[cond0 & cond1 & cond2]['img_name'])
print(len(testing_files))
# cond_gsd = db['gsd'] <= 1.8
# cond_off = db['off_nadir_angle'] <= 18
# cond_sun = db['sun_elevation'] <= 50
cond_gsd = db['gsd'] > 1.8
cond_off = db['off_nadir_angle'] > 18
cond_sun = db['sun_elevation'] > 50

GSD = list(db[cond0 & cond1 & cond2 & cond_gsd]['img_name'])
print(len(GSD))
OFF = list(db[cond0 & cond1 & cond2 & cond_off]['img_name'])
print(len(OFF))
SUN = list(db[cond0 & cond1 & cond2 & cond_sun]['img_name'])
print(len(SUN))

GSD_OFF = list(db[cond0 & cond1 & cond2 & cond_gsd & cond_off]['img_name'])
print(len(GSD_OFF))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.hist(db[cond0 & cond2]['sun_elevation'], bins=50)
# plt.subplot(122), plt.plot(db[cond2]['off_nadir_angle'], db[cond2]['gsd'], '.k')
plt.subplot(122)
plt.vlines(18, 1.3, 2.3, 'c')
plt.hlines(1.8, 5, 28, 'r')
plt.plot(db[cond0 & cond2]['off_nadir_angle'], db[cond0 & cond2]['gsd'], '.b')
plt.plot(db[cond0 & cond2 & cond_gsd]['off_nadir_angle'], db[cond0 & cond2 & cond_gsd]['gsd'], 'or', mfc='none', ms=13)
plt.plot(db[cond0 & cond2 & cond_off]['off_nadir_angle'], db[cond0 & cond2 & cond_off]['gsd'], 'sc', mfc='none', ms=7)
plt.show()

# a = db[(db['Group'] == 'Test') & (db['Pre_Post'] == 'pre')]['sun_elevation']
# b = db[(db['Group'] == 'Test') & (db['Pre_Post'] == 'post')]['sun_elevation']
# plt.figure()
# plt.hist(np.array(a) - np.array(b), bins=50)
# plt.show()
# ======================================================================================================================
CLASSES = ['buildings']                                                 # Localization
n_classes = 1

TASK = 'localization'

BATCH = 1           # Define batch size for processing
PATCH = 1024        # Define patch size for cropping. Patches are square.
# ======================================================================================================================
NAME = 'efficientnetb7_sk_unet'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation
from modules import SKConv
localization_pretrained = Unet('efficientnetb7', classes=n_classes, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
pre_features = localization_branch(input_pre)
head = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001)(pre_features)
head = Conv2D(n_classes, (3, 3), padding='same', name='localization_conv')(head)
output = Activation("sigmoid")(head)
localization_effsknet = Model(input_pre, output)
localization_branch.trainable = True
localization_effsknet.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_sk_unet_localization_5_256_10092023.h5')
# ......................................................................................................................
preprocess = get_preprocessing('efficientnetb7')
parameters_test = {'shuffle': False, 'batch_size': BATCH, 'patch_size': PATCH, 'preprocessing': preprocess}
effnet_generator = xBD_DataGenerator(path_to_jsons=SUN, **parameters_test)
# ......................................................................................................................
tp = []
fp = []
tn = []
fn = []

for idx in range(len(SUN)):
    x2, y2 = effnet_generator.__getitem__(idx)
    y_effnet = localization_effsknet.predict(x2)

    for j in range(BATCH):
        TP = np.sum((y2[j, :, :, 0] == 1) & ((y_effnet[j, :, :, 0] > 0.5) == 1))
        TN = np.sum((y2[j, :, :, 0] == 0) & ((y_effnet[j, :, :, 0] > 0.5) == 0))
        FP = np.sum((y2[j, :, :, 0] == 0) & ((y_effnet[j, :, :, 0] > 0.5) == 1))
        FN = np.sum((y2[j, :, :, 0] == 1) & ((y_effnet[j, :, :, 0] > 0.5) == 0))

        tp.append(TP)
        fp.append(FP)
        tn.append(TN)
        fn.append(FN)

    # print(idx, '\t\t', len(tp), len(fp), len(tn), len(fn))

EVAL = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

with open(f'E:/Ahmadi/PaperProject/eval_SUNgt50_localization', 'wb') as file_pi:    # less than or equal to (le) 1.8
    pkl.dump(EVAL, file_pi)
    # evaluation = pkl.load(file_pi)