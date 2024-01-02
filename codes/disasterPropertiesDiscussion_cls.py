# Libraries ------------------------------------------------------------------------------------------
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import pickle as pkl
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl

from skimage.io import imread
from skimage.filters import median

import tensorflow as tf
import keras

os.environ['SM_FRAMEWORK'] = 'tf.keras'
keras.backend.set_image_data_format('channels_last')
from segmentation_models import Unet
from segmentation_models import get_preprocessing

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams['svg.fonttype'] = 'none'
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
        self.post_img_paths = []
        self.loc_target_paths = []
        self.cls_target_paths = []

        for path in self.path_to_jsons:
            self.pre_img_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/images/').replace('_post_', '_pre_') + '.png')
            self.post_img_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/images/') + '.png')
            if 'tier3' not in path:
                self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '_target.png')
                self.cls_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/') + '_target.png')
            else:
                self.loc_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/').replace('_post_', '_pre_') + '.png')
                self.cls_target_paths.append('E:/Ahmadi/Data/xBD/' + '/'.join(path.rsplit('/', 3)[1:]).replace('/labels/', '/targets/') + '.png')

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
        post_image_batch = list(np.array(self.post_img_paths)[batch_indices])
        loc_target_batch = list(np.array(self.loc_target_paths)[batch_indices])
        cls_target_batch = list(np.array(self.cls_target_paths)[batch_indices])

        x_pre_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype='float32')
        x_post_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (3,), dtype='float32')
        y_target_batch = np.empty((self.batch_size,) + (self.patch_size, self.patch_size) + (5,), dtype='float32')
        for b in range(self.batch_size):
            pre_image = imread(pre_image_batch[b])
            post_image = imread(post_image_batch[b])

            target_image = imread(cls_target_batch[b])
            # Treat "unclassified" as no-damage
            target_image[target_image == 5] = 1

            localization_image = imread(loc_target_batch[b])
            localization_image[localization_image > 1] = 1

            # for CATEGORICAL ONE-HOT classes
            categorical = np.zeros((target_image.shape[0], target_image.shape[1], 5))
            categorical[:, :, 0] = localization_image
            categorical[:, :, 1] = target_image == 1
            categorical[:, :, 2] = target_image == 2
            categorical[:, :, 3] = target_image == 3
            categorical[:, :, 4] = target_image == 4
            target_image = categorical

            if self.preprocessing is None:
                x_pre_batch[b] = pre_image
                x_post_batch[b] = post_image
            else:
                x_pre_batch[b] = self.preprocessing(pre_image)
                x_post_batch[b] = self.preprocessing(post_image)
            y_target_batch[b] = target_image

        return (x_pre_batch, x_post_batch), y_target_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==================================================================================================
print('\n\n\t===> Data splits: ')
# ==================================================================================================

# Testing data generation
testing_files = []
cond0 = db['Group'] == 'Test'                 # use images in Train folder
cond1 = db['Pre_Post'] == 'pre'              # choose from pre or post
cond2 = db['Pre_Post'] == 'post'              # choose from pre or post
cond3 = db['destroyed#'] + db['major-damage#'] + db['minor-damage#'] > 0
testing_files = list(db[cond0 & cond2 & cond3]['img_name'])
print(len(testing_files))

a = np.array(db['Group'] == 'Test')
b = np.array(db['Pre_Post'] == 'post')
c = np.array(db['destroyed#'] + db['major-damage#'] + db['minor-damage#'] > 0)
those_482 = np.where(a*b*c)[0]
# e = np.array(db['destroyed#'] + db['major-damage#'] + db['minor-damage#'] > 10)
e = np.array(db['buildings#'] > 20)
f = np.array(db['disaster_type'] != 'flooding')
d = np.where(a*b*e*f)[0]
inds = []
for id in d:
    if id in those_482:
        inds.append(list(those_482).index(id))
print(len(inds))

d_sun = np.abs(np.array(db[cond0 & cond1 & (db['buildings#']>0)]['sun_elevation']) - np.array(db[cond0 & cond2 & (db['buildings#']>0)]['sun_elevation']))
d_gsd = np.abs(np.array(db[cond0 & cond1 & (db['buildings#']>0)]['gsd']) - np.array(db[cond0 & cond2 & (db['buildings#']>0)]['gsd']))
d_off = np.abs(np.array(db[cond0 & cond1 & (db['buildings#']>0)]['off_nadir_angle']) - np.array(db[cond0 & cond2 & (db['buildings#']>0)]['off_nadir_angle']))
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.grid(True, c='gray', ls=':', alpha=0.6)
# plt.hist(db[cond0 & cond1 & (db['buildings#']>0)]['sun_elevation'], bins=50, edgecolor='k', facecolor='c')
# plt.vlines(x=50, ymin=0, ymax=200, colors='r', ls='--', lw=1)
# plt.subplot(232)
# plt.grid(True, c='gray', ls=':', alpha=0.6)
# plt.plot(db[cond0 & cond1 & (db['buildings#']>0)]['off_nadir_angle'], db[cond0 & cond1 & (db['buildings#']>0)]['gsd'], 'oc', mec='k', ms=5, alpha=0.5)
# plt.vlines(x=18, ymin=1.2, ymax=2.4, colors='r', ls='--', lw=1)
# plt.hlines(y=1.8, xmin=5, xmax=30, colors='r', ls='--', lw=1)
# plt.subplot(234)
# plt.grid(True, c='gray', ls=':', alpha=0.6)
# plt.hist(d_sun, bins=50, edgecolor='k', facecolor='c')
# plt.vlines(x=10, ymin=0, ymax=200, colors='r', ls='--', lw=1)
# plt.subplot(235)
# plt.grid(True, c='gray', ls=':', alpha=0.6)
# plt.hist(d_gsd, bins=50, edgecolor='k', facecolor='c')
# plt.vlines(x=0.5, ymin=0, ymax=200, colors='r', ls='--', lw=1)
# plt.subplot(236)
# plt.grid(True, c='gray', ls=':', alpha=0.6)
# plt.hist(d_off, bins=50, edgecolor='k', facecolor='c')
# plt.vlines(x=12, ymin=0, ymax=200, colors='r', ls='--', lw=1)
#
# plt.subplot(231), plt.xlabel('Sun elevation angles'), plt.ylabel('Number of images'), plt.title('Histogram of pre-disaster sun elevation angles')
# plt.subplot(232), plt.xlabel('Off nadir looking angles'), plt.ylabel('Spatial resolution (GSD - m)'), plt.title('Off nadir angle vs. ground resolution ')
# plt.subplot(234), plt.xlabel('Sun elevation angle difference'), plt.ylabel('Number of images'), plt.title('Histogram of differences between\npre- and post-disaster Sun elevation angles')
# plt.subplot(235), plt.xlabel('Spatial resolution difference'), plt.ylabel('Number of images'), plt.title('Histogram of differences between\npre- and post-disaster GSDs')
# plt.subplot(236), plt.xlabel('Off-nadir look angle difference'), plt.ylabel('Number of images'), plt.title('Histogram of differences between\npre- and post-disaster off-nadir angles')
# plt.tight_layout()
# plt.savefig('E:/Ahmadi/PaperProject/discussion_onlyTest_Bld_gt0.svg', dpi=600)
# plt.show()

CLASSES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']    # Classification

n_classes = 5
BATCH = 1
PATCH = 1024

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
TEST_0 = (db['Group'] == 'Test') & (db['Pre_Post'] == 'post') & (db['buildings#'] > 0)
TEST_1 = db[TEST_0][d_sun < 10]
TEST_2 = db[TEST_0][d_sun > 10]
TEST_3 = db[TEST_0][d_gsd < 0.5]
TEST_4 = db[TEST_0][d_gsd > 0.5]
TEST_5 = db[TEST_0][d_off < 12]
TEST_6 = db[TEST_0][d_off > 12]

TESTS = [db[TEST_0], TEST_1, TEST_2, TEST_3, TEST_4, TEST_5, TEST_6]

TYPE_1 = db[TEST_0][np.array(db[TEST_0]['disaster_type'] == 'earthquake')]
TYPE_2 = db[TEST_0][np.array(db[TEST_0]['disaster_type'] == 'fire')]
TYPE_3 = db[TEST_0][np.array(db[TEST_0]['disaster_type'] == 'flooding')]
TYPE_4 = db[TEST_0][np.array(db[TEST_0]['disaster_type'] == 'tsunami')]
TYPE_5 = db[TEST_0][np.array(db[TEST_0]['disaster_type'] == 'volcano')]
TYPE_6 = db[TEST_0][np.array(db[TEST_0]['disaster_type'] == 'wind')]

TYPES = [TYPE_1, TYPE_2, TYPE_3, TYPE_4, TYPE_5, TYPE_6]
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
BACKBONE = 'efficientnetb7'
preprocess_input = get_preprocessing(BACKBONE)
parameters_eff = {'shuffle': False, 'batch_size': BATCH, 'patch_size': PATCH, 'preprocessing': preprocess_input}
# ==================================================================================================
print('\n===> Model definition: ')
# ==================================================================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Concatenate
from modules import SKConv
from models import base_unet

# ......................................................................................................................
# ......................................................................................................................
# DEFINE LOCALIZATION MODEL
localization_pretrained_effsk = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
tf.compat.v1.reset_default_graph()
localization_branch_effsk = Model(inputs=localization_pretrained_effsk.inputs, outputs=localization_pretrained_effsk.get_layer('decoder_stage4b_relu').output)
# DEFINE CLASSIFICATION MODEL
input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
output_pre = localization_branch_effsk(input_pre)
pre_sk = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='sk_loc_pre')(output_pre)
input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
output_post = localization_branch_effsk(input_post)
post_sk = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='sk_loc_post')(output_post)
head = Concatenate()([pre_sk, post_sk])
head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
output = Activation("sigmoid")(head)
classification_effskunet = Model([input_pre, input_post], output)  # CLASSIFICATION MODEL
classification_effskunet.load_weights('E:/Ahmadi/PaperProject/model_siamese_efficientnetb7_sk_unet_classification_2_256_09122023-00_AnotherTry.h5')
# ......................................................................................................................


import segmentation_models as sm
from utils import Semantic_loss_functions, weighted_categorical_crossentropy

semantic_loss = Semantic_loss_functions()
LOSS = [semantic_loss.focal_loss, semantic_loss.dice_loss, weighted_categorical_crossentropy([0.5, 2, 8, 7, 4])]
METRICS = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
classification_effskunet.compile(loss=LOSS, metrics=METRICS)
# classification_effskunet.evaluate(eff_generator, batch_size=BATCH, verbose=2)

# print('RUNNING EXPERIMENTS ..............................................................')
# ALL_EXPERIMENTS = {}
# count = 0
# for t in TYPES:
#     testing_files = list(t['img_name'])
#     print(len(testing_files))
#     eff_generator = xBD_DataGenerator(path_to_jsons=testing_files, **parameters_eff)
#     metric = np.zeros((len(testing_files), 4))
#     for idx in range(len(testing_files)):
#         x, y = eff_generator.__getitem__(idx)
#         los, acc, iou, f1s = classification_effskunet.evaluate(x, y, batch_size=BATCH, verbose=0)
#         metric[idx, :] = los, acc, iou, f1s
#     print('\t\t', metric.shape)
#     ALL_EXPERIMENTS[count] = metric
#     count += 1
#
#     with open(f'E:/Ahmadi/PaperProject/experiments_types.pkl', 'wb') as file_pi:
#         pkl.dump(ALL_EXPERIMENTS, file_pi)

with open(f'E:/Ahmadi/PaperProject/experiments_types.pkl', 'rb') as file_pi:
    ALL_EXPERIMENTS = pkl.load(file_pi)

_los = []
_acc = []
_iou = []
_f1s = []
# for i in range(7):
for i in range(6):
    _los.append(ALL_EXPERIMENTS[i][:, 0])
    _acc.append(ALL_EXPERIMENTS[i][:, 1])
    _iou.append(ALL_EXPERIMENTS[i][:, 2])
    _f1s.append(ALL_EXPERIMENTS[i][:, 3])
METRICS = [_los, _acc, _iou, _f1s]
NAMES = ['Loss', 'Accuracy', 'IoU', 'F1-Score']
marker = dict(markerfacecolor='r', marker='x', alpha=0.4, markersize=3)
colors = ['orange', 'b', 'teal', 'm', 'limegreen', 'gold']
count = 1
plt.figure(figsize=(14, 5))
for m in METRICS:
    plt.subplot(1, 4, count)
    plt.grid(axis='y', ls='--', lw=1, alpha=0.4, color='gray', zorder=0)
    box = plt.boxplot(m, notch=False, flierprops=marker, patch_artist=True, zorder=2, showmeans=True, bootstrap=5000)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel(NAMES[count-1] + ' %')
    # plt.xticks(ticks=[1, 2, 3, 4, 5, 6, 7],
    plt.xticks(ticks=[1, 2, 3, 4, 5, 6],
               # labels=['All', 'Sun < 10', 'Sun > 10', 'GSD < 0.5', 'GSD > 0.5', 'Off < 12', 'Off > 12'],
               labels=['earthquake', 'fire', 'flooding', 'tsunami', 'volcano', 'wind'],
               # labels=['GSD<=1.8', 'GSD>1.8', 'Nadir<=18', 'Nadir>18', 'GSD<=1.8&Sun>50', 'Sun<=50', 'Sun>50'],
               rotation=90)
    if count > 1:
        plt.yticks(ticks=np.arange(0, 1.1, 0.1), labels=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
        plt.ylim([-0.1, 1.1])
    elif count == 1:
        plt.yticks(ticks=np.arange(0, 0.05, 0.01), labels=['0', '0.01', '0.02', '0.03', '0.04',])
        plt.ylim([0.0, 0.04])
    count += 1
plt.tight_layout()
plt.subplots_adjust(left=0.055, bottom=0.352, right=0.977, top=0.969, wspace=0.279, hspace=0.2)
plt.savefig(f'E:/Ahmadi/PaperProject/metrics_discussion_types.svg', dpi=600)
# plt.savefig(f'E:/Ahmadi/PaperProject/metrics_wSwin.png', dpi=300)
plt.show()






plt.figure()