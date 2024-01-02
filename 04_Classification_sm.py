# Libraries ------------------------------------------------------------------------------------------
import os
from copy import copy

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

from utils import Semantic_loss_functions, weighted_categorical_crossentropy, f1_m, precision_m, recall_m, wcce_masked

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


            # AUGMENTATION TO BE ADDED #
            transform = A.Compose([
                A.RandomCrop(width=self.patch_size, height=self.patch_size, p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(min_holes=4, max_holes=12, max_height=15, max_width=15, mask_fill_value=0, p=0.6),
                A.OneOf([
                    A.MotionBlur(p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.6),
                    A.Blur(blur_limit=3, p=0.6)], p=0.7),
            ], additional_targets={'image_post': 'image'})
            transformed = transform(image=pre_image, image_post=post_image, mask=target_image)

            if self.preprocessing is None:
                x_pre_batch[b] = transformed['image']
                x_post_batch[b] = transformed['image_post']
            else:
                x_pre_batch[b] = self.preprocessing(transformed['image'])
                x_post_batch[b] = self.preprocessing(transformed['image_post'])
            y_target_batch[b] = transformed['mask']

            # # r_start, c_start = np.random.randint(low=0, high=(pre_image.shape[0] - self.patch_size), size=2)
            # # x_pre_batch[b] = self.preprocessing(pre_image[r_start:r_start+self.patch_size, c_start:c_start+self.patch_size, :])       # preprocess input images based on backbone preprocessing
            # x_pre_batch[b] = self.preprocessing(pre_image)       # preprocess input images based on backbone preprocessing
            # # x_post_batch[b] = self.preprocessing(post_image[r_start:r_start+self.patch_size, c_start:c_start+self.patch_size, :])
            # x_post_batch[b] = self.preprocessing(post_image)
            # # y_target_batch[b] = target_image[r_start:r_start+self.patch_size, c_start:c_start+self.patch_size, :]
            # y_target_batch[b] = target_image

        return (x_pre_batch, x_post_batch), y_target_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==================================================================================================
print('\n\n\t===> Data splits: ')
# ==================================================================================================

# Training data generation
training_files = []
cond00 = db['Group'] == 'Train'                # use images in Train folder
cond01 = db['Group'] == 'Tier3'                # use images in Train folder
# cond1 = db['buildings#'] > 20                  # ensure there are buildings
cond2 = db['Pre_Post'] == 'post'               # choose from pre or post
cond3 = db['destroyed#'] + db['minor-damage#'] + db['major-damage#'] > 10
cond4 = db['disaster_type'] != 'flooding'      # exclude flood events
training_files = list(db[(cond00 | cond01) & cond2 & cond3]['img_name'])
# training_files = list(db[cond00 & cond2 & cond3]['img_name'])
print(len(training_files))

# Testing data generation
testing_files = []
cond0 = db['Group'] == 'Test'                 # use images in Train folder
cond1 = db['destroyed#'] > 0                  # ensure there are buildings
cond2 = db['Pre_Post'] == 'post'              # choose from pre or post
cond3 = db['destroyed#'] + db['major-damage#'] + db['minor-damage#'] > 5
testing_files = list(db[cond0 & cond2 & cond3]['img_name'])
print(len(testing_files))

# Validation data generation
validation_files = []
cond0 = db['Group'] == 'Hold'                 # use images in Train folder
# cond1 = db['buildings#'] > 30                 # ensure there are buildings
cond2 = db['Pre_Post'] == 'post'              # choose from pre or post
cond3 = db['destroyed#'] + db['minor-damage#'] + db['major-damage#'] > 10
# cond4 = db['disaster_type'] != 'flooding'      # exclude flood events
validation_files = list(db[cond0 & cond2 & cond3]['img_name'])
print(len(validation_files))

# # K-FOLD CROSS VALIDATION =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# print('Split training data for k-fold cross-validation')
# training_files_ = copy(training_files)
# # training_files = training_files_[131:]
# training_files = training_files_[:524] + training_files_[655:]
# # validation_files = training_files_[:131]
# validation_files = training_files_[524:655]
# print(len(training_files))
# print(len(validation_files))
# # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# CLASSES = ['buildings']                                                 # Localization
CLASSES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']    # Classification
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
# n_classes = 4

if n_classes == 1:
    TASK = 'localization'
else:
    TASK = 'classification'

BATCH = 2            # Define batch size for processing
PATCH = 256          # Define patch size for cropping. Patches are square.

# ==================================================================================================
print('\n===> Model definition: ')
# ==================================================================================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Concatenate, BatchNormalization, Subtract, SpatialDropout2D
from modules import SKConv

# NAME = 'siamese_efficientnetb7_unet_sk'   # THIS IS COMPLETED.
# NAME = 'siamese_base_unet'   # THIS IS COMPLETED.
# NAME = 'siamese_base_unet_sk'   # THIS IS COMPLETED.
NAME = 'siamese_efficientnetb7_sk_unet'   # THIS IS COMPLETED.
# NAME = 'siamese_base_unet_sk_new'   # THIS IS COMPLETED.
# NAME = 'siamese_mobilev2_sk_unet'   # THIS IS COMPLETED.
# NAME = 'siamese_dual_effnet_sk_unet'   # THIS IS COMPLETED.

if NAME == 'siamese_efficientnetb7_unet_sk':
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    # DEFINE LOCALIZATION MODEL
    localization_pretrained = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    localization_pretrained.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_unet_localization_10_256_08092023.h5')
    tf.compat.v1.reset_default_graph()
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
    # DEFINE CLASSIFICATION MODEL
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    output_pre = localization_branch(input_pre)
    input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
    output_post = localization_branch(input_post)
    # Segmentation Head can be configured to get different results.
    head = Concatenate()([output_pre, output_post])
    head = SKConv(M=2, r=16, L=32, G=32, convolutions='same', dropout_rate=0.001)(head)
    head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    output = Activation("sigmoid")(head)
    classification_model = Model([input_pre, input_post], output)  # CLASSIFICATION MODEL

elif NAME == 'siamese_base_unet':
    from models import base_unet
    preprocess_input = None
    # DEFINE LOCALIZATION MODEL
    localization_pretrained = base_unet(filters=32, output_channels=1, width=PATCH, height=PATCH, input_channels=3, conv_layers=4)
    localization_pretrained.load_weights('E:\Ahmadi\PaperProject\model_base_unet_localization_10_256_09092023.h5')
    tf.compat.v1.reset_default_graph()
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('activation_39').output)
    # DEFINE CLASSIFICATION MODEL
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    output_pre = localization_branch(input_pre)
    input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
    output_post = localization_branch(input_post)
    # Segmentation Head can be configured to get different results.
    head = Concatenate()([output_pre, output_post])
    head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    output = Activation("sigmoid")(head)
    classification_model = Model([input_pre, input_post], output)  # CLASSIFICATION MODEL

elif NAME == 'siamese_base_unet_sk':
    from models import base_unet
    preprocess_input = None
    # DEFINE LOCALIZATION MODEL
    localization_pretrained = base_unet(filters=32, output_channels=1, width=PATCH, height=PATCH, input_channels=3, conv_layers=4)
    localization_pretrained.load_weights('E:\Ahmadi\PaperProject\model_base_unet_localization_10_256_09092023.h5')
    tf.compat.v1.reset_default_graph()
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('activation_39').output)
    # DEFINE CLASSIFICATION MODEL
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    output_pre = localization_branch(input_pre)
    input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
    output_post = localization_branch(input_post)
    # Segmentation Head can be configured to get different results.
    head = Concatenate()([output_pre, output_post])
    head = SKConv(M=2, r=16, L=32, G=32, convolutions='same', dropout_rate=0.001)(head)
    head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    output = Activation("sigmoid")(head)
    classification_model = Model([input_pre, input_post], output)  # CLASSIFICATION MODEL

elif NAME == 'siamese_efficientnetb7_sk_unet':
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    # DEFINE LOCALIZATION MODEL
    localization_pretrained = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    localization_pretrained.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_unet_localization_10_256_08092023.h5')
    tf.compat.v1.reset_default_graph()
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
    # DEFINE CLASSIFICATION MODEL
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    output_pre = localization_branch(input_pre)
    pre_sk = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='sk_loc_pre')(output_pre)
    input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
    output_post = localization_branch(input_post)
    post_sk = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='sk_loc_post')(output_post)
    head = Concatenate()([pre_sk, post_sk])
    head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    output = Activation("sigmoid")(head)
    classification_model = Model([input_pre, input_post], output)  # CLASSIFICATION MODEL
    # classification_model.load_weights('E:/Ahmadi/PaperProject/model_siamese_efficientnetb7_sk_unet_classification_3_256_15092023_V2.h5')

elif NAME == 'siamese_mobilev2_sk_unet':
    BACKBONE = 'mobilenetv2'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_pretrained_pre = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3), decoder_filters=(512, 256, 128, 64, 32), decoder_block_type='transpose')
    tf.compat.v1.reset_default_graph()
    localization_branch_pre = Model(inputs=localization_pretrained_pre.inputs, outputs=localization_pretrained_pre.get_layer('decoder_stage4b_relu').output, name='localization_branch_pre')
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    pre_features = localization_branch_pre(input_pre)
    head_pre = SKConv(M=3, r=16, L=32, G=32, convolutions='different', dropout_rate=0.01, name='local_skconv_pre')(pre_features)
    head_pre = Conv2D(1, (3, 3), padding='same', name='localization_conv_pre')(head_pre)
    output_pre = Activation("sigmoid")(head_pre)
    localization_model_pre = Model(input_pre, output_pre, name='localization_SK_branch_pre')
    localization_model_pre.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')
    localization_sk_branch_pre = Model(inputs=localization_model_pre.inputs, outputs=localization_model_pre.get_layer('local_skconv_pre_axpby').output, name='localization_SK_pre_features')
    localization_sk_branch_pre.trainable = False

    localization_pretrained_post = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3), decoder_filters=(512, 256, 128, 64, 32), decoder_block_type='transpose')
    tf.compat.v1.reset_default_graph()
    localization_branch_post = Model(inputs=localization_pretrained_post.inputs, outputs=localization_pretrained_post.get_layer('decoder_stage4b_relu').output, name='localization_branch_post')
    input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
    post_features = localization_branch_post(input_post)
    head_post = SKConv(M=3, r=16, L=32, G=32, convolutions='different', dropout_rate=0.01, name='local_skconv_post')(post_features)
    head_post = Conv2D(1, (3, 3), padding='same', name='localization_conv_post')(head_post)
    output_post = Activation("sigmoid")(head_post)
    localization_model_post = Model(input_post, output_post, name='localization_SK_branch_post')
    localization_model_post.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')
    localization_sk_branch_post = Model(inputs=localization_model_post.inputs, outputs=localization_model_post.get_layer('local_skconv_post_axpby').output, name='localization_SK_post_features')
    localization_sk_branch_post.trainable = False

    output_pre_sk_local = localization_sk_branch_pre(input_pre)
    output_post_sk_local = localization_sk_branch_post(input_post)
    # -------------------------------------------------------- Simple head.
    # head = Concatenate()([output_pre_sk_local, output_post_sk_local])
    # head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    # -------------------------------------------------------- New combination of features into the head.
    # head = Concatenate()([output_pre_sk_local, output_post_sk_local])
    # head = BatchNormalization(name='bn_sk')(head)
    # head = Conv2D(16, (3, 3), padding='same', name='middle_conv')(head)
    # head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    # head = BatchNormalization(name='bn_final')(head)
    # --------------------------------------------------------
    head_01 = Subtract()([pre_features, post_features])
    head_02 = Subtract()([output_pre_sk_local, output_post_sk_local])
    head_11 = Concatenate()([head_01, head_02])
    head_12 = Concatenate()([output_pre_sk_local, output_post_sk_local])
    head = Concatenate()([head_11, head_12])
    head = BatchNormalization(name='bn_sk')(head)
    head = Conv2D(16, (3, 3), padding='same', name='middle_conv')(head)
    head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    head = BatchNormalization(name='bn_final')(head)
    # --------------------------------------------------------
    output = Activation("sigmoid")(head)    # softmax
    classification_model = Model([input_pre, input_post], output, name='siamese_localization_sk_model')

elif NAME == 'siamese_dual_effnet_sk_unet':
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_pretrained_pre = Unet('efficientnetb7', classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    tf.compat.v1.reset_default_graph()
    localization_branch_pre = Model(inputs=localization_pretrained_pre.inputs, outputs=localization_pretrained_pre.get_layer('decoder_stage4b_relu').output, name='localization_branch_pre')
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    pre_features = localization_branch_pre(input_pre)
    head_pre = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.2, name='local_skconv_pre')(pre_features)
    head_pre = Conv2D(1, (3, 3), padding='same', name='localization_conv_pre')(head_pre)
    output_pre = Activation("sigmoid")(head_pre)
    localization_effsknet_pre = Model(input_pre, output_pre, name='localization_SK_branch_pre')
    localization_branch_pre.trainable = True
    localization_effsknet_pre.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_sk_unet_localization_5_256_10092023.h5')
    localization_sk_branch_pre = Model(inputs=localization_effsknet_pre.inputs, outputs=localization_effsknet_pre.get_layer('local_skconv_pre_axpby').output, name='localization_SK_pre_features')
    localization_sk_branch_pre.trainable = False

    localization_pretrained_post = Unet('efficientnetb7', classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    tf.compat.v1.reset_default_graph()
    localization_branch_post = Model(inputs=localization_pretrained_post.inputs, outputs=localization_pretrained_post.get_layer('decoder_stage4b_relu').output, name='localization_branch_post')
    input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
    post_features = localization_branch_post(input_post)
    head_post = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.2, name='local_skconv_post')(post_features)
    head_post = Conv2D(1, (3, 3), padding='same', name='localization_conv_post')(head_post)
    output_post = Activation("sigmoid")(head_post)
    localization_effsknet_post = Model(input_post, output_post, name='localization_SK_branch_post')
    localization_branch_post.trainable = True
    localization_effsknet_post.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_sk_unet_localization_5_256_10092023.h5')
    localization_sk_branch_post = Model(inputs=localization_effsknet_post.inputs, outputs=localization_effsknet_post.get_layer('local_skconv_post_axpby').output, name='localization_SK_post_features')
    localization_sk_branch_post.trainable = False
    # --------------------------------------------------------
    output_pre_sk_local = localization_sk_branch_pre(input_pre)
    output_post_sk_local = localization_sk_branch_post(input_post)
    # --------------------------------------------------------
    head_01 = Subtract()([pre_features, post_features])
    head_01 = SpatialDropout2D(rate=0.4)(head_01)
    head_02 = Subtract()([output_pre_sk_local, output_post_sk_local])
    head_02 = SpatialDropout2D(rate=0.4)(head_02)
    head_11 = Concatenate()([head_01, head_02])
    head_12 = Concatenate()([output_pre_sk_local, output_post_sk_local])
    head = Concatenate()([head_11, head_12])
    head = BatchNormalization(name='bn_sk')(head)
    head = Conv2D(16, (3, 3), padding='same', name='middle_conv')(head)
    head = SpatialDropout2D(rate=0.4)(head)
    head = Conv2D(n_classes, (3, 3), padding='same', name='class_conv')(head)
    head = BatchNormalization(name='bn_final')(head)
    # --------------------------------------------------------
    output = Activation("sigmoid")(head)    # softmax
    classification_model = Model([input_pre, input_post], output, name='siamese_classification_effsk_model')


print(classification_model.summary(line_length=150))
save_name = f'E:/Ahmadi/PaperProject/model_{NAME}_{TASK}_{BATCH}_{PATCH}_{dt.datetime.now().strftime("%d%m%Y-%H")}_AnotherTry.h5'
print(save_name)

# ==================================================================================================    Data Generators
parameters_train = {'shuffle': True, 'batch_size': BATCH, 'patch_size': PATCH, 'preprocessing': preprocess_input}
train_generator = xBD_DataGenerator(path_to_jsons=training_files, **parameters_train)
valid_generator = xBD_DataGenerator(path_to_jsons=validation_files, **parameters_train)

parameters_test = {'shuffle': False, 'batch_size': BATCH, 'patch_size': PATCH, 'preprocessing': preprocess_input}
test_generator = xBD_DataGenerator(path_to_jsons=testing_files, **parameters_test)
# ==================================================================================================
# ==================================================================================================

TRAIN = True
if TRAIN:
    # ==================================================================================================
    print('===> Callbacks: ')
    # ==================================================================================================
    EPOCHS = 40

    from utils import Semantic_loss_functions

    semantic_loss = Semantic_loss_functions()
    # LOSS = [semantic_loss.dice_loss, semantic_loss.unet3p_hybrid_loss, weighted_categorical_crossentropy([0.01, 0.1, 0.7, 0.7, 0.7])]
    # LOSS = sm.losses.CategoricalFocalLoss() + sm.losses.DiceLoss(class_weights=[1, 7, 7, 7])  #     sm.losses.CategoricalCELoss(class_weights=np.array([0.01, 0.1, 0.7, 0.7, 0.7])) +
    LOSS = [semantic_loss.focal_loss, semantic_loss.dice_loss, weighted_categorical_crossentropy([0.5, 2, 8, 7, 4])]
    # LOSS = [semantic_loss.focal_loss, semantic_loss.dice_loss]    # <<<<<<
    # LOSS = [semantic_loss.focal_loss_masked, semantic_loss.dice_loss_masked, wcce_masked([2, 8, 7, 4])]
    # LOSS = semantic_loss.unet3p_hybrid_loss
    LOSS_WEIGHTS = [9, 5, 10]
    # LOSS_WEIGHTS = [3, 1]
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)
    METRICS = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    CALLBACKS = [
        ModelCheckpoint(save_name, save_weights_only=True, save_best_only=True, mode='min'),
        # EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000001, mode='min'),
    ]

    classification_model.compile(OPTIMIZER, loss=LOSS, metrics=METRICS, loss_weights=LOSS_WEIGHTS)

    # ==================================================================================================
    print(f'\n    =================> Start training    {NAME} model... <=================  \n')
    # ==================================================================================================

    history = classification_model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=CALLBACKS
    )

    # ==================================================================================================
    print('===> Saving model: ')
    # ==================================================================================================

    with open(f'E:/Ahmadi/PaperProject/model_{NAME}_{TASK}_{BATCH}_{PATCH}_AnotherTry_trainHistory', 'wb') as file_pi:
        pkl.dump(history.history, file_pi)

    # ==================================================================================================
    print('     ===> FINISHED <=== \n\n')
    # ==================================================================================================
else:
    import matplotlib.pyplot as plt

    with open('E:/Ahmadi/PaperProject/model_siamese_base_unet_classification_2_256_kfold_all_trainHistory', 'rb') as file_pi:
        historyE = pkl.load(file_pi)

    plt.figure(figsize=(8, 4))
    plt.subplot(131), plt.plot(historyE['loss'][:-1], 'b')
    plt.subplot(131), plt.plot(historyE['val_loss'][:-1], '--b')
    plt.subplot(132), plt.plot(historyE['accuracy'][:-1], 'k')
    plt.subplot(132), plt.plot(historyE['val_accuracy'][:-1], '--k')
    plt.subplot(133), plt.plot(historyE['lr'][:-1], 'r')
    plt.show()

    # classification_model.load_weights('E:/Ahmadi/PaperProject/model_siamese_efficientnetb7_unet_sk_classification_3_256_10092023.h5')
    # classification_model.load_weights('E:/Ahmadi/PaperProject/model_siamese_dual_effnet_sk_unet_classification_5_256_26092023-08_V6_NewHead_5Class_Sigmoid.h5')
    # classification_model.load_weights('E:/Ahmadi/PaperProject/model_siamese_dual_effnet_sk_unet_classification_5_256_26092023-16_V7_NewHead_5Class_Sigmoid_3F1D.h5')
    classification_model.load_weights('E:/Ahmadi/PaperProject/model_siamese_base_unet_classification_2_256_07122023-12_kfold_all.h5')

    for idx in range(test_generator.__len__()):
        (x_pre, x_post), y = test_generator.__getitem__(idx)
        y_ = classification_model.predict([x_pre, x_post])

        plt.figure(figsize=(10, BATCH))
        for i in range(BATCH):
            plt.subplot(BATCH, 9, 9*i+1), plt.imshow((x_pre[i] - np.amin(x_pre[i])) / (np.amax(x_pre[i]) - np.amin(x_pre[i]))), plt.xticks([]), plt.yticks([]), plt.title('PRE')
            plt.subplot(BATCH, 9, 9*i+2), plt.imshow((x_post[i] - np.amin(x_post[i])) / (np.amax(x_post[i]) - np.amin(x_post[i]))), plt.xticks([]), plt.yticks([]), plt.title('POST')
            y[i, :, :, 0] = 0
            plt.subplot(BATCH, 9, 9*i+3), plt.imshow(np.argmax(y[i], axis=-1), vmin=0, vmax=4), plt.xticks([]), plt.yticks([]), plt.title('GT')
            locs = y_[i, :, :, 0] > 0.5
            plt.subplot(BATCH, 9, 9*i+4), plt.imshow(locs), plt.xticks([]), plt.yticks([]), plt.title('localization')
            y_[i, :, :, 0] = 0
            plt.subplot(BATCH, 9, 9*i+5), plt.imshow(np.argmax(y_[i], axis=-1) * locs, vmin=0, vmax=4), plt.xticks([]), plt.yticks([]), plt.title('classification')
            plt.subplot(BATCH, 9, 9*i+6), plt.imshow(y_[i, :, :, 1] * locs, vmin=0, vmax=1), plt.xticks([]), plt.yticks([]), plt.title('1')
            plt.subplot(BATCH, 9, 9*i+7), plt.imshow(y_[i, :, :, 2] * locs, vmin=0, vmax=1), plt.xticks([]), plt.yticks([]), plt.title('2')
            plt.subplot(BATCH, 9, 9*i+8), plt.imshow(y_[i, :, :, 3] * locs, vmin=0, vmax=1), plt.xticks([]), plt.yticks([]), plt.title('3')
            plt.subplot(BATCH, 9, 9*i+9), plt.imshow(y_[i, :, :, 4] * locs, vmin=0, vmax=1), plt.xticks([]), plt.yticks([]), plt.title('4')
        plt.tight_layout()
        plt.show()

        # # FOR MOBILENET WITH BATCH 1 AND PATCH 1024 + 4 classes
        # fig, ax = plt.subplots(2, 4, sharey=True, sharex=True)
        # for i in range (BATCH):
        #     gt = np.zeros((PATCH, PATCH, 5))
        #     gt[:, :, 1:] = y[i, :, :, :]
        #     GT = np.argmax(gt, axis=-1)
        #     pred = np.zeros((PATCH, PATCH, 5))
        #     pred[:, :, 1:] = y_[i, :, :, :]
        #     ax[0, 0].imshow((x_pre[i] - np.amin(x_pre[i])) / (np.amax(x_pre[i]) - np.amin(x_pre[i]))), ax[0, 0].set_xticks([]), ax[0, 0].set_yticks([])
        #     ax[0, 1].imshow((x_post[i] - np.amin(x_post[i])) / (np.amax(x_post[i]) - np.amin(x_post[i]))), ax[0, 1].set_xticks([]), ax[0, 1].set_yticks([])
        #     ax[0, 2].imshow(GT, vmin=0, vmax=4), ax[0, 2].set_title('GT'), ax[0, 2].set_xticks([]), ax[0, 2].set_yticks([])
        #     ax[0, 3].imshow(np.argmax(pred, axis=-1) * (GT > 0), vmin=0, vmax=4), ax[0, 3].set_title('classification'), ax[0, 3].set_xticks([]), ax[0, 3].set_yticks([])
        #     ax[1, 0].imshow(y_[i, :, :, 0], vmin=0, vmax=1, cmap='gray'), ax[1, 0].set_xticks([]), ax[1, 0].set_yticks([])   #  * (GT > 0)
        #     ax[1, 1].imshow(y_[i, :, :, 1], vmin=0, vmax=1, cmap='gray'), ax[1, 1].set_xticks([]), ax[1, 1].set_yticks([])   #  * (GT > 0)
        #     ax[1, 2].imshow(y_[i, :, :, 2], vmin=0, vmax=1, cmap='gray'), ax[1, 2].set_xticks([]), ax[1, 2].set_yticks([])   #  * (GT > 0)
        #     ax[1, 3].imshow(y_[i, :, :, 3], vmin=0, vmax=1, cmap='gray'), ax[1, 3].set_xticks([]), ax[1, 3].set_yticks([])   #  * (GT > 0)
        #
        # plt.tight_layout()
        # plt.show()






















