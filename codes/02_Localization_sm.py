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
from segmentation_models import Unet, PSPNet
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

            # r_start, c_start = np.random.randint(low=0, high=(pre_image.shape[0] - self.patch_size), size=2)
            # x_pre_batch[b] = self.preprocessing(pre_image[r_start:r_start+self.patch_size, c_start:c_start+self.patch_size, :])       # preprocess input images based on backbone preprocessing
            # y_target_batch[b, :, :, 0] = target_image[r_start:r_start+self.patch_size, c_start:c_start+self.patch_size]

        return x_pre_batch, y_target_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.pre_img_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==================================================================================================
print('===> Data splits: ')
# ==================================================================================================

# Training data generation
cond00 = db['Group'] == 'Train'                # use images in Train folder
cond01 = db['Group'] == 'Tier3'                # use images in Train folder
cond1 = db['buildings#'] > 5                  # ensure there are buildings
cond2 = db['Pre_Post'] == 'pre'               # choose from pre or post
training_files = list(db[(cond00 | cond01) & cond1 & cond2]['img_name'])
print(len(training_files))

# Testing data generation
cond0 = db['Group'] == 'Test'                 # use images in Train folder
cond1 = db['buildings#'] > 0                  # ensure there are buildings
cond2 = db['Pre_Post'] == 'pre'              # choose from pre or post
testing_files = list(db[cond0 & cond1 & cond2]['img_name'])
print(len(testing_files))

# Validation data generation
cond0 = db['Group'] == 'Hold'                 # use images in Train folder
cond1 = db['buildings#'] > 0                 # ensure there are buildings
cond2 = db['Pre_Post'] == 'pre'              # choose from pre or post
validation_files = list(db[cond0 & cond1 & cond2]['img_name'])
print(len(validation_files))

CLASSES = ['buildings']                                                 # Localization
# CLASSES = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']    # Classification
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation

if n_classes == 1:
    TASK = 'localization'
else:
    TASK = 'classification'

BATCH = 10           # Define batch size for processing
PATCH = 288          # Define patch size for cropping. Patches are square.

# ==================================================================================================
print('===> Model definition: ')
# ==================================================================================================

# NAME = 'mobilev2_sk_unet'   # Put a custom name here
NAME = 'efficientnetb7_sk_unet'   # Put a custom name here

if NAME == 'resnet34_unet':
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_model = Unet(BACKBONE, classes=n_classes, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))

elif NAME == 'efficientnetb7_unet':
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_model = Unet(BACKBONE, classes=n_classes, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))

elif NAME == 'efficientnetb7_sk_unet':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, Activation
    from modules import SKConv
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_pretrained = Unet(BACKBONE, classes=n_classes, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    # localization_pretrained.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_unet_localization_10_256_08092023.h5')
    tf.compat.v1.reset_default_graph()
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
    # localization_branch.trainable = False
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    pre_features = localization_branch(input_pre)
    head = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.1)(pre_features)
    head = Conv2D(n_classes, (3, 3), padding='same', name='localization_conv')(head)
    output = Activation("sigmoid")(head)
    localization_model = Model(input_pre, output)
    # localization_model.load_weights('E:\Ahmadi\PaperProject\model_init_efficientnetb7_sk_unet_localization_10_256_09092023.h5')
    localization_branch.trainable = True

elif NAME == 'base_unet':
    from models import base_unet
    preprocess_input = None
    localization_model = base_unet(filters=32, output_channels=n_classes, width=PATCH, height=PATCH, input_channels=3, conv_layers=4)

elif NAME == 'attention_unet':
    from models import attention_unet
    preprocess_input = None
    localization_model = attention_unet(filters=32, output_channels=n_classes, width=PATCH, height=PATCH, input_channels=3, conv_layers=4)

elif NAME == 'residual_unet':
    from models import residual_unet
    preprocess_input = None
    localization_model = residual_unet(filters=32, output_channels=n_classes, width=PATCH, height=PATCH, input_channels=3, conv_layers=4)

elif NAME == 'mobilev2_sk_unet':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, Activation
    from modules import SKConv
    BACKBONE = 'mobilenetv2'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_pretrained = Unet(BACKBONE, classes=n_classes, encoder_weights='imagenet', encoder_freeze=True,
                                   activation='sigmoid', input_shape=(PATCH, PATCH, 3),
                                   decoder_filters=(512, 256, 128, 64, 32), decoder_block_type='transpose') # upsampling transpose
    tf.compat.v1.reset_default_graph()
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
    # localization_branch.trainable = True
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    pre_features = localization_branch(input_pre)
    head = SKConv(M=3, r=16, L=32, G=32, convolutions='different', dropout_rate=0.1, name='local_skconv')(pre_features)
    head = Conv2D(n_classes, (3, 3), padding='same', name='localization_conv')(head)
    output = Activation("sigmoid")(head)
    localization_model = Model(input_pre, output)
    localization_model.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')

elif NAME == 'swin_unet':
    from keras_unet_collection import models, losses
    tf.keras.backend.clear_session()
    localization_model = models.swin_unet_2d((PATCH, PATCH, 3), filter_num_begin=32,   # 16
                                             n_labels=n_classes, depth=4, stack_num_down=2, stack_num_up=2,
                                             patch_size=(4, 4), num_heads=[4, 8, 8, 8],
                                             window_size=[4, 2, 2, 2], num_mlp=200,             # 156
                                             output_activation='Sigmoid', shift_window=True,
                                             name='swin_unet')
    preprocess_input = None
    print(localization_model.summary())

elif NAME == 'pspnet':
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_model = PSPNet(BACKBONE, classes=n_classes, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))

else:
    print('No model is defined for localization. Please check.')
    preprocess_input = None
    localization_model = None

# print(localization_model.summary(line_length=150))
save_name = f'E:/Ahmadi/PaperProject/model_{NAME}_{TASK}_{BATCH}_{PATCH}_{dt.datetime.now().strftime("%d%m%Y")}_v0.h5'
print(save_name)

# ==================================================================================================    Data Generators
parameters_train = {'shuffle': True, 'batch_size': BATCH, 'patch_size': PATCH, 'preprocessing': preprocess_input}
train_generator = xBD_DataGenerator(path_to_jsons=training_files, **parameters_train)
valid_generator = xBD_DataGenerator(path_to_jsons=validation_files, **parameters_train)

parameters_test = {'shuffle': False, 'batch_size': BATCH, 'patch_size': PATCH, 'preprocessing': preprocess_input}
test_generator = xBD_DataGenerator(path_to_jsons=testing_files, **parameters_test)
# ==================================================================================================
# ==================================================================================================


TRAIN = False
if TRAIN:

    # ==================================================================================================
    print('===> Callbacks: ')
    # ==================================================================================================
    EPOCHS = 70

     # semantic_loss = Semantic_loss_functions()
    # LOSS = [semantic_loss.focal_loss, semantic_loss.dice_loss, tf.keras.losses.BinaryCrossentropy]
    LOSS = [sm.losses.DiceLoss(), sm.losses.BinaryCELoss(), sm.losses.BinaryFocalLoss()]
    OPTIMIZER = tf.keras.optimizers.Adamax(learning_rate=0.001)
    METRICS = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    CALLBACKS = [
        ModelCheckpoint(save_name, save_weights_only=True, save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8, min_lr=0.000001, mode='auto'),
    ]

    localization_model.compile(OPTIMIZER, loss=LOSS, metrics=METRICS)

    # ==================================================================================================
    print(f'\n    =================> Start training    {NAME} model...    <=================  \n')
    # ==================================================================================================

    history = localization_model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=CALLBACKS
    )

    # ==================================================================================================
    print('===> Saving model: ')
    # ==================================================================================================

    with open(f'E:/Ahmadi/PaperProject/model_{NAME}_{TASK}_{BATCH}_{PATCH}_v0_trainHistory', 'wb') as file_pi:
        pkl.dump(history.history, file_pi)

    # ==================================================================================================
    print('     ===> FINISHED <=== \n\n')
    # ==================================================================================================

else:
    import matplotlib.pyplot as plt

    with open('E:/Ahmadi/PaperProject/model_efficientnetb7_sk_unet_localization_5_256_trainHistory', 'rb') as file_pi:
        historyS = pkl.load(file_pi)

    with open('E:/Ahmadi/PaperProject/model_efficientnetb7_unet_localization_10_256_trainHistory', 'rb') as file_pi:
        historyE = pkl.load(file_pi)

    with open('E:/Ahmadi/PaperProject/model_resnet34_unet_localization_10_256_trainHistory', 'rb') as file_pi:
        historyR = pkl.load(file_pi)

    with open('E:/Ahmadi/PaperProject/model_base_unet_localization_10_256_trainHistory', 'rb') as file_pi:
        historyB = pkl.load(file_pi)

    with open('E:/Ahmadi/PaperProject/model_attention_unet_localization_10_256_trainHistory', 'rb') as file_pi:
        historyA = pkl.load(file_pi)

    # with open('E:/Ahmadi/PaperProject/model_residual_unet_localization_10_256_trainHistory', 'rb') as file_pi:
    with open('E:/Ahmadi/PaperProject/model_pspnet_localization_10_288_v0_trainHistory', 'rb') as file_pi:
        historyRe = pkl.load(file_pi)

    plt.figure(figsize=(8, 4))
    plt.subplot(121), plt.plot(historyE['loss'][:70], 'b')
    plt.subplot(121), plt.plot(historyR['loss'][:70], 'r')
    plt.subplot(121), plt.plot(historyB['loss'][:70], 'k')
    plt.subplot(121), plt.plot(historyA['loss'][:70], 'c')
    plt.subplot(121), plt.plot(historyRe['loss'][:70], 'm')
    plt.subplot(121), plt.plot(historyS['loss'][:70], 'y')
    plt.subplot(122), plt.plot(historyE['accuracy'][:70], 'b')
    plt.subplot(122), plt.plot(historyR['accuracy'][:70], 'r')
    plt.subplot(122), plt.plot(historyB['accuracy'][:70], 'k')
    plt.subplot(122), plt.plot(historyA['accuracy'][:70], 'c')
    plt.subplot(122), plt.plot(historyRe['accuracy'][:70], 'm')
    plt.subplot(122), plt.plot(historyS['accuracy'][:70], 'y')
    plt.show()

    # localization_model.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')
    localization_model.load_weights('E:/Ahmadi/PaperProject/model_efficientnetb7_sk_unet_localization_5_256_10092023.h5')

    # for idx in range(test_generator.__len__()):
    for idx in range(10):
        x, y = test_generator.__getitem__(idx)
        y_ = localization_model.predict(x)

        plt.figure(figsize=(10, BATCH))
        for i in range(BATCH):
            plt.subplot(3, BATCH, i+1), plt.imshow( (x[i] - np.amin(x[i])) / (np.amax(x[i]) - np.amin(x[i])) ), plt.xticks([]), plt.yticks([]), plt.title('PRE')
            plt.subplot(3, BATCH, i+11), plt.imshow( y[i] ), plt.xticks([]), plt.yticks([]), plt.title('GT')
            plt.subplot(3, BATCH, i+21), plt.imshow( y_[i] ), plt.xticks([]), plt.yticks([]), plt.title('Prediction')
        plt.tight_layout()
        plt.show()

