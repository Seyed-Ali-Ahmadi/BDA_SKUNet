# Libraries ------------------------------------------------------------------------------------------
import os

import pandas as pd
import numpy as np
import pickle as pkl
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.transform import resize

import tensorflow as tf
import keras

os.environ['SM_FRAMEWORK'] = 'tf.keras'
keras.backend.set_image_data_format('channels_last')
from segmentation_models import Unet
from segmentation_models import get_preprocessing

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Concatenate, Subtract, SpatialDropout2D
from modules import SKConv


BATCH = 1
PATCH = 1024
PATH = 'E:/Ahmadi/Data/Bam/'
GT = os.listdir(PATH + 'GT/')

localization_model = None
classification_model = None
preprocess_input = None

localization = False
classification = True

c = mpl.colors.ListedColormap(['black', 'forestgreen', 'gold', 'darkorange', 'red'])
n = mpl.colors.Normalize(vmin=0, vmax=4)

# if localization:
#     BACKBONE = 'mobilenetv2'
#     preprocess_input = get_preprocessing(BACKBONE)
#     localization_pretrained = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3), decoder_filters=(512, 256, 128, 64, 32), decoder_block_type='transpose')
#     tf.compat.v1.reset_default_graph()
#     localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
#     input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
#     pre_features = localization_branch(input_pre)
#     head = SKConv(M=3, r=16, L=32, G=32, convolutions='different', dropout_rate=0.1, name='local_skconv')(pre_features)
#     head = Conv2D(1, (3, 3), padding='same', name='localization_conv')(head)
#     output = Activation("sigmoid")(head)
#     localization_model = Model(input_pre, output)
#     localization_model.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')

if localization:
    preprocess_input = get_preprocessing('efficientnetb7')
    localization_pretrained = Unet('efficientnetb7', classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    localization_branch = Model(inputs=localization_pretrained.inputs, outputs=localization_pretrained.get_layer('decoder_stage4b_relu').output)
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    pre_features = localization_branch(input_pre)
    head = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001)(pre_features)
    head = Conv2D(1, (3, 3), padding='same', name='localization_conv')(head)
    output = Activation("sigmoid")(head)
    localization_model = Model(input_pre, output)
    localization_branch.trainable = True
    localization_model.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_sk_unet_localization_5_256_10092023.h5')

# if classification:
#     preprocess_input = get_preprocessing('efficientnetb7')
#     localization_pretrained_effsk = Unet('efficientnetb7', classes=1, encoder_weights='imagenet', encoder_freeze=True,activation='sigmoid', input_shape=(PATCH, PATCH, 3))
#     localization_pretrained_effsk.load_weights('E:\Ahmadi\PaperProject\model_efficientnetb7_unet_localization_10_256_08092023.h5')
#     tf.compat.v1.reset_default_graph()
#     localization_branch_effsk = Model(inputs=localization_pretrained_effsk.inputs,outputs=localization_pretrained_effsk.get_layer('decoder_stage4b_relu').output)
#     # DEFINE CLASSIFICATION MODEL
#     input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
#     output_pre = localization_branch_effsk(input_pre)
#     pre_sk = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='sk_loc_pre')(output_pre)
#     input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
#     output_post = localization_branch_effsk(input_post)
#     post_sk = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='sk_loc_post')(output_post)
#     head = Concatenate()([pre_sk, post_sk])
#     head = Conv2D(5, (3, 3), padding='same', name='class_conv')(head)
#     output = Activation("sigmoid")(head)
#     classification_model = Model([input_pre, input_post], output)  # CLASSIFICATION MODEL
#     classification_model.load_weights('E:\Ahmadi\PaperProject\model_siamese_efficientnetb7_sk_unet_classification_3_256_12092023.h5')


if classification:
    BACKBONE = 'efficientnetb7'
    preprocess_input = get_preprocessing(BACKBONE)
    localization_pretrained_pre = Unet('efficientnetb7', classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3))
    tf.compat.v1.reset_default_graph()
    localization_branch_pre = Model(inputs=localization_pretrained_pre.inputs, outputs=localization_pretrained_pre.get_layer('decoder_stage4b_relu').output, name='localization_branch_pre')
    input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
    pre_features = localization_branch_pre(input_pre)
    head_pre = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='local_skconv_pre')(pre_features)
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
    head_post = SKConv(M=2, r=16, L=32, G=16, convolutions='same', dropout_rate=0.001, name='local_skconv_post')(post_features)
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
    head_01 = SpatialDropout2D(rate=0.4)(head_01)  #
    head_02 = Subtract()([output_pre_sk_local, output_post_sk_local])
    head_02 = SpatialDropout2D(rate=0.4)(head_02)  #
    head_11 = Concatenate()([head_01, head_02])
    head_12 = Concatenate()([output_pre_sk_local, output_post_sk_local])
    head = Concatenate()([head_11, head_12])
    head = BatchNormalization(name='bn_sk')(head)
    head = Conv2D(16, (3, 3), padding='same', name='middle_conv')(head)
    head = SpatialDropout2D(rate=0.4)(head)  #
    head = Conv2D(5, (3, 3), padding='same', name='class_conv')(head)
    head = BatchNormalization(name='bn_final')(head)
    # --------------------------------------------------------
    output = Activation("sigmoid")(head)    # softmax
    classification_model = Model([input_pre, input_post], output, name='siamese_classification_effsk_model')
    classification_model.load_weights(r"E:\Ahmadi\PaperProject\model_siamese_dual_effnet_sk_unet_classification_5_256_26092023-16_V7_NewHead_5Class_Sigmoid_3F1D.h5")

# if classification:
#     BACKBONE = 'mobilenetv2'
#     preprocess_input = get_preprocessing(BACKBONE)
#     localization_pretrained_pre = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3), decoder_filters=(512, 256, 128, 64, 32), decoder_block_type='transpose')
#     tf.compat.v1.reset_default_graph()
#     localization_branch_pre = Model(inputs=localization_pretrained_pre.inputs, outputs=localization_pretrained_pre.get_layer('decoder_stage4b_relu').output, name='localization_branch_pre')
#     input_pre = Input(shape=(PATCH, PATCH, 3), name="pre_input")
#     pre_features = localization_branch_pre(input_pre)
#     head_pre = SKConv(M=3, r=16, L=32, G=32, convolutions='different', dropout_rate=0.01, name='local_skconv_pre')(pre_features)
#     head_pre = Conv2D(1, (3, 3), padding='same', name='localization_conv_pre')(head_pre)
#     output_pre = Activation("sigmoid")(head_pre)
#     localization_model_pre = Model(input_pre, output_pre, name='localization_SK_branch_pre')
#     localization_model_pre.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')
#     localization_sk_branch_pre = Model(inputs=localization_model_pre.inputs, outputs=localization_model_pre.get_layer('local_skconv_pre_axpby').output, name='localization_SK_pre_features')
#
#     localization_pretrained_post = Unet(BACKBONE, classes=1, encoder_weights='imagenet', encoder_freeze=True, activation='sigmoid', input_shape=(PATCH, PATCH, 3), decoder_filters=(512, 256, 128, 64, 32), decoder_block_type='transpose')
#     tf.compat.v1.reset_default_graph()
#     localization_branch_post = Model(inputs=localization_pretrained_post.inputs, outputs=localization_pretrained_post.get_layer('decoder_stage4b_relu').output, name='localization_branch_post')
#     input_post = Input(shape=(PATCH, PATCH, 3), name="post_input")
#     post_features = localization_branch_post(input_post)
#     head_post = SKConv(M=3, r=16, L=32, G=32, convolutions='different', dropout_rate=0.01, name='local_skconv_post')(post_features)
#     head_post = Conv2D(1, (3, 3), padding='same', name='localization_conv_post')(head_post)
#     output_post = Activation("sigmoid")(head_post)
#     localization_model_post = Model(input_post, output_post, name='localization_SK_branch_post')
#     localization_model_post.load_weights('E:/Ahmadi/PaperProject/model_mobilev2_sk_unet_localization_5_256_17092023_wTransposeDecoderV2.h5')
#     localization_sk_branch_post = Model(inputs=localization_model_post.inputs, outputs=localization_model_post.get_layer('local_skconv_post_axpby').output, name='localization_SK_post_features')
#
#     output_pre_sk_local = localization_sk_branch_pre(input_pre)
#     output_post_sk_local = localization_sk_branch_post(input_post)
#     # -------------------------------------------------------- New combination of features into the head.
#     head = Concatenate()([output_pre_sk_local, output_post_sk_local])
#     head = BatchNormalization(name='bn_sk')(head)
#     head = Conv2D(16, (3, 3), padding='same', name='middle_conv')(head)
#     head = Conv2D(4, (3, 3), padding='same', name='class_conv')(head)
#     head = BatchNormalization(name='bn_final')(head)
#     # --------------------------------------------------------
#     output = Activation("sigmoid")(head)  # softmax
#     classification_model = Model([input_pre, input_post], output, name='siamese_localization_sk_model')
#     classification_model.load_weights(r"E:\Ahmadi\PaperProject\model_siamese_mobilev2_sk_unet_classification_3_224_21092023-22_V3_NewHead_4Class_Softmax.h5")


# for i in range(1, 46):
for i in [12, 16, 24, 27, 30, 34]:
    if f'{i}.tif' not in GT:
        continue
    Pre = imread(PATH + f'Pre_Cropped/{i}.tif')[:, :, :3]
    Post = imread(PATH + f'Post_Cropped/{i}.tif')[:, :, :3]
    gt = imread(PATH + f'GT/{i}.tif')

    pre = preprocess_input(resize(Pre, output_shape=(1024, 1024, 3))[np.newaxis, :] * 255)
    post = preprocess_input(resize(Post, output_shape=(1024, 1024, 3))[np.newaxis, :] * 255)

    if localization:
        y_loc = localization_model.predict(pre)
        # imsave(f'E:/Ahmadi/PaperProject/bam_results/loc_{i}.tif', y_loc)
        np.save(f'E:/Ahmadi/PaperProject/bam_results/loc_{i}.npy', y_loc)
        print(y_loc.shape)

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        ax[0].imshow(Pre, alpha=0.9)
        ax[0].contour(gt > 0, 1, colors='lime', linewidths=2)
        ax[0].contour(y_loc[0, :, :, 0] > 0.5, 1, colors='b', linewidths=1)
        ax[0].set_xticks([]), ax[0].set_yticks([])  # , ax[0].set_title('Localization results overlaid on Pre-disaster image')
        ax[1].imshow(y_loc[0, :, :, 0], cmap='jet')
        ax[1].set_xticks([]), ax[1].set_yticks([])  # , ax[1].set_title('Ground truth overlaid on Pre-disaster image')
        plt.tight_layout()
        plt.show()

    if classification:
        # loc = imread(f'E:/Ahmadi/PaperProject/bam_results/loc_{i}.tif')[0, :, :, 0]
        y_hat = classification_model.predict((pre, post))
        # y_hat[:, :, 0] = y_hat[:, :, 0] * loc
        # y_hat[:, :, 1] = y_hat[:, :, 1] * loc
        # y_hat[:, :, 2] = y_hat[:, :, 2] * loc
        # y_hat[:, :, 3] = y_hat[:, :, 3] * loc

        # imsave(f'E:/Ahmadi/PaperProject/bam_results/cls_{i}.tif', y_hat[0, :, :, :])
        np.save(f'E:/Ahmadi/PaperProject/bam_results/cls_m2_{i}.npy', y_hat)
        # plt.figure()
        # plt.imshow(y_hat[0, :, :, 0])
        # plt.show()
        # exit()
        # ===============================================
        # print(y_hat.shape)
        # y_loc = np.copy(y_hat[0, :, :, 0])
        # y_hat[0, :, :, 0] = 0

        # loc = np.load(f'E:/Ahmadi/PaperProject/bam_results/loc_{i}.npy')[0, :, :, 0]
        # fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 6.5))
        # ax[0, 0].imshow(Pre)
        # ax[0, 0].contour(gt > 0, 1, colors='lime', linewidths=0.9)
        # ax[0, 0].contour(loc > 0.5, 1, colors='b', linewidths=0.7)
        #
        # ax[0, 1].imshow(loc, cmap='jet')
        # ax[0, 2].imshow(y_loc, cmap='jet')
        # ax[0, 2].contour(y_loc > 0.57, 1, colors='k', linewidths=0.8)
        # ax[0, 2].contour(y_loc > 0.52, 1, colors='k', linewidths=0.8, linestyle=':')
        #
        # # plt.imshow(y_hat[0, :, :, 1], cmap='jet'), plt.xticks([]), plt.yticks([])
        #
        # ax[1, 0].imshow(Post)
        # ax[1, 1].imshow(np.argmax(y_hat[0, :, :, :], axis=-1) * (loc > 0.5), cmap=c, norm=n)
        # ax[1, 2].imshow(y_hat[0, :, :, 4], cmap='jet')
        #
        # ax[0, 0].set_xticks([]), ax[0, 0].set_yticks([])
        # ax[0, 1].set_xticks([]), ax[0, 1].set_yticks([])
        # ax[0, 2].set_xticks([]), ax[0, 2].set_yticks([])
        # ax[1, 0].set_xticks([]), ax[1, 0].set_yticks([])
        # ax[1, 1].set_xticks([]), ax[1, 1].set_yticks([])
        # ax[1, 2].set_xticks([]), ax[1, 2].set_yticks([])
        # plt.tight_layout()
        # # figManager = plt.get_current_fig_manager()
        # # figManager.resize(*figManager.window.maxsize())
        # plt.show()
        # ===============================================



    # fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
    # ax[0].imshow(Pre)
    # ax[1].imshow(Post)
    # # ax[2].imshow(y_hat[0, :, :, 0])
    # ax[2].imshow(Pre)
    # ax[2].contour(y_loc[0, :, :, 0] > 0.5, 1, colors='k', linewidths=1)
    # # ax[2].contour(y_hat[0, :, :, 0] > 0.35, 1, colors='w', linewidths=1)
    # ax[3].imshow(gt > 0, cmap='gray')
    # # ax[3].imshow(gt, vmin=0, vmax=4)
    # ax[0].set_xticks([]), ax[0].set_yticks([]), ax[0].set_title('Pre-disaster image')
    # ax[1].set_xticks([]), ax[1].set_yticks([]), ax[1].set_title('Post-disaster image')
    # ax[2].set_xticks([]), ax[2].set_yticks([]), ax[2].set_title('Localization')
    # ax[3].set_xticks([]), ax[3].set_yticks([]), ax[3].set_title('Ground truth')
    # plt.tight_layout()
    # plt.show()


    # loc = y_hat[0, :, :, 0] > 0.35
    # fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    # ax[0, 1].imshow(y_hat[0, :, :, 1] * loc)
    # ax[0, 2].imshow(y_hat[0, :, :, 2] * loc)
    # ax[1, 0].imshow(y_hat[0, :, :, 3] * loc)
    # ax[1, 1].imshow(y_hat[0, :, :, 4] * loc)
    # y_hat[0, :, :, 0] = 0
    # ax[0, 0].imshow(np.argmax(y_hat[0], axis=2) * loc, vmin=0, vmax=4)
    # ax[1, 2].imshow(post[0])
    # plt.tight_layout()
    # plt.show()



# conda activate E:/Ahmadi/KerasEnv
# cd E:/Ahmadi/PaperProject