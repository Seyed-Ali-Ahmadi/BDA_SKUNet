from tensorflow.keras import backend as K
import tensorflow as tf
from keras.losses import BinaryCrossentropy


beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1


class Semantic_loss_functions(object):
    def __init__(self):
        print("semantic loss functions initialized")

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coef(y_true, y_pred)
        return loss

    # ------------------------------------------------------------------------------------------------------------------
    def dice_loss_masked(self, y_true, y_pred):
        y_pred = tf.math.multiply(tf.cast(y_true > 0, tf.float32), y_pred)
        loss = 1 - self.dice_coef(y_true, y_pred)
        return loss
    # ------------------------------------------------------------------------------------------------------------------

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))
        loss = self.focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)

    # ------------------------------------------------------------------------------------------------------------------
    def focal_loss_masked(self, y_true, y_pred):
        y_pred = tf.math.multiply(tf.cast(y_true > 0, tf.float32), y_pred)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))
        loss = self.focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)
    # ------------------------------------------------------------------------------------------------------------------

    def tversky_index(self, y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def jacard_similarity(self, y_true, y_pred):
        """ Intersection-Over-Union (IoU), also known as the Jaccard Index ."""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        return intersection / union

    def jacard_loss(self, y_true, y_pred):
        return 1 - self.jacard_similarity(y_true, y_pred)

    def ssim_loss(self, y_true, y_pred):
        """ Structural Similarity Index (SSIM) loss """
        return 1 - tf.image.ssim(y_true, y_pred, max_val=1)

    def unet3p_hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss proposed in UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
        Hybrid loss for segmentation in three-level hierarchy – pixel, patch and map-level,
        which is able to capture both large-scale and fine structures with clear boundaries.
        """
        focal_loss = self.focal_loss(y_true, y_pred)
        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)
        return focal_loss + ms_ssim_loss + jacard_loss

    def basnet_hybrid_loss(self, y_true, y_pred):
        """
        Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
        The hybrid loss is a combination of the binary cross entropy, structural similarity
        and intersection-over-union losses, which guide the network to learn
        three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
        """
        bce_loss = BinaryCrossentropy(from_logits=False)
        bce_loss = bce_loss(y_true, y_pred)

        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)
        return bce_loss + ms_ssim_loss + jacard_loss


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss


# ------------------------------------------------------------------------------------------------------------------
def wcce_masked(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred = tf.math.multiply(tf.cast(y_true > 0, tf.float32), y_pred)
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss
# ------------------------------------------------------------------------------------------------------------------


# Different metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



