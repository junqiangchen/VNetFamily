'''

'''
from .layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add,
                        weight_xavier_init, bias_variable)
import tensorflow as tf
import numpy as np
import os


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=20, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def full_connected_relu(x, kernal, activefunction='relu', scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        FC = tf.matmul(x, W) + B
        if activefunction == 'relu':
            FC = tf.nn.relu(FC)
        elif activefunction == 'softmax':
            FC = tf.nn.softmax(FC)
        elif activefunction == 'sigmoid':
            FC = tf.nn.sigmoid(FC)
        return FC


def conv_sigomd(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.sigmoid(conv)
        return conv


def squeeze_excitation_model(x, out_dim, name='ssceaddcsse', ratio=4, scope=None):
    with tf.name_scope(scope):
        if name == 'sscemaxcsse':
            recalibrate1 = spatial_squeeze_channel_excitation_layer(x, out_dim, ratio, scope=scope + 'ssce')
            recalibrate2 = channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=scope + 'csse')
            recalibrate = tf.maximum(recalibrate1, recalibrate2)
            return recalibrate
        if name == 'ssceaddcsse':
            recalibrate1 = spatial_squeeze_channel_excitation_layer(x, out_dim, ratio, scope=scope + 'ssce')
            recalibrate2 = channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=scope + 'csse')
            recalibrate = tf.add(recalibrate1, recalibrate2)
            return recalibrate
        if name == 'sscemutiplycsse':
            recalibrate1 = spatial_squeeze_channel_excitation_layer(x, out_dim, ratio, scope=scope + 'ssce')
            recalibrate2 = channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=scope + 'csse')
            recalibrate = tf.multiply(recalibrate1, recalibrate2)
            return recalibrate
        if name == 'ssceconcatcsse':
            recalibrate1 = spatial_squeeze_channel_excitation_layer(x, out_dim, ratio, scope=scope + 'ssce')
            recalibrate2 = channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=scope + 'csse')
            recalibrate = crop_and_concat(recalibrate1, recalibrate2)
            recalibrate = conv_sigomd(recalibrate, kernal=(1, 1, 1, out_dim * 2, out_dim),
                                      scope=scope + 'ssceconcatcsse')
            return recalibrate


def spatial_squeeze_channel_excitation_layer(x, out_dim, ratio=4, scope=None):
    with tf.name_scope(scope):
        # Global_Average_Pooling,channel_squeeze
        squeeze = tf.reduce_mean(x, axis=(1, 2, 3), name=scope + 'channel_squeeze')
        # full_connect
        excitation = full_connected_relu(squeeze, kernal=(out_dim, out_dim // ratio), activefunction='relu',
                                         scope=scope + '_fully_connected1')
        excitation = full_connected_relu(excitation, kernal=(out_dim // ratio, out_dim),
                                         activefunction='sigmoid', scope=scope + '_fully_connected2')
        # scale the x
        excitation = tf.reshape(excitation, [-1, 1, 1, 1, out_dim])
        scale = x * excitation
        return scale


def channel_squeeze_spatial_excitiation_layer(x, out_dim, scope=None):
    with tf.name_scope(scope):
        squeeze = conv_sigomd(x, kernal=(1, 1, 1, out_dim, 1), scope=scope + 'spatial_squeeze')
        scale = x * squeeze
        return scale


def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=20, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv


def _createscsevnet(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=1):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 20), phase=phase, drop=drop,
                               scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 3, 20, 20), phase=phase, drop=drop,
                               scope='layer1')
    layer1 = squeeze_excitation_model(layer1, out_dim=20, scope='sem1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 20, 40), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop,
                               scope='layer2_2')
    layer2 = squeeze_excitation_model(layer2, out_dim=40, scope='sem2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 40, 80), phase=phase, drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_2')
    layer3 = squeeze_excitation_model(layer3, out_dim=80, scope='sem3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 80, 160), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_2')
    layer4 = squeeze_excitation_model(layer4, out_dim=160, scope='sem4')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 160, 320), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = squeeze_excitation_model(layer5, out_dim=320, scope='sem5')
    layer5 = resnet_Add(x1=down4, x2=layer5)

    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 160, 320), scope='deconv1')
    # layer8->convolution
    layer6 = crop_and_concat(layer4, deconv1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 320, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = squeeze_excitation_model(layer6, out_dim=160, scope='sem6')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 80, 160), scope='deconv2')
    # layer8->convolution
    layer7 = crop_and_concat(layer3, deconv2)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 160, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = squeeze_excitation_model(layer7, out_dim=80, scope='sem7')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 40, 80), scope='deconv3')
    # layer8->convolution
    layer8 = crop_and_concat(layer2, deconv3)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 80, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = squeeze_excitation_model(layer8, out_dim=40, scope='sem8')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 20, 40), scope='deconv4')
    # layer8->convolution
    layer9 = crop_and_concat(layer1, deconv4)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 40, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = squeeze_excitation_model(layer9, out_dim=20, scope='sem9')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_sigomd(x=layer9, kernal=(1, 1, 1, 20, n_class), scope='output')
    return output_map
