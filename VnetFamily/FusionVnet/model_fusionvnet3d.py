'''

'''
from .layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, weight_xavier_init,
                    bias_variable)
import tensorflow as tf


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=16, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=16, scope=scope)
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


def conv_sigmoid(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.sigmoid(conv)
        return conv


def _createfusionnet(X1, X2, X3, X4, image_z, image_width, image_height, image_channel, phase, drop, n_class=1):
    inputX1 = tf.reshape(X1, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    inputX2 = tf.reshape(X2, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    inputX3 = tf.reshape(X3, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    inputX4 = tf.reshape(X4, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0_1 = conv_bn_relu_drop(x=inputX1, kernal=(3, 3, 3, image_channel, 16), phase=phase, drop=drop,
                                 scope='layer0_1')
    layer1_1 = conv_bn_relu_drop(x=layer0_1, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop,
                                 scope='layer1_1')
    layer1_1 = resnet_Add(x1=layer0_1, x2=layer1_1)
    layer0_2 = conv_bn_relu_drop(x=inputX2, kernal=(3, 3, 3, image_channel, 16), phase=phase, drop=drop,
                                 scope='layer0_2')
    layer1_2 = conv_bn_relu_drop(x=layer0_2, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop,
                                 scope='layer1_2')
    layer1_2 = resnet_Add(x1=layer0_2, x2=layer1_2)
    layer0_3 = conv_bn_relu_drop(x=inputX3, kernal=(3, 3, 3, image_channel, 16), phase=phase, drop=drop,
                                 scope='layer0_3')
    layer1_3 = conv_bn_relu_drop(x=layer0_3, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop,
                                 scope='layer1_3')
    layer1_3 = resnet_Add(x1=layer0_3, x2=layer1_3)
    layer0_4 = conv_bn_relu_drop(x=inputX4, kernal=(3, 3, 3, image_channel, 16), phase=phase, drop=drop,
                                 scope='layer0_4')
    layer1_4 = conv_bn_relu_drop(x=layer0_4, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop,
                                 scope='layer1_4')
    layer1_4 = resnet_Add(x1=layer0_4, x2=layer1_4)

    layer1 = crop_and_concat(crop_and_concat(layer1_1, layer1_2), crop_and_concat(layer1_3, layer1_4))
    layer1 = conv_bn_relu_drop(x=layer1, kernal=(3, 3, 3, 16 * 4, 16), phase=phase, drop=drop,
                               scope='layer1')
    # down sampling1
    down1_1 = down_sampling(x=layer1_1, kernal=(3, 3, 3, 16, 32), phase=phase, drop=drop, scope='down1_1')
    down1_2 = down_sampling(x=layer1_2, kernal=(3, 3, 3, 16, 32), phase=phase, drop=drop, scope='down1_2')
    down1_3 = down_sampling(x=layer1_3, kernal=(3, 3, 3, 16, 32), phase=phase, drop=drop, scope='down1_3')
    down1_4 = down_sampling(x=layer1_4, kernal=(3, 3, 3, 16, 32), phase=phase, drop=drop, scope='down1_4')
    # layer2->convolution
    layer2_1 = conv_bn_relu_drop(x=down1_1, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_1')
    layer2_1 = conv_bn_relu_drop(x=layer2_1, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_2')
    layer2_1 = resnet_Add(x1=down1_1, x2=layer2_1)
    layer2_2 = conv_bn_relu_drop(x=down1_2, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_3')
    layer2_2 = conv_bn_relu_drop(x=layer2_2, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_4')
    layer2_2 = resnet_Add(x1=down1_2, x2=layer2_2)
    layer2_3 = conv_bn_relu_drop(x=down1_3, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_5')
    layer2_3 = conv_bn_relu_drop(x=layer2_3, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_6')
    layer2_3 = resnet_Add(x1=down1_3, x2=layer2_3)
    layer2_4 = conv_bn_relu_drop(x=down1_4, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_7')
    layer2_4 = conv_bn_relu_drop(x=layer2_4, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                                 scope='layer2_8')
    layer2_4 = resnet_Add(x1=down1_4, x2=layer2_4)

    layer2 = crop_and_concat(crop_and_concat(layer2_1, layer2_2), crop_and_concat(layer2_3, layer2_4))
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 32 * 4, 32), phase=phase, drop=drop,
                               scope='layer2')
    # down sampling2
    down2_1 = down_sampling(x=layer2_1, kernal=(3, 3, 3, 32, 64), phase=phase, drop=drop, scope='down2_1')
    down2_2 = down_sampling(x=layer2_2, kernal=(3, 3, 3, 32, 64), phase=phase, drop=drop, scope='down2_2')
    down2_3 = down_sampling(x=layer2_3, kernal=(3, 3, 3, 32, 64), phase=phase, drop=drop, scope='down2_3')
    down2_4 = down_sampling(x=layer2_4, kernal=(3, 3, 3, 32, 64), phase=phase, drop=drop, scope='down2_4')
    # layer3->convolution
    layer3_1 = conv_bn_relu_drop(x=down2_1, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_1')
    layer3_1 = conv_bn_relu_drop(x=layer3_1, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_2')
    layer3_1 = conv_bn_relu_drop(x=layer3_1, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_2_1')
    layer3_1 = resnet_Add(x1=down2_1, x2=layer3_1)
    layer3_2 = conv_bn_relu_drop(x=down2_2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_3')
    layer3_2 = conv_bn_relu_drop(x=layer3_2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_4')
    layer3_2 = conv_bn_relu_drop(x=layer3_2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_4_1')
    layer3_2 = resnet_Add(x1=down2_2, x2=layer3_2)
    layer3_3 = conv_bn_relu_drop(x=down2_3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_5')
    layer3_3 = conv_bn_relu_drop(x=layer3_3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_6')
    layer3_3 = conv_bn_relu_drop(x=layer3_3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_6_1')
    layer3_3 = resnet_Add(x1=down2_3, x2=layer3_3)
    layer3_4 = conv_bn_relu_drop(x=down2_4, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_7')
    layer3_4 = conv_bn_relu_drop(x=layer3_4, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_8')
    layer3_4 = conv_bn_relu_drop(x=layer3_4, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                                 scope='layer3_8_1')
    layer3_4 = resnet_Add(x1=down2_4, x2=layer3_4)

    layer3 = crop_and_concat(crop_and_concat(layer3_1, layer3_2), crop_and_concat(layer3_3, layer3_4))
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64 * 4, 64), phase=phase, drop=drop,
                               scope='layer3')
    # down sampling3
    down3_1 = down_sampling(x=layer3_1, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3_1')
    down3_2 = down_sampling(x=layer3_2, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3_2')
    down3_3 = down_sampling(x=layer3_3, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3_3')
    down3_4 = down_sampling(x=layer3_4, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3_4')
    # layer4->convolution
    layer4_1 = conv_bn_relu_drop(x=down3_1, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_1')
    layer4_1 = conv_bn_relu_drop(x=layer4_1, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_2')
    layer4_1 = conv_bn_relu_drop(x=layer4_1, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_2_1')
    layer4_1 = resnet_Add(x1=down3_1, x2=layer4_1)
    layer4_2 = conv_bn_relu_drop(x=down3_2, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_3')
    layer4_2 = conv_bn_relu_drop(x=layer4_2, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_4')
    layer4_2 = conv_bn_relu_drop(x=layer4_2, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_4_1')
    layer4_2 = resnet_Add(x1=down3_2, x2=layer4_2)
    layer4_3 = conv_bn_relu_drop(x=down3_3, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_5')
    layer4_3 = conv_bn_relu_drop(x=layer4_3, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_6')
    layer4_3 = conv_bn_relu_drop(x=layer4_3, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_6_1')
    layer4_3 = resnet_Add(x1=down3_3, x2=layer4_3)
    layer4_4 = conv_bn_relu_drop(x=down3_4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_7')
    layer4_4 = conv_bn_relu_drop(x=layer4_4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_8')
    layer4_4 = conv_bn_relu_drop(x=layer4_4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                                 scope='layer4_8_1')
    layer4_4 = resnet_Add(x1=down3_4, x2=layer4_4)

    layer4 = crop_and_concat(crop_and_concat(layer4_1, layer4_2), crop_and_concat(layer4_3, layer4_4))
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128 * 4, 128), phase=phase, drop=drop,
                               scope='layer4')
    # down sampling4
    down4_1 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4_1')
    down4_2 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4_2')
    down4_3 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4_3')
    down4_4 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4_4')
    # layer5->convolution
    layer5_1 = conv_bn_relu_drop(x=down4_1, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_1')
    layer5_1 = conv_bn_relu_drop(x=layer5_1, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_2')
    layer5_1 = conv_bn_relu_drop(x=layer5_1, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_2_1')
    layer5_1 = resnet_Add(x1=down4_1, x2=layer5_1)
    layer5_2 = conv_bn_relu_drop(x=down4_2, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_3')
    layer5_2 = conv_bn_relu_drop(x=layer5_2, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_4')
    layer5_2 = conv_bn_relu_drop(x=layer5_2, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_4_1')
    layer5_2 = resnet_Add(x1=down4_2, x2=layer5_2)
    layer5_3 = conv_bn_relu_drop(x=down4_3, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_5')
    layer5_3 = conv_bn_relu_drop(x=layer5_3, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_6')
    layer5_3 = conv_bn_relu_drop(x=layer5_3, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_6_1')
    layer5_3 = resnet_Add(x1=down4_3, x2=layer5_3)
    layer5_4 = conv_bn_relu_drop(x=down4_4, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_7')
    layer5_4 = conv_bn_relu_drop(x=layer5_4, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_8')
    layer5_4 = conv_bn_relu_drop(x=layer5_4, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                                 scope='layer5_8_1')
    layer5_4 = resnet_Add(x1=down4_4, x2=layer5_4)

    layer5 = crop_and_concat(crop_and_concat(layer5_1, layer5_2), crop_and_concat(layer5_3, layer5_4))
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256 * 4, 256), phase=phase, drop=drop,
                               scope='layer5')
    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 128, 256), scope='deconv1')
    # layer8->convolution
    layer6 = crop_and_concat(layer4, deconv1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 256, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_3')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 64, 128), scope='deconv2')
    # layer8->convolution
    layer7 = crop_and_concat(layer3, deconv2)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 128, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_3')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 32, 64), scope='deconv3')
    # layer8->convolution
    layer8 = crop_and_concat(layer2, deconv3)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 64, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 16, 32), scope='deconv4')
    # layer8->convolution
    layer9 = crop_and_concat(layer1, deconv4)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 32, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 16, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_sigmoid(x=layer9, kernal=(1, 1, 1, 16, n_class), scope='output')

    return output_map
