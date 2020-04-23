'''

'''
from Vnet.layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, max_pool3d, upsample3d,
                        weight_xavier_init, bias_variable)
import tensorflow as tf


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=4, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def conv_relu(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.relu(conv)
        return conv



def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  G=4, scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop)
        return conv


def conv_sigmod(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.sigmod(conv)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv


def _create_convtripleplus_net(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=2):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnettripleplus model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 20), phase=phase, drop=drop, scope='layer1_1')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 3, 20, 20), phase=phase, drop=drop, scope='layer1_2')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 20, 40), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop, scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop, scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 40, 80), phase=phase, drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop, scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop, scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop, scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 80, 160), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop, scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop, scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop, scope='layer4_4')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 160, 320), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop, scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop, scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop, scope='layer5_3')
    layer5 = resnet_Add(x1=down4, x2=layer5)

    _, Z, H, W, _ = layer4.get_shape().as_list()
    # layer9->decode1
    upsample1_1 = upsample3d(x=layer5, scale_factor=2, scope='upsample1_1')
    decode1_1 = conv_bn_relu_drop(upsample1_1, kernal=(3, 3, 3, 160, 32), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode1_1')
    decode1_2 = conv_bn_relu_drop(layer4, kernal=(3, 3, 3, 160, 32), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode1_2')
    decode1_3 = max_pool3d(x=layer3, depth=True)
    decode1_3 = conv_bn_relu_drop(decode1_3, kernal=(3, 3, 3, 80, 32), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode1_3')
    decode1_4 = max_pool3d(x=layer2, depth=True)
    decode1_4 = max_pool3d(x=decode1_4, depth=True)
    decode1_4 = conv_bn_relu_drop(decode1_4, kernal=(3, 3, 3, 40, 32), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode1_4')
    decode1_5 = max_pool3d(x=layer1, depth=True)
    decode1_5 = max_pool3d(x=decode1_5, depth=True)
    decode1_5 = max_pool3d(x=decode1_5, depth=True)
    decode1_5 = conv_bn_relu_drop(decode1_5, kernal=(3, 3, 3, 20, 32), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode1_5')

    decode1 = tf.concat([decode1_1, decode1_2, decode1_3, decode1_4, decode1_5], axis=4)
    decode1 = conv_bn_relu_drop(x=decode1, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                                drop=drop, scope='decode1_6')
    # layer9->decode2
    _, Z, H, W, _ = layer3.get_shape().as_list()
    upsample2_1 = upsample3d(x=layer5, scale_factor=4, scope='upsample2_1')
    decode2_1 = conv_bn_relu_drop(upsample2_1, kernal=(3, 3, 3, 80, 16), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode2_1')
    decode2_2 = upsample3d(x=decode1, scale_factor=2, scope='upsample2_2')
    decode2_2 = conv_bn_relu_drop(decode2_2, kernal=(3, 3, 3, 80, 16), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode2_2')
    decode2_3 = conv_bn_relu_drop(layer3, kernal=(3, 3, 3, 80, 16), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode2_3')
    decode2_4 = max_pool3d(x=layer2, depth=True)
    decode2_4 = conv_bn_relu_drop(decode2_4, kernal=(3, 3, 3, 40, 16), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode2_4')
    decode2_5 = max_pool3d(x=layer1, depth=True)
    decode2_5 = max_pool3d(x=decode2_5, depth=True)
    decode2_5 = conv_bn_relu_drop(decode2_5, kernal=(3, 3, 3, 20, 16), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode2_5')

    decode2 = tf.concat([decode2_1, decode2_2, decode2_3, decode2_4, decode2_5], axis=4)
    decode2 = conv_bn_relu_drop(x=decode2, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                                drop=drop, scope='decode2_6')
    # layer9->decode3
    _, Z, H, W, _ = layer2.get_shape().as_list()
    upsample3_1 = upsample3d(x=layer5, scale_factor=8, scope='upsample3_1')
    decode3_1 = conv_bn_relu_drop(upsample3_1, kernal=(3, 3, 3, 40, 8), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode3_1')
    decode3_2 = upsample3d(x=decode1, scale_factor=4, scope='upsample3_2')
    decode3_2 = conv_bn_relu_drop(decode3_2, kernal=(3, 3, 3, 40, 8), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode3_2')
    decode3_3 = upsample3d(x=decode2, scale_factor=2, scope='upsample3_3')
    decode3_3 = conv_bn_relu_drop(decode3_3, kernal=(3, 3, 3, 40, 8), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode3_3')
    decode3_4 = conv_bn_relu_drop(layer2, kernal=(3, 3, 3, 40, 8), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode3_4')
    decode3_5 = max_pool3d(x=layer1, depth=True)
    decode3_5 = conv_bn_relu_drop(decode3_5, kernal=(3, 3, 3, 20, 8), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode3_5')

    decode3 = tf.concat([decode3_1, decode3_2, decode3_3, decode3_4, decode3_5], axis=4)
    decode3 = conv_bn_relu_drop(x=decode3, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                                drop=drop, scope='decode3_6')
    # layer9->decode4
    _, Z, H, W, _ = layer1.get_shape().as_list()
    upsample4_1 = upsample3d(x=layer5, scale_factor=16, scope='upsample4_1')
    decode4_1 = conv_bn_relu_drop(upsample4_1, kernal=(3, 3, 3, 20, 4), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode4_1')
    decode4_2 = upsample3d(x=decode1, scale_factor=8, scope='upsample4_2')
    decode4_2 = conv_bn_relu_drop(decode4_2, kernal=(3, 3, 3, 20, 4), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode4_2')
    decode4_3 = upsample3d(x=decode2, scale_factor=4, scope='upsample4_3')
    decode4_3 = conv_bn_relu_drop(decode4_3, kernal=(3, 3, 3, 20, 4), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode4_3')
    decode4_4 = upsample3d(x=decode3, scale_factor=2, scope='upsample4_4')
    decode4_4 = conv_bn_relu_drop(decode4_4, kernal=(3, 3, 3, 20, 4), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode4_4')
    decode4_5 = conv_bn_relu_drop(layer1, kernal=(3, 3, 3, 20, 4), image_z=Z, height=H, width=W, phase=phase,
                                  drop=drop, scope='decode4_5')

    decode4 = tf.concat([decode4_1, decode4_2, decode4_3, decode4_4, decode4_5], axis=4)
    decode4 = conv_bn_relu_drop(x=decode4, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W,
                                phase=phase, drop=drop, scope='decode4_6')
    # layer14->output
    output_map = conv_sigmod(x=decode4, kernal=(1, 1, 1, 20, n_class), scope='output')
    return output_map