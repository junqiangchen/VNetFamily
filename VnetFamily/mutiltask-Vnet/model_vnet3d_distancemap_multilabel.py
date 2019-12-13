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
                                  G=20, scope=scope)
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


def conv_softmax(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.softmax(conv)
        return conv


def conv_sigmoid(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.sigmoid(conv)
        return conv


def _create_conv_net(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=2):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 20), phase=phase, drop=drop,
                               scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 3, 20, 20), phase=phase, drop=drop,
                               scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 20, 40), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 40, 40), phase=phase, drop=drop,
                               scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 40, 80), phase=phase, drop=drop, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 80, 80), phase=phase, drop=drop,
                               scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 80, 160), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 160, 160), phase=phase, drop=drop,
                               scope='layer4_3')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 160, 320), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 320, 320), phase=phase, drop=drop,
                               scope='layer5_3')
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
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_softmax(x=layer9, kernal=(1, 1, 1, 20, n_class), scope='output_map')

    # layer9->deconvolution
    deconv1_1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 160, 320), scope='deconv1_1')
    # layer8->convolution
    layer6_1 = crop_and_concat(layer4, deconv1_1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer6_1 = conv_bn_relu_drop(x=layer6_1, kernal=(3, 3, 3, 320, 160), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer6_1_1')
    layer6_1 = conv_bn_relu_drop(x=layer6_1, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer6_2_1')
    layer6_1 = resnet_Add(x1=deconv1_1, x2=layer6_1)
    # layer9->deconvolution
    deconv2_1 = deconv_relu(x=layer6_1, kernal=(3, 3, 3, 80, 160), scope='deconv2_1')
    # layer8->convolution
    layer7_1 = crop_and_concat(layer3, deconv2_1)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer7_1 = conv_bn_relu_drop(x=layer7_1, kernal=(3, 3, 3, 160, 80), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer7_1_1')
    layer7_1 = conv_bn_relu_drop(x=layer7_1, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer7_2_1')
    layer7_1 = resnet_Add(x1=deconv2_1, x2=layer7_1)
    # layer9->deconvolution
    deconv3_1 = deconv_relu(x=layer7_1, kernal=(3, 3, 3, 40, 80), scope='deconv3_1')
    # layer8->convolution
    layer8_1 = crop_and_concat(layer2, deconv3_1)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer8_1 = conv_bn_relu_drop(x=layer8_1, kernal=(3, 3, 3, 80, 40), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer8_1_1')
    layer8_1 = conv_bn_relu_drop(x=layer8_1, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer8_2_1')
    layer8_1 = resnet_Add(x1=deconv3_1, x2=layer8_1)
    # layer9->deconvolution
    deconv4_1 = deconv_relu(x=layer8_1, kernal=(3, 3, 3, 20, 40), scope='deconv4_1')
    # layer8->convolution
    layer9_1 = crop_and_concat(layer1, deconv4_1)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer9_1 = conv_bn_relu_drop(x=layer9_1, kernal=(3, 3, 3, 40, 20), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer9_1_1')
    layer9_1 = conv_bn_relu_drop(x=layer9_1, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W, phase=phase,
                                 drop=drop, scope='layer9_2_1')
    layer9_1 = resnet_Add(x1=deconv4_1, x2=layer9_1)
    # layer14->output
    output_distance_map = conv_sigmoid(x=layer9_1, kernal=(1, 1, 1, 20, n_class - 1), scope='output_distance_map')

    return output_map, output_distance_map
