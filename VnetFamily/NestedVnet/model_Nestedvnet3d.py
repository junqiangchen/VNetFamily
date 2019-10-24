'''

'''
from .layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, weight_xavier_init,
                    bias_variable)
import tensorflow as tf


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    """
    :param x:
    :param kernal:
    :param phase:
    :param drop:
    :param image_z:
    :param height:
    :param width:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'conv_W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'conv_B')
        conv = conv3d(x, W) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
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
                                  scope=scope)
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


def conv_sigmod(x, kernal, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        conv = tf.nn.sigmoid(conv)
        return conv


def _createNestednet(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=1):
    inputX = tf.reshape(X, [-1, image_z, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # Vnet model
    # layer1->convolution
    layer0 = conv_bn_relu_drop(x=inputX, kernal=(3, 3, 3, image_channel, 16), phase=phase, drop=drop,
                               scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop,
                               scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernal=(3, 3, 3, 16, 32), phase=phase, drop=drop, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop,
                               scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernal=(3, 3, 3, 32, 64), phase=phase, drop=drop, scope='down2')
    # Nested block1
    deconv1_1 = deconv_relu(x=layer2, kernal=(3, 3, 3, 16, 32), scope='deconv1_1')
    layer1_1 = crop_and_concat(layer1, deconv1_1)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer1_1 = conv_bn_relu_drop(x=layer1_1, kernal=(3, 3, 3, 32, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_1_1')
    layer1_1 = conv_bn_relu_drop(x=layer1_1, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_1_2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop, scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3')
    # Nested block2
    deconv2_1 = deconv_relu(x=layer3, kernal=(3, 3, 3, 32, 64), scope='deconv2_1')
    layer2_1 = crop_and_concat(layer2, deconv2_1)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer2_1 = conv_bn_relu_drop(x=layer2_1, kernal=(3, 3, 3, 64, 32), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer2_1_1')
    layer2_1 = conv_bn_relu_drop(x=layer2_1, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer2_1_2')
    # Nested block3
    deconv1_2 = deconv_relu(x=layer2_1, kernal=(3, 3, 3, 16, 32), scope='deconv1_2')
    layer1_2 = crop_and_concat(layer1_1, deconv1_2)
    layer1_2 = crop_and_concat(layer1_2, layer1)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer1_2 = conv_bn_relu_drop(x=layer1_2, kernal=(3, 3, 3, 16 * 3, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_2_1')
    layer1_2 = conv_bn_relu_drop(x=layer1_2, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_2_2')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop, scope='layer4_3')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4')
    # Nested block4
    deconv3_1 = deconv_relu(x=layer4, kernal=(3, 3, 3, 64, 128), scope='deconv3_1')
    layer3_1 = crop_and_concat(layer3, deconv3_1)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer3_1 = conv_bn_relu_drop(x=layer3_1, kernal=(3, 3, 3, 128, 64), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer3_1_1')
    layer3_1 = conv_bn_relu_drop(x=layer3_1, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer3_1_2')
    # Nested block5
    deconv2_2 = deconv_relu(x=layer3_1, kernal=(3, 3, 3, 32, 64), scope='deconv2_2')
    layer2_2 = crop_and_concat(layer2_1, deconv2_2)
    layer2_2 = crop_and_concat(layer2_2, layer2)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer2_2 = conv_bn_relu_drop(x=layer2_2, kernal=(3, 3, 3, 32 * 3, 32), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer2_2_1')
    layer2_2 = conv_bn_relu_drop(x=layer2_2, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer2_2_2')
    # Nested block6
    deconv1_3 = deconv_relu(x=layer2_2, kernal=(3, 3, 3, 16, 32), scope='deconv1_3')
    layer1_3 = crop_and_concat(layer1_2, deconv1_3)
    layer1_3 = crop_and_concat(layer1_3, layer1_1)
    layer1_3 = crop_and_concat(layer1_3, layer1)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer1_3 = conv_bn_relu_drop(x=layer1_3, kernal=(3, 3, 3, 16 * 4, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_3_1')
    layer1_3 = conv_bn_relu_drop(x=layer1_3, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_3_2')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop, scope='layer5_3')
    layer5 = resnet_Add(x1=down4, x2=layer5)
    # Nested block7
    deconv4_1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 128, 256), scope='deconv4_1')
    layer4_1 = crop_and_concat(layer4, deconv4_1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer4_1 = conv_bn_relu_drop(x=layer4_1, kernal=(3, 3, 3, 256, 128), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer4_1_1')
    layer4_1 = conv_bn_relu_drop(x=layer4_1, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer4_1_2')
    # Nested block8
    deconv3_2 = deconv_relu(x=layer4_1, kernal=(3, 3, 3, 64, 128), scope='deconv3_2')
    layer3_2 = crop_and_concat(layer3_1, deconv3_2)
    layer3_2 = crop_and_concat(layer3_2, layer3)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer3_2 = conv_bn_relu_drop(x=layer3_2, kernal=(3, 3, 3, 64 * 3, 64), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer3_2_1')
    layer3_2 = conv_bn_relu_drop(x=layer3_2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer3_2_2')
    # Nested block9
    deconv2_3 = deconv_relu(x=layer3_2, kernal=(3, 3, 3, 32, 64), scope='deconv2_3')
    layer2_3 = crop_and_concat(layer2_2, deconv2_3)
    layer2_3 = crop_and_concat(layer2_3, layer2_1)
    layer2_3 = crop_and_concat(layer2_3, layer2)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer2_3 = conv_bn_relu_drop(x=layer2_3, kernal=(3, 3, 3, 32 * 4, 32), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer2_3_1')
    layer2_3 = conv_bn_relu_drop(x=layer2_3, kernal=(3, 3, 3, 32, 32), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer2_3_2')
    # Nested block10
    deconv1_4 = deconv_relu(x=layer2_3, kernal=(3, 3, 3, 16, 32), scope='deconv1_4')
    layer1_4 = crop_and_concat(layer1_3, deconv1_4)
    layer1_4 = crop_and_concat(layer1_4, layer1_2)
    layer1_4 = crop_and_concat(layer1_4, layer1_1)
    layer1_4 = crop_and_concat(layer1_4, layer1)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer1_4 = conv_bn_relu_drop(x=layer1_4, kernal=(3, 3, 3, 16 * 5, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_4_1')
    layer1_4 = conv_bn_relu_drop(x=layer1_4, kernal=(3, 3, 3, 16, 16), phase=phase, drop=drop, image_z=Z,
                                 height=H, width=W, scope='layer1_4_2')
    # layer14->output
    output_map1 = conv_sigmod(x=layer1_1, kernal=(1, 1, 1, 16, n_class), scope='output1')
    output_map2 = conv_sigmod(x=layer1_2, kernal=(1, 1, 1, 16, n_class), scope='output2')
    output_map3 = conv_sigmod(x=layer1_3, kernal=(1, 1, 1, 16, n_class), scope='output3')
    output_map4 = conv_sigmod(x=layer1_4, kernal=(1, 1, 1, 16, n_class), scope='output4')
    return output_map1, output_map2, output_map3, output_map4
