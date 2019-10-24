'''

'''
from Vnet.layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, upsample3d,
                        weight_xavier_init, bias_variable, save_images)
import tensorflow as tf
import numpy as np
import os


def gatingsignal3d(x, kernal, phase, image_z=None, height=None, width=None, scope=None):
    """this is simply 1x1x1 convolution, bn, activation,Gating Signal(Query)
    :param x:
    :param kernal:(1,1,1,inputfilters,outputfilters)
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
        conv = tf.nn.relu(conv)
        return conv


def attngatingblock(x, g, inputfilters, outfilters, scale_factor, phase, image_z=None, height=None, width=None,
                    scope=None):
    """
    take g which is the spatially smaller signal, do a conv to get the same number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsample g to be same size as x add x and g (concat_xg) relu, 1x1x1 conv, then sigmoid then upsample the final -
    this gives us attn coefficients
    :param x:
    :param g:
    :param inputfilters:
    :param outfilters:
    :param scale_factor:2
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        kernalx = (1, 1, 1, inputfilters, outfilters)
        Wx = weight_xavier_init(shape=kernalx, n_inputs=kernalx[0] * kernalx[1] * kernalx[2] * kernalx[3],
                                n_outputs=kernalx[-1], activefunction='relu', variable_name=scope + 'conv_Wx')
        Bx = bias_variable([kernalx[-1]], variable_name=scope + 'conv_Bx')
        theta_x = conv3d(x, Wx, scale_factor) + Bx
        kernalg = (1, 1, 1, inputfilters, outfilters)
        Wg = weight_xavier_init(shape=kernalg, n_inputs=kernalg[0] * kernalg[1] * kernalg[2] * kernalg[3],
                                n_outputs=kernalg[-1], activefunction='relu', variable_name=scope + 'conv_Wg')
        Bg = bias_variable([kernalg[-1]], variable_name=scope + 'conv_Bg')
        phi_g = conv3d(g, Wg) + Bg

        add_xg = resnet_Add(theta_x, phi_g)
        act_xg = tf.nn.relu(add_xg)

        kernalpsi = (1, 1, 1, outfilters, 1)
        Wpsi = weight_xavier_init(shape=kernalpsi, n_inputs=kernalpsi[0] * kernalpsi[1] * kernalpsi[2] * kernalpsi[3],
                                  n_outputs=kernalpsi[-1], activefunction='relu', variable_name=scope + 'conv_Wpsi')
        Bpsi = bias_variable([kernalpsi[-1]], variable_name=scope + 'conv_Bpsi')
        psi = conv3d(act_xg, Wpsi) + Bpsi
        sigmoid_psi = tf.nn.sigmoid(psi)

        upsample_psi = upsample3d(sigmoid_psi, scale_factor=scale_factor, scope=scope + "resampler")

        # Attention: upsample_psi * x
        # upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
        #                              arguments={'repnum': outfilters})(upsample_psi)
        gat_x = tf.multiply(upsample_psi, x)
        kernal_gat_x = (1, 1, 1, outfilters, outfilters)
        Wgatx = weight_xavier_init(shape=kernal_gat_x,
                                   n_inputs=kernal_gat_x[0] * kernal_gat_x[1] * kernal_gat_x[2] * kernal_gat_x[3],
                                   n_outputs=kernal_gat_x[-1], activefunction='relu',
                                   variable_name=scope + 'conv_Wgatx')
        Bgatx = bias_variable([kernalpsi[-1]], variable_name=scope + 'conv_Bgatx')
        gat_x_out = conv3d(gat_x, Wgatx) + Bgatx
        gat_x_out = normalizationlayer(gat_x_out, is_train=phase, height=height, width=width, image_z=image_z,
                                       norm_type='group', scope=scope)
    return gat_x_out


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


def _createattentionnet(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=1):
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
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64), phase=phase, drop=drop,
                               scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernal=(3, 3, 3, 64, 128), phase=phase, drop=drop, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128), phase=phase, drop=drop,
                               scope='layer4_3')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernal=(3, 3, 3, 128, 256), phase=phase, drop=drop, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernal=(3, 3, 3, 256, 256), phase=phase, drop=drop,
                               scope='layer5_3')
    layer5 = resnet_Add(x1=down4, x2=layer5)

    # layer9->attngating
    g1 = gatingsignal3d(layer5, kernal=(1, 1, 1, 256, 128), phase=phase, scope='g1')
    attn1 = attngatingblock(layer4, g1, 128, 128, scale_factor=2, phase=phase, scope='attn1')
    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 128, 256), scope='deconv1')
    # layer8->convolution
    layer6 = crop_and_concat(attn1, deconv1)
    _, Z, H, W, _ = attn1.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 256, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_3')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer9->attngating
    g2 = gatingsignal3d(layer6, kernal=(1, 1, 1, 128, 64), phase=phase, scope='g2')
    attn2 = attngatingblock(layer3, g2, 64, 64, scale_factor=2, phase=phase, scope='attn2')
    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 64, 128), scope='deconv2')
    # layer8->convolution
    layer7 = crop_and_concat(attn2, deconv2)
    _, Z, H, W, _ = attn2.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 128, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_3')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->attngating
    g3 = gatingsignal3d(layer7, kernal=(1, 1, 1, 64, 32), phase=phase, scope='g3')
    attn3 = attngatingblock(layer2, g3, 32, 32, scale_factor=2, phase=phase, scope='attn3')
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 32, 64), scope='deconv3')
    # layer8->convolution
    layer8 = crop_and_concat(attn3, deconv3)
    _, Z, H, W, _ = attn3.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 64, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_3')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->attngating
    g4 = gatingsignal3d(layer8, kernal=(1, 1, 1, 32, 16), phase=phase, scope='g4')
    attn4 = attngatingblock(layer1, g4, 16, 16, scale_factor=2, phase=phase, scope='attn4')
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 16, 32), scope='deconv4')
    # layer8->convolution
    layer9 = crop_and_concat(attn4, deconv4)
    _, Z, H, W, _ = attn4.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_3')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_sigmod(x=layer9, kernal=(1, 1, 1, 32, n_class), scope='output')
    return output_map
