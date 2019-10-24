'''

'''
from .layer import (conv3d, deconv3d, normalizationlayer, crop_and_concat, resnet_Add, weight_xavier_init,
                    bias_variable)
import tensorflow as tf


def positionAttentionblock(x, inputfilters, outfilters, kernal_size=1, scope=None):
    """
    Position attention module
    :param x:
    :param inputfilters:inputfilter number
    :param outfilters:outputfilter number
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        m_batchsize, Z, H, W, C = x.get_shape().as_list()

        kernalquery = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wquery = weight_xavier_init(shape=kernalquery,
                                    n_inputs=kernalquery[0] * kernalquery[1] * kernalquery[2] * kernalquery[3],
                                    n_outputs=kernalquery[-1], activefunction='relu',
                                    variable_name=scope + 'conv_Wquery')
        Bquery = bias_variable([kernalquery[-1]], variable_name=scope + 'conv_Bquery')
        query_conv = conv3d(x, Wquery) + Bquery
        query_conv_new = tf.reshape(query_conv, [-1, Z * H * W])

        kernalkey = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wkey = weight_xavier_init(shape=kernalkey, n_inputs=kernalkey[0] * kernalkey[1] * kernalkey[2] * kernalkey[3],
                                  n_outputs=kernalkey[-1], activefunction='relu', variable_name=scope + 'conv_Wkey')
        Bkey = bias_variable([kernalkey[-1]], variable_name=scope + 'conv_Bkey')
        key_conv = conv3d(x, Wkey) + Bkey
        key_conv_new = tf.reshape(key_conv, [-1, Z * H * W])

        # OOM,such as 512x512x32 then matric is 8388608x8388608
        # key_conv_new = tf.transpose(key_conv_new, [0, 2, 1])
        # (2,2,2,3)*(2,2,3,4)=(2,2,2,4),(2,2,3)*(2,3,4)=(2,2,4)
        # energy = tf.matmul(query_conv_new, key_conv_new)  # (m_batchsize,Z*H*W,Z*H*W)

        energy = tf.multiply(query_conv_new, key_conv_new)
        attention = tf.nn.sigmoid(energy)

        kernalproj = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wproj = weight_xavier_init(shape=kernalproj,
                                   n_inputs=kernalproj[0] * kernalproj[1] * kernalproj[2] * kernalproj[3],
                                   n_outputs=kernalproj[-1], activefunction='relu', variable_name=scope + 'conv_Wproj')
        Bproj = bias_variable([kernalproj[-1]], variable_name=scope + 'conv_Bproj')
        proj_value = conv3d(x, Wproj) + Bproj
        proj_value_new = tf.reshape(proj_value, [-1, Z * H * W])

        out = tf.multiply(attention, proj_value_new)
        out_new = tf.reshape(out, [-1, Z, H, W, C])

        out_new = resnet_Add(out_new, x)
        return out_new


def channelAttentionblock(x, scope=None):
    """
    Channel attention module
    :param x:input
    :param scope: scope name
    :return:channelattention result
    """
    with tf.name_scope(scope):
        m_batchsize, Z, H, W, C = x.get_shape().as_list()

        proj_query = tf.reshape(x, [-1, C])
        proj_key = tf.reshape(x, [-1, C])
        proj_query = tf.transpose(proj_query, [1, 0])

        energy = tf.matmul(proj_query, proj_key)  # (C,C)
        attention = tf.nn.sigmoid(energy)

        proj_value = tf.reshape(x, [-1, C])
        proj_value = tf.transpose(proj_value, [1, 0])
        out = tf.matmul(attention, proj_value)  # (C,-1)

        out = tf.reshape(out, [-1, Z, H, W, C])
        out = resnet_Add(out, x)
        return out


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


def _createdualattentionnet(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=1):
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
    # layer9->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 128, 256), scope='deconv1')
    # dual model1
    pos_attenfeat1 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128 // 2), phase=phase, drop=drop,
                                       scope='dual_layer1_1')
    pos_attenfeat1 = positionAttentionblock(pos_attenfeat1, 128 // 2, 128 // 2, scope='dual_pos_atten1')
    pos_attenfeat1 = conv_bn_relu_drop(x=pos_attenfeat1, kernal=(3, 3, 3, 128 // 2, 128 // 2), phase=phase, drop=drop,
                                       scope='dual_layer1_2')

    cha_attenfeat1 = conv_bn_relu_drop(x=layer4, kernal=(3, 3, 3, 128, 128 // 2), phase=phase, drop=drop,
                                       scope='dual_layer1_3')
    cha_attenfeat1 = channelAttentionblock(cha_attenfeat1, scope='dual_cha_atten1')
    cha_attenfeat1 = conv_bn_relu_drop(x=cha_attenfeat1, kernal=(3, 3, 3, 128 // 2, 128 // 2), phase=phase, drop=drop,
                                       scope='dual_layer1_4')

    feat_sum1 = resnet_Add(pos_attenfeat1, cha_attenfeat1)
    sasc_output1 = conv_bn_relu_drop(x=feat_sum1, kernal=(1, 1, 1, 128 // 2, 128), phase=phase, drop=drop,
                                     scope='dual_layer1_5')
    # layer8->convolution
    layer6 = crop_and_concat(sasc_output1, deconv1)
    _, Z, H, W, _ = sasc_output1.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 256, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 128, 128), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_3')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)

    # layer9->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 64, 128), scope='deconv2')
    # dual model2
    pos_attenfeat2 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64 // 2), phase=phase, drop=drop,
                                       scope='dual_layer2_1')
    pos_attenfeat2 = positionAttentionblock(pos_attenfeat2, 64 // 2, 64 // 2, scope='dual_pos_atten2')
    pos_attenfeat2 = conv_bn_relu_drop(x=pos_attenfeat2, kernal=(3, 3, 3, 64 // 2, 64 // 2), phase=phase, drop=drop,
                                       scope='dual_layer2_2')
    cha_attenfeat2 = conv_bn_relu_drop(x=layer3, kernal=(3, 3, 3, 64, 64 // 2), phase=phase, drop=drop,
                                       scope='dual_layer2_3')
    cha_attenfeat2 = channelAttentionblock(cha_attenfeat2, scope='dual_cha_atten2')
    cha_attenfeat2 = conv_bn_relu_drop(x=cha_attenfeat2, kernal=(3, 3, 3, 64 // 2, 64 // 2), phase=phase, drop=drop,
                                       scope='dual_layer2_4')
    feat_sum2 = resnet_Add(pos_attenfeat2, cha_attenfeat2)
    sasc_output2 = conv_bn_relu_drop(x=feat_sum2, kernal=(1, 1, 1, 64 // 2, 64), phase=phase, drop=drop,
                                     scope='dual_layer2_5')
    # layer8->convolution
    layer7 = crop_and_concat(sasc_output2, deconv2)
    _, Z, H, W, _ = sasc_output2.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 128, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 64, 64), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_3')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer9->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 32, 64), scope='deconv3')
    # dual model3
    pos_attenfeat3 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 32, 32 // 2), phase=phase, drop=drop,
                                       scope='dual_layer3_1')
    pos_attenfeat3 = positionAttentionblock(pos_attenfeat3, 32 // 2, 32 // 2, scope='dual_pos_atten3')
    pos_attenfeat3 = conv_bn_relu_drop(x=pos_attenfeat3, kernal=(3, 3, 3, 32 // 2, 32 // 2), phase=phase, drop=drop,
                                       scope='dual_layer3_2')
    cha_attenfeat3 = conv_bn_relu_drop(x=layer2, kernal=(3, 3, 3, 32, 32 // 2), phase=phase, drop=drop,
                                       scope='dual_layer3_3')
    cha_attenfeat3 = channelAttentionblock(cha_attenfeat3, scope='dual_cha_atten3')
    cha_attenfeat3 = conv_bn_relu_drop(x=cha_attenfeat3, kernal=(3, 3, 3, 32 // 2, 32 // 2), phase=phase, drop=drop,
                                       scope='dual_layer3_4')
    feat_sum3 = resnet_Add(pos_attenfeat3, cha_attenfeat3)
    sasc_output3 = conv_bn_relu_drop(x=feat_sum3, kernal=(1, 1, 1, 32 // 2, 32), phase=phase, drop=drop,
                                     scope='dual_layer3_5')
    # layer8->convolution
    layer8 = crop_and_concat(sasc_output3, deconv3)
    _, Z, H, W, _ = sasc_output3.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 64, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 32, 32), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_3')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer9->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 16, 32), scope='deconv4')
    # dual model4
    pos_attenfeat4 = conv_bn_relu_drop(x=layer1, kernal=(3, 3, 3, 16, 16 // 2), phase=phase, drop=drop,
                                       scope='dual_layer4_1')
    pos_attenfeat4 = positionAttentionblock(pos_attenfeat4, 16 // 2, 16 // 2, scope='dual_pos_atten4')
    pos_attenfeat4 = conv_bn_relu_drop(x=pos_attenfeat4, kernal=(3, 3, 3, 16 // 2, 16 // 2), phase=phase, drop=drop,
                                       scope='dual_layer4_2')
    cha_attenfeat4 = conv_bn_relu_drop(x=layer1, kernal=(3, 3, 3, 16, 16 // 2), phase=phase, drop=drop,
                                       scope='dual_layer4_3')
    cha_attenfeat4 = channelAttentionblock(cha_attenfeat4, scope='dual_cha_atten4')
    cha_attenfeat4 = conv_bn_relu_drop(x=cha_attenfeat4, kernal=(3, 3, 3, 16 // 2, 16 // 2), phase=phase, drop=drop,
                                       scope='dual_layer4_4')
    feat_sum4 = resnet_Add(pos_attenfeat4, cha_attenfeat4)
    sasc_output4 = conv_bn_relu_drop(x=feat_sum4, kernal=(1, 1, 1, 16 // 2, 16), phase=phase, drop=drop,
                                     scope='dual_layer4_5')
    # layer8->convolution
    layer9 = crop_and_concat(sasc_output4, deconv4)
    _, Z, H, W, _ = sasc_output4.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 32, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 16, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 16, 16), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_3')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->output
    output_map = conv_sigmod(x=layer9, kernal=(1, 1, 1, 16, n_class), scope='output')
    return output_map
