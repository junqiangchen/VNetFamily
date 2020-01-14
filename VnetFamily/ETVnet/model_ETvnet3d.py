'''

'''
from .layer import (conv3d, deconv3d, upsample3d, normalizationlayer, crop_and_concat, resnet_Add,
                        weight_xavier_init, bias_variable, save_images)



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


def deconv_relu(x, kernal, samefeture=False, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.relu(conv)
        return conv



def conv_active(x, kernal, active=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        if active == 'relu':
            conv = tf.nn.softmax(conv)
        if active == 'sigmod':
            conv = tf.nn.sigmoid(conv)
        return conv


def Spatial_squeeze_Channel_excitation_layer(x, out_dim, ratio=4, scope=None):
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


def weighted_aggregation_model(x1, x2, x3, x4, Channel, scope=None):
    """
    weighted_aggregation_model
    :param x1:
    :param x2:
    :param x3:
    :param x4:
    :param Channel:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        wb1 = Spatial_squeeze_Channel_excitation_layer(x1, Channel, scope=scope + 'wb1')
        wb1 = conv_active(wb1, kernal=(1, 1, 1, Channel, Channel // 2), active='relu', scope=scope + 'layer1')
        wb1 = upsample3d(wb1, 2, scope=scope + 'up1')
        wb2 = Spatial_squeeze_Channel_excitation_layer(x2, Channel // 2, scope=scope + 'wb2')
        wb2 = resnet_Add(wb1, wb2)
        wb2 = conv_active(wb2, kernal=(1, 1, 1, Channel // 2, Channel // 4), active='relu', scope=scope + 'layer2')
        wb2 = upsample3d(wb2, 2, scope=scope + 'up2')
        wb3 = Spatial_squeeze_Channel_excitation_layer(x3, Channel // 4, scope=scope + 'wb3')
        wb3 = resnet_Add(wb3, wb2)
        wb3 = conv_active(wb3, kernal=(1, 1, 1, Channel // 4, Channel // 8), active='relu', scope=scope + 'layer3')
        wb3 = upsample3d(wb3, 2, scope=scope + 'up3')
        wb4 = Spatial_squeeze_Channel_excitation_layer(x4, Channel // 8, scope=scope + 'wb4')
        wb4 = resnet_Add(wb3, wb4)
        return wb4


def edge_guidance_model(x1, x2, scope=None):
    """
    edge_guidance_model
    :param x1:
    :param x2:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        C1 = x1.get_shape().as_list()[4]
        layer1 = conv_active(x1, kernal=(1, 1, 1, C1, C1), active='relu', scope=scope + 'layer1_0')
        layer1 = conv_active(layer1, kernal=(3, 3, 3, C1, C1), active='relu', scope=scope + 'layer1_1')

        C2 = x2.get_shape().as_list()[4]
        layer2 = upsample3d(x2, scale_factor=2, scope=scope + 'up1')
        layer2 = conv_active(layer2, kernal=(1, 1, 1, C2, C2), active='relu', scope=scope + 'layer2_0')
        layer2 = conv_active(layer2, kernal=(3, 3, 3, C2, C2), active='relu', scope=scope + 'layer2_1')

        egm = crop_and_concat(layer1, layer2)

        C = C1 + C2
        egm = conv_active(egm, kernal=(1, 1, 1, C, C), scope=scope + 'layer3')
        return egm


def _create_etconv_net(X, image_z, image_width, image_height, image_channel, phase, drop, n_class=2):
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

    # layer6->deconvolution
    deconv1 = deconv_relu(x=layer5, kernal=(3, 3, 3, 160, 320), scope='deconv1')
    # layer7->convolution
    layer6 = crop_and_concat(layer4, deconv1)
    _, Z, H, W, _ = layer4.get_shape().as_list()
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 320, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_2')
    layer6 = conv_bn_relu_drop(x=layer6, kernal=(3, 3, 3, 160, 160), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer6_3')
    layer6 = resnet_Add(x1=deconv1, x2=layer6)
    # layer8->deconvolution
    deconv2 = deconv_relu(x=layer6, kernal=(3, 3, 3, 80, 160), scope='deconv2')
    # layer9->convolution
    layer7 = crop_and_concat(layer3, deconv2)
    _, Z, H, W, _ = layer3.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 160, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_2')
    layer7 = conv_bn_relu_drop(x=layer7, kernal=(3, 3, 3, 80, 80), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer7_3')
    layer7 = resnet_Add(x1=deconv2, x2=layer7)
    # layer10->deconvolution
    deconv3 = deconv_relu(x=layer7, kernal=(3, 3, 3, 40, 80), scope='deconv3')
    # layer11->convolution
    layer8 = crop_and_concat(layer2, deconv3)
    _, Z, H, W, _ = layer2.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 80, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_2')
    layer8 = conv_bn_relu_drop(x=layer8, kernal=(3, 3, 3, 40, 40), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer8_3')
    layer8 = resnet_Add(x1=deconv3, x2=layer8)
    # layer12->deconvolution
    deconv4 = deconv_relu(x=layer8, kernal=(3, 3, 3, 20, 40), scope='deconv4')
    # layer13->convolution
    layer9 = crop_and_concat(layer1, deconv4)
    _, Z, H, W, _ = layer1.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 40, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_2')
    layer9 = conv_bn_relu_drop(x=layer9, kernal=(3, 3, 3, 20, 20), image_z=Z, height=H, width=W, phase=phase,
                               drop=drop, scope='layer9_3')
    layer9 = resnet_Add(x1=deconv4, x2=layer9)
    # layer14->edge_guidance_model
    egm_output = edge_guidance_model(x1=layer1, x2=layer2, scope='edm')
    _, _, _, _, egm_output_C = egm_output.get_shape().as_list()
    # layer15->weighted_aggregation_model
    wam_output = weighted_aggregation_model(x1=layer6, x2=layer7, x3=layer8, x4=layer9, Channel=160, scope='wam')
    # # layer16->output
    fusion_output = crop_and_concat(x1=egm_output, x2=wam_output)
    output_map = conv_active(x=fusion_output, kernal=(1, 1, 1, 20 + egm_output_C, n_class),active='sigmod', scope='output')
    return output_map