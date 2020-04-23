'''
covlution layer，pool layer，initialization。。。。
'''
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, activefunction='sigomd', uniform=True, variable_name=None):
    if activefunction == 'sigomd':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs))
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
    elif activefunction == 'relu':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * np.sqrt(2)
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
    elif activefunction == 'tan':
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * 4
            initial = tf.random_uniform(shape, -init_range, init_range)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * 4
            initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# Bias initialization
def bias_variable(shape, variable_name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)


# 3D convolution
def conv3d(x, W, stride=1, dilation=1):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME',
                           dilations=[1, dilation, dilation, dilation, 1])
    return conv_3d


# 3D upsampling
def upsample3d(x, scale_factor, scope=None):
    ''''
    X shape is [nsample,dim,rows, cols, channel]
    out shape is[nsample,dim*scale_factor,rows*scale_factor, cols*scale_factor, channel]
    '''
    x_shape = tf.shape(x)
    k = tf.ones([scale_factor, scale_factor, scale_factor, x_shape[-1] // scale_factor, x_shape[-1]])
    # note k.shape = [dim,rows, cols, depth_in, depth_output]
    output_shape = tf.stack(
        [x_shape[0], x_shape[1] * scale_factor, x_shape[2] * scale_factor, x_shape[3] * scale_factor,
         x_shape[4] // scale_factor])
    upsample = tf.nn.conv3d_transpose(value=x, filter=k, output_shape=output_shape,
                                      strides=[1, scale_factor, scale_factor, scale_factor, 1],
                                      padding='SAME', name=scope)
    return upsample


# 3D deconvolution
def deconv3d(x, W, samefeature=False, depth=False):
    """
    depth flag:False is z axis is same between input and output,true is z axis is input is twice than output
    """
    x_shape = tf.shape(x)
    if depth:
        if samefeature:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4]])
        else:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
        deconv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
    else:
        if samefeature:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4]])
        else:
            output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4] // 2])
        deconv = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, 2, 2, 1, 1], padding='SAME')
    return deconv


# Max Pooling
def max_pool3d(x, depth=False):
    """
        depth flag:False is z axis is same between input and output,true is z axis is input is twice than output
        """
    if depth:
        pool3d = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    else:
        pool3d = tf.nn.max_pool3d(x, ksize=[1, 2, 2, 1, 1], strides=[1, 2, 2, 1, 1], padding='SAME')
    return pool3d


# Unet crop and concat
def crop_and_concat(x1, x2):
    """
    concat x1 and x2
    :param x1:
    :param x2:
    :return:
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)


# Batch Normalization
def normalizationlayer(x, is_train, height=None, width=None, image_z=None, norm_type=None, G=16, esp=1e-5, scope=None):
    """
    normalizationlayer
    :param x:input data with shap of[batch,height,width,channel]
    :param is_train:flag of normalizationlayer,True is training,False is Testing
    :param height:in some condition,the data height is in Runtime determined,such as through deconv layer and conv2d
    :param width:in some condition,the data width is in Runtime determined
    :param image_z:
    :param norm_type:normalization type:support"batch","group","None"
    :param G:in group normalization,channel is seperated with group number(G)
    :param esp:Prevent divisor from being zero
    :param scope:normalizationlayer scope
    :return:
    """
    with tf.name_scope(scope + norm_type):
        if norm_type == None:
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_train=is_train)
        elif norm_type == "group":
            # tranpose:[bs,z,h,w,c]to[bs,c,z,h,w]following the paper
            x = tf.transpose(x, [0, 4, 1, 2, 3])
            N, C, Z, H, W = x.get_shape().as_list()
            G = min(G, C)
            if H == None and W == None and Z == None:
                Z, H, W = image_z, height, width
            x = tf.reshape(x, [-1, G, C // G, Z, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4, 5], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            gama = tf.get_variable(scope + norm_type + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(scope + norm_type + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
            gama = tf.reshape(gama, [1, C, 1, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1, 1])
            output = tf.reshape(x, [-1, C, Z, H, W]) * gama + beta
            # tranpose:[bs,c,z,h,w]to[bs,z,h,w,c]following the paper
            output = tf.transpose(output, [0, 2, 3, 4, 1])
        return output


# resnet add_connect
def resnet_Add(x1, x2):
    """
    add x1 and x2
    :param x1:
    :param x2:
    :return:
    """
    residual_connection = tf.add(x1, x2)
    return residual_connection


def save_images(images, size, path, pixelvalue=255.):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image
    result = merge_img * pixelvalue
    result = np.clip(result, 0, 255.).astype('uint8')
    return cv2.imwrite(path, result)


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
        query_conv_new = tf.reshape(query_conv, [-1, Z * H * W, C])

        kernalkey = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wkey = weight_xavier_init(shape=kernalkey, n_inputs=kernalkey[0] * kernalkey[1] * kernalkey[2] * kernalkey[3],
                                  n_outputs=kernalkey[-1], activefunction='relu', variable_name=scope + 'conv_Wkey')
        Bkey = bias_variable([kernalkey[-1]], variable_name=scope + 'conv_Bkey')
        key_conv = conv3d(x, Wkey) + Bkey
        key_conv_new = tf.reshape(key_conv, [-1, Z * H * W, C])

        # OOM,such as 512x512x32 then matric is 8388608x8388608
        key_conv_new = tf.transpose(key_conv_new, [0, 2, 1])
        # (2,2,2,3)*(2,2,3,4)=(2,2,2,4),(2,2,3)*(2,3,4)=(2,2,4)
        energy = tf.matmul(query_conv_new, key_conv_new)  # (m_batchsize,Z*H*W,Z*H*W)
        attention = tf.nn.softmax(energy, -1)

        kernalproj = (kernal_size, kernal_size, kernal_size, inputfilters, outfilters)
        Wproj = weight_xavier_init(shape=kernalproj,
                                   n_inputs=kernalproj[0] * kernalproj[1] * kernalproj[2] * kernalproj[3],
                                   n_outputs=kernalproj[-1], activefunction='relu', variable_name=scope + 'conv_Wproj')
        Bproj = bias_variable([kernalproj[-1]], variable_name=scope + 'conv_Bproj')
        proj_value = conv3d(x, Wproj) + Bproj
        proj_value_new = tf.reshape(proj_value, [-1, Z * H * W, C])

        out = tf.matmul(attention, proj_value_new)  # (m_batchsize,Z*H*W,C)
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

        proj_query = tf.reshape(x, [-1, Z * H * W, C])
        proj_key = tf.reshape(x, [-1, Z * H * W, C])
        proj_query = tf.transpose(proj_query, [0, 2, 1])

        energy = tf.matmul(proj_query, proj_key)  # (-1,C,C)
        attention = tf.nn.softmax(energy, -1)  # (-1,C,C)

        proj_value = tf.reshape(x, [-1, Z * H * W, C])
        proj_value = tf.transpose(proj_value, [0, 2, 1])
        out = tf.matmul(attention, proj_value)  # (-1,C,Z*H*W)

        out = tf.reshape(out, [-1, Z, H, W, C])
        out = resnet_Add(out, x)
        return out


def NonLocalBlock(input_x, phase, image_z=None, image_height=None, image_width=None, scope=None):
    """
    Non-local netural network
    :param input_x:
    :param out_channels:
    :param scope:
    :return:
    """
    batchsize, dimensizon, height, width, out_channels = input_x.get_shape().as_list()
    with tf.name_scope(scope):
        kernal_thela = (1, 1, 1, out_channels, out_channels // 2)
        W_thela = weight_xavier_init(shape=kernal_thela,
                                     n_inputs=kernal_thela[0] * kernal_thela[1] * kernal_thela[2] * kernal_thela[3],
                                     n_outputs=kernal_thela[-1], activefunction='relu',
                                     variable_name=scope + 'conv_W_thela')
        B_thela = bias_variable([kernal_thela[-1]], variable_name=scope + 'conv_B_thela')
        thela = conv3d(input_x, W_thela) + B_thela
        thela = normalizationlayer(thela, is_train=phase, height=image_height, width=image_width, image_z=image_z,
                                   norm_type='group', scope=scope + "NonLocalbn1")

        kernal_phi = (1, 1, 1, out_channels, out_channels // 2)
        W_phi = weight_xavier_init(shape=kernal_phi,
                                   n_inputs=kernal_phi[0] * kernal_phi[1] * kernal_phi[2] * kernal_phi[3],
                                   n_outputs=kernal_phi[-1], activefunction='relu',
                                   variable_name=scope + 'conv_W_phi')
        B_phi = bias_variable([kernal_phi[-1]], variable_name=scope + 'conv_B_phi')
        phi = conv3d(input_x, W_phi) + B_phi
        phi = normalizationlayer(phi, is_train=phase, height=image_height, width=image_width, image_z=image_z,
                                 norm_type='group', scope=scope + "NonLocalbn2")

        kernal_g = (1, 1, 1, out_channels, out_channels // 2)
        W_g = weight_xavier_init(shape=kernal_g,
                                 n_inputs=kernal_g[0] * kernal_g[1] * kernal_g[2] * kernal_g[3],
                                 n_outputs=kernal_g[-1], activefunction='relu',
                                 variable_name=scope + 'conv_W_g')
        B_g = bias_variable([kernal_g[-1]], variable_name=scope + 'conv_B_g')
        g = conv3d(input_x, W_g) + B_g
        g = normalizationlayer(g, is_train=phase, height=image_height, width=image_width, image_z=image_z,
                               norm_type='group', scope=scope + "NonLocalbn3")

        g_x = tf.reshape(g, [-1, dimensizon * height * width, out_channels // 2])
        theta_x = tf.reshape(thela, [-1, dimensizon * height * width, out_channels // 2])
        phi_x = tf.reshape(phi, [-1, dimensizon * height * width, out_channels // 2])
        phi_x = tf.transpose(phi_x, [0, 2, 1])

        f = tf.matmul(theta_x, phi_x)

        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        y = tf.reshape(y, [-1, dimensizon, height, width, out_channels // 2])

        kernal_y = (1, 1, 1, out_channels // 2, out_channels)
        W_y = weight_xavier_init(shape=kernal_y,
                                 n_inputs=kernal_y[0] * kernal_y[1] * kernal_y[2] * kernal_y[3],
                                 n_outputs=kernal_y[-1], activefunction='relu',
                                 variable_name=scope + 'conv_W_y')
        B_y = bias_variable([kernal_y[-1]], variable_name=scope + 'conv_B_y')
        w_y = conv3d(y, W_y) + B_y
        w_y = normalizationlayer(w_y, is_train=phase, height=image_height, width=image_width, image_z=image_z,
                                 norm_type='group', scope=scope + "NonLocalbn4")
        z = resnet_Add(input_x, w_y)
        return z


def conv_bn_relu_drop(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    """
    conv+bn+relu+drop
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
        conv = tf.nn.dropout(tf.nn.leaky_relu(conv), drop)
        return conv


def down_sampling(x, kernal, phase, drop, image_z=None, height=None, width=None, scope=None):
    """
    downsampling with conv stride=2
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
                               n_outputs=kernal[-1],
                               activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W, 2) + B
        conv = normalizationlayer(conv, is_train=phase, height=height, width=width, image_z=image_z, norm_type='group',
                                  scope=scope)
        conv = tf.nn.dropout(tf.nn.leaky_relu(conv), drop)
        return conv


def deconv_relu(x, kernal, samefeture=False, scope=None):
    """
    deconv+relu
    :param x:
    :param kernal:
    :param samefeture:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[-1],
                               n_outputs=kernal[-2], activefunction='relu', variable_name=scope + 'W')
        B = bias_variable([kernal[-2]], variable_name=scope + 'B')
        conv = deconv3d(x, W, samefeture, True) + B
        conv = tf.nn.leaky_relu(conv)
        return conv


def conv_sigmod(x, kernal, scope=None, activeflag=True):
    """
    conv_sigmod
    :param x:
    :param kernal:
    :param scope:
    :return:
    """
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernal, n_inputs=kernal[0] * kernal[1] * kernal[2] * kernal[3],
                               n_outputs=kernal[-1], activefunction='sigomd', variable_name=scope + 'W')
        B = bias_variable([kernal[-1]], variable_name=scope + 'B')
        conv = conv3d(x, W) + B
        if activeflag:
            conv = tf.nn.sigmoid(conv)
        return conv
