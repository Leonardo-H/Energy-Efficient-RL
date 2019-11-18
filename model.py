import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy

def BatchNorm(x, is_training=None):
    assert is_training is not None
    return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True,
        updates_collections=None, is_training=is_training, trainable=True, fused=True)


def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    ret = x + tf.stop_gradient(tf.sign(x) - x)
    return ret

def BinarizedAffine(x, nOutputPlane, name=None, reuse=False, is_training=True, use_sign=False):

    with tf.variable_scope(name,'Affine',[x], reuse=reuse):
        '''
        Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
        we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
        '''
        if use_sign:
            print('use sign in Affine')
            bin_x = tf.sign(x)
        else:
            bin_x = binarize(x)        
        nInputPlane = bin_x.get_shape().as_list()[1]

        w = tf.get_variable('weight', [nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer())        
        if use_sign:
            print('use sign in Affine')
            bin_w = tf.sign(w)
        else:
            bin_w = binarize(w)

        output = tf.matmul(bin_x, bin_w)
    return output

def BinarizedSpatialConvolution(x, nOutputPlane, kW, kH, dW=1, dH=1,
        padding='SAME', reuse=False, name='BinarizedSpatialConvolution', is_training=True, use_sign=False):
    nInputPlane = x.get_shape().as_list()[3]
    
    with tf.variable_scope(name, None,[x], reuse=reuse):
        w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        if use_sign:
            print('use sign in Conv')
            bin_w = tf.sign(w)
            bin_x = tf.sign(x)
        else:
            bin_w = binarize(w)
            bin_x = binarize(x)
        '''
        Note that we use binarized version of the input and the weights. Since the binarized function uses STE
        we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
        '''
        out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
    return out

def BinarizedWeightOnlySpatialConvolution(x, nOutputPlane, kW, kH, dW=1, dH=1, is_training=True,
        padding='SAME', reuse=False, name='BinarizedWeightOnlySpatialConvolution', use_sign=False):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    
    nInputPlane = x.get_shape().as_list()[3]
    with tf.variable_scope(name, None, [x], reuse=reuse):
        w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        if use_sign:
            print('use sign in first conv')
            bin_w = tf.sign(w)
        else:                   
            bin_w = binarize(w)
        out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
    return out


def HardTanh(x):
    return tf.clip_by_value(x, -1, 1)

def v_func_model(img_in, scope, reuse=False, batch_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=1, activation_fn=None)
        return out


# binary model used for bootstrapping algorithm
def binary_model_1(img_in, num_actions, scope, is_training, reuse=False, use_sign=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            out = BinarizedWeightOnlySpatialConvolution(
                            x=out,
                            nOutputPlane=32,
                            kW=8,
                            kH=8,
                            dW=4,
                            dH=4,
                            name='BWOSC_0',
                            use_sign=use_sign,
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=4,
                            kH=4,
                            dW=2,
                            dH=2,
                            name='BSC_0',
                            use_sign=use_sign,
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=3,
                            kH=3,
                            dW=1,
                            dH=1,
                            name='BSC_1',
                            use_sign=use_sign,
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)

        out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            actions_hidden = BinarizedAffine(out, nOutputPlane=512, use_sign=use_sign)
            actions_hidden = BatchNorm(actions_hidden, is_training)
            actions_hidden = HardTanh(actions_hidden) 

            action_scores = BinarizedAffine(actions_hidden, nOutputPlane=num_actions, use_sign=use_sign)
            action_scores = BatchNorm(action_scores, is_training)

            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)
        return action_scores

# binary model used for imitation learning algorithm
def binary_model_2(img_in, num_actions, scope, is_training, reuse=False, use_sign=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            out = BinarizedWeightOnlySpatialConvolution(
                            x=out,
                            nOutputPlane=32,
                            kW=8,
                            kH=8,
                            dW=4,
                            dH=4,
                            name='BWOSC_0',
                            use_sign=use_sign,
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=4,
                            kH=4,
                            dW=2,
                            dH=2,
                            name='BSC_0',
                            use_sign=use_sign,
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=3,
                            kH=3,
                            dW=1,
                            dH=1,
                            name='BSC_1',
                            use_sign=use_sign,
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)

        out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            actions_hidden = BinarizedAffine(out, nOutputPlane=512, use_sign=use_sign)
            actions_hidden = BatchNorm(actions_hidden, is_training)
            actions_hidden = HardTanh(actions_hidden) 

            action_scores = BinarizedAffine(actions_hidden, nOutputPlane=num_actions, use_sign=use_sign)
            action_scores = BatchNorm(action_scores, is_training)
            
        return action_scores



# binary model used for bootstrapping algorithm
def binary_model_3(img_in, num_actions, scope, is_training, reuse=False, ):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            out = BinarizedWeightOnlySpatialConvolution(
                            x=out,
                            nOutputPlane=32,
                            kW=8,
                            kH=8,
                            dW=4,
                            dH=4,
                            name='BWOSC_0',
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=4,
                            kH=4,
                            dW=2,
                            dH=2,
                            name='BSC_0'
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=3,
                            kH=3,
                            dW=1,
                            dH=1,
                            name='BSC_1'
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)

        out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            actions_hidden = BinarizedAffine(out, nOutputPlane=512)
            actions_hidden = BatchNorm(actions_hidden, is_training)
            actions_hidden = HardTanh(actions_hidden) 

            action_scores = BinarizedAffine(actions_hidden, nOutputPlane=num_actions)
            action_scores = BatchNorm(action_scores, is_training)

            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)
        return action_scores

# binary model used for imitation learning algorithm
def binary_model_4(img_in, num_actions, scope, is_training, reuse=False,):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            out = BinarizedWeightOnlySpatialConvolution(
                            x=out,
                            nOutputPlane=32,
                            kW=8,
                            kH=8,
                            dW=4,
                            dH=4,
                            name='BWOSC_0',
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=4,
                            kH=4,
                            dW=2,
                            dH=2,
                            name='BSC_0'
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)
            out = BinarizedSpatialConvolution(
                            x=out,
                            nOutputPlane=64,
                            kW=3,
                            kH=3,
                            dW=1,
                            dH=1,
                            name='BSC_1'
                        )
            out = BatchNorm(out, is_training)
            out = HardTanh(out)

        out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            actions_hidden = BinarizedAffine(out, nOutputPlane=512)
            actions_hidden = BatchNorm(actions_hidden, is_training)
            actions_hidden = HardTanh(actions_hidden) 

            action_scores = BinarizedAffine(actions_hidden, nOutputPlane=num_actions)
            action_scores = BatchNorm(action_scores, is_training)
            
        return action_scores
