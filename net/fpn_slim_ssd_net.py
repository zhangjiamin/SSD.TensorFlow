# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4

def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None):
  """Performs a batch normalization followed by a ReLU.
  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    name: the name of the batch normalization layer
  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training_bn,
      fused=True,
      gamma_initializer=gamma_initializer,
      name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


class VGG16Backbone(object):
    def __init__(self, data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format

    def arg_scope(self, data_format='NCHW'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
                return sc

    def nearest_upsampling(self, data, scale):
        """Nearest neighbor upsampling implementation.
        Args:
            data: A float32 tensor of size [batch, height_in, width_in, channels].
            scale: An integer multiple to scale resolution of input data.
        Returns:
            data_up: A float32 tensor of size
            [batch, height_in*scale, width_in*scale, channels].
        """
        with tf.name_scope('nearest_upsampling'):
            bs, c, h, w = data.get_shape().as_list()
            bs = -1 if bs is None else bs
            # Use reshape to quickly upsample the input.  The nearest pixel is selected
            # implicitly via broadcasting.
            data = tf.reshape(data, [bs, c, h, 1, w, 1]) * tf.ones([1, 1, 1, scale, 1, scale], dtype=data.dtype)
            return tf.reshape(data, [bs, c, h * scale, w * scale])

    def forward(self, inputs, training=False):
        with slim.arg_scope(self.arg_scope()):
            return self.__net(inputs, training=training)

    def __net(self, inputs, training=False):
        feats_bottom_up = {}
        feats_lateral   = {}
        feature_layers  = []

        print(inputs.get_shape())

        # Block 1.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3,3], scope='conv1')
        net = slim.max_pool2d(net, [2,2], scope='pool1')
        print(net.get_shape())

        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3,3], scope='conv2')
        net = slim.max_pool2d(net, [2,2], scope='pool2')
        print(net.get_shape())

        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3,3], scope='conv3')
        net = slim.max_pool2d(net, [2,2], scope='pool3')
        print(net.get_shape())

        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], scope='conv4')
        print(net.get_shape())

        with tf.variable_scope('conv4_3_scale') as scope:
            weight_scale = tf.Variable([20.] * 512, trainable=training, name='weights')
            if self._data_format == 'channels_last':
                weight_scale = tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            else:
                weight_scale = tf.reshape(weight_scale, [1, -1, 1, 1], name='reshape')
        l2_norm_net = tf.multiply(weight_scale, self.l2_normalize(net, name='norm'), name='rescale')
        print(l2_norm_net.get_shape())
        feats_bottom_up[2] = l2_norm_net

        net = slim.max_pool2d(net, [2,2], scope='pool4')
        print(net.get_shape())

        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3,3], scope='conv5')
        net = slim.max_pool2d(net, [3,3], stride=1, scope='pool5')
        print(net.get_shape())

        # FC6
        net = slim.conv2d(net, 1024, [3,3], rate=6, scope='fc6')
        print(net.get_shape())

        # FC7
        net = slim.conv2d(net, 1024, [1,1], scope='fc7')
        print(net.get_shape())
        feats_bottom_up[3] = net

        # SSD layers
        with tf.variable_scope('additional_layers') as scope:
            with tf.variable_scope('conv8'):
                net = slim.conv2d(net, 256, [1,1], scope='conv8_1')
                net = slim.conv2d(net, 512, [3,3], stride=2, scope='conv8_2')
                print(net.get_shape())
                feats_bottom_up[4] = net

            with tf.variable_scope('conv9'):
                net = slim.conv2d(net, 128, [1,1], scope='conv9_1')
                net = slim.conv2d(net, 256, [3,3], stride=2, scope='conv9_2')
                print(net.get_shape())
                feats_bottom_up[5] = net

            with tf.variable_scope('conv10'):
                net = slim.conv2d(net, 128, [1,1], scope='conv10_1', padding='VALID')
                net = slim.conv2d(net, 256, [3,3], scope='conv10_2', padding='VALID')
                print(net.get_shape())
                feats_bottom_up[6] = net

            with tf.variable_scope('conv11'):
                net = slim.conv2d(net, 128, [1,1], scope='conv11_1', padding='VALID')
                net = slim.conv2d(net, 256, [3,3], scope='conv11_2', padding='VALID')
                print(net.get_shape())
                feats_bottom_up[7] = net

        min_layer = 2
        max_layer = 7

        print('features:')
        with tf.variable_scope('lateral_connections') as scope:
            for level in range(min_layer, max_layer + 1):
                print(feats_bottom_up[level].get_shape())
                channels = feats_bottom_up[level].get_shape().as_list()[1]
                channels = 256
                #feats_lateral[level] = slim.conv2d(feats_bottom_up[level], channels, [1,1], activation_fn=None, scope='lateral_{}'.format(level))
                feats_lateral[level] = slim.conv2d(feats_bottom_up[level], channels, [1,1], scope='lateral_{}'.format(level))

        print('upsampling:')
        feats = {max_layer: feats_lateral[max_layer]}
        for level in range(max_layer-1 , min_layer-1, -1):
            scale = 2
            if 6 == level:
                scale = 3
            upsampling = self.nearest_upsampling(feats[level + 1], scale)
            print(upsampling.get_shape())
            if 5 == level:
                upsampling = upsampling[:,:,0:5,0:5]
            if 3 == level:
                upsampling = upsampling[:,:,0:19,0:19]
            feats[level] = upsampling + feats_lateral[level]

        with tf.variable_scope('post_hocs') as scope:
            for level in range(min_layer, max_layer + 1):
                channels = feats[level].get_shape().as_list()[1]
                #feats[level] = slim.conv2d(feats[level], channels, [3,3], activation_fn=None, scope='post_hoc_{}'.format(level))
                feats[level] = slim.conv2d(feats[level], channels, [3,3], scope='post_hoc_{}'.format(level))

        for level in range(min_layer, max_layer + 1):
            feats[level] = tf.layers.batch_normalization(
                           inputs=feats[level],
                           axis=1, # NCHW data_format
                           momentum=0.997,
                           epsilon=0.0001,
                           center=True,
                           scale=True,
                           training=training,
                           fused=True,
                           name='p%d-bn' % level)
            feature_layers.append(feats[level])

        return feature_layers

    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)

def arg_scope(data_format='NCHW'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc

def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first'):
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            with slim.arg_scope(arg_scope()):
                net = slim.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, [3,3], activation_fn=None, scope='loc_{}'.format(ind))
                loc_preds.append(net)
                net = slim.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, [3,3], activation_fn=None, scope='cls_{}'.format(ind))
                cls_preds.append(net)

        return loc_preds, cls_preds

