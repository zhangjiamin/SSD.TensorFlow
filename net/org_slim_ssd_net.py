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

class VGG16Backbone(object):
    def __init__(self, data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format
        if 'channels_first' == data_format:
            self.data_format = 'NCHW'
        elif 'channels_last' == data_format:
            self.data_format = 'NHWC'

    def arg_scope(self):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.glorot_uniform_initializer(),
                        biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=self.data_format) as sc:
                return sc

    def forward(self, inputs, training=False):
        with slim.arg_scope(self.arg_scope()):
            return self.__net(inputs, training=training)

    def __net(self, inputs, training=False):
        feats_bottom_up = {}
        feats_lateral   = {}
        feature_layers  = []
        net = inputs
        print(inputs.get_shape())

        # Block 1.
        net = slim.repeat(net, 2, slim.conv2d, 64, [3,3], scope='conv1')
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
            if self.data_format == 'NHWC':
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

        for level in range(2, 7 + 1):
            feature_layers.append(feats_bottom_up[level])

        return feature_layers

    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self.data_format == 'NHWC' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)

def arg_scope(data_format):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.glorot_uniform_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc

def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first'):
    if 'channels_first' == data_format:
        data_format_ = 'NCHW'
    elif 'channels_last' == data_format:
        data_format_ = 'NHWC'

    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            with slim.arg_scope(arg_scope(data_format_)):
                loc = slim.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, [3,3], activation_fn=None, scope='loc_{}'.format(ind))
                loc_preds.append(loc)
                cls = slim.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, [3,3], activation_fn=None, scope='cls_{}'.format(ind))
                cls_preds.append(cls)

        return loc_preds, cls_preds

