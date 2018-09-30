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

    def basic_RFB(self, endpoint, inputs, out_planes, stride=1, scale=0.1, visual=1, is_training=True):
        inter_planes = out_planes // 8
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay':0.99, 'epsilon':0.00001, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused':None}):
            with tf.variable_scope(endpoint) as scope:
                with tf.variable_scope('Branch0') as scope:
                    branch0 = slim.conv2d(inputs, 2*inter_planes, [1,1], stride=stride)
                    branch0 = slim.conv2d(branch0, 2*inter_planes, [3,3], rate=visual, activation_fn=None)
                with tf.variable_scope('Branch1') as scope:
                    branch1 = slim.conv2d(inputs, inter_planes, [1,1])
                    branch1 = slim.conv2d(branch1, 2*inter_planes, [3,3], stride=stride)
                    branch1 = slim.conv2d(branch1, 2*inter_planes, [3,3], rate=visual+1, activation_fn=None)
                with tf.variable_scope('Branch2') as scope:
                    branch2 = slim.conv2d(inputs, inter_planes, [1,1])
                    branch2 = slim.conv2d(branch2, (inter_planes//2)*3, [3,3])
                    branch2 = slim.conv2d(branch2, 2*inter_planes, [3,3], stride=stride)
                    branch2 = slim.conv2d(branch2, 2*inter_planes, [3,3], rate=2*visual+1, activation_fn=None)
                net = tf.concat(axis=1, values=[branch0, branch1, branch2])
                net = slim.conv2d(net, out_planes, [1,1], activation_fn=None, scope='conv-linear')
                shortcut = slim.conv2d(inputs, out_planes, [1,1], stride=stride, activation_fn=None, scope='shortcut')
                net = net*scale + shortcut
                net = tf.nn.relu(net)
                return net 

    def basic_RFB_a(self, endpoint, inputs, out_planes, stride=1, scale=0.1, is_training=True):
        inter_planes = out_planes // 4
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay':0.99, 'epsilon':0.00001, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused':None}):
            with tf.variable_scope(endpoint) as scope:
                with tf.variable_scope('Branch0') as scope:
                    branch0 = slim.conv2d(inputs, inter_planes, [1,1], stride=stride)
                    branch0 = slim.conv2d(branch0, inter_planes, [3,3], activation_fn=None)
                with tf.variable_scope('Branch1') as scope:
                    branch1 = slim.conv2d(inputs, inter_planes, [1,1])
                    branch1 = slim.conv2d(branch1, inter_planes, [3,1], stride=stride)
                    branch1 = slim.conv2d(branch1, inter_planes, [3,3], rate=3, activation_fn=None)
                with tf.variable_scope('Branch2') as scope:
                    branch2 = slim.conv2d(inputs, inter_planes, [1,1])
                    branch2 = slim.conv2d(branch2, inter_planes, [1,3], stride=stride)
                    branch2 = slim.conv2d(branch2, inter_planes, [3,3], rate=3, activation_fn=None)
                with tf.variable_scope('Branch3') as scope:
                    branch3 = slim.conv2d(inputs, inter_planes//2, [1,1])
                    branch3 = slim.conv2d(branch3, (inter_planes//4)*3, [1,3])
                    branch3 = slim.conv2d(branch3, inter_planes, [3,1], stride=stride)
                    branch3 = slim.conv2d(branch3, inter_planes, [3,3], rate=5, activation_fn=None)
                net = tf.concat(axis=1, values=[branch0, branch1, branch2, branch3])
                net = slim.conv2d(net, out_planes, [1,1], activation_fn=None, scope='conv-linear')
                shortcut = slim.conv2d(inputs, out_planes, [1,1], stride=stride, activation_fn=None, scope='shortcut')
                net = net*scale + shortcut
                net = tf.nn.relu(net)
                return net 

    def arg_scope(self, data_format='NCHW'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
                return sc

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

        norm_net = self.basic_RFB_a("normal", net, 512, stride=1, scale=1.0, is_training=training)
        print(norm_net.get_shape())
        feats_bottom_up[2] = norm_net

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

        # SSD layers
        with tf.variable_scope('additional_layers') as scope:
            with tf.variable_scope('conv8'):
                net = self.basic_RFB('RFB', net, 1024, stride=1, scale=1.0, visual=2, is_training=training) 
                print(net.get_shape())
                feats_bottom_up[3] = net
                net = self.basic_RFB('RFB-S', net, 512, stride=2, scale=1.0, visual=2, is_training=training) 
                print(net.get_shape())
                feats_bottom_up[4] = net

            with tf.variable_scope('conv9'):
                net = self.basic_RFB('RFB-S', net, 256, stride=2, scale=1.0, visual=2, is_training=training) 
                print(net.get_shape())
                feats_bottom_up[5] = net

            with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        biases_initializer=None,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': training, 'decay':0.99, 'epsilon':0.00001, 'updates_collections':tf.GraphKeys.UPDATE_OPS, 'fused':None}):
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

