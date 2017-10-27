import tensorflow as tf
from networks.network import Network

class vgg16_convs(Network):
    def __init__(self, input_format, num_classes, num_units, scales, trainable=True):
        self.input_format = input_format
        self.num_classes = num_classes
        self.num_units = num_units
        self.scale = 1 / scales[0]

        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        if input_format == 'RGBD':
            self.data_p = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.gt_label_2d = tf.placeholder(tf.float32, shape=[None, None, None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # define a queue
        queue_size = 25
        if input_format == 'RGBD':
            q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.data, self.data_p, self.gt_label_2d, self.keep_prob])
            data, data_p, gt_label_2d, self.keep_prob_queue = q.dequeue()
            self.layers = dict({'data': data, 'data_p': data_p, 'gt_label_2d': gt_label_2d})
        else:
            q = tf.FIFOQueue(queue_size, [tf.float32, tf.float32, tf.float32])
            self.enqueue_op = q.enqueue([self.data, self.gt_label_2d, self.keep_prob])
            data, gt_label_2d, self.keep_prob_queue = q.dequeue()
            self.layers = dict({'data': data, 'gt_label_2d': gt_label_2d})
        self.close_queue_op = q.close(cancel_pending_enqueues=True)

        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3)
             .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64)
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64)
             .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128)
             .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256)
             .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256)
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256)
             .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512)
             .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512)
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512)
             .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512)
             .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512))

        if self.input_format == 'RGBD':
            (self.feed('data_p')
                 .conv(3, 3, 64, 1, 1, name='conv1_1_p', c_i=3)
                 .conv(3, 3, 64, 1, 1, name='conv1_2_p', c_i=64)
                 .max_pool(2, 2, 2, 2, name='pool1_p')
                 .conv(3, 3, 128, 1, 1, name='conv2_1_p', c_i=64)
                 .conv(3, 3, 128, 1, 1, name='conv2_2_p', c_i=128)
                 .max_pool(2, 2, 2, 2, name='pool2_p')
                 .conv(3, 3, 256, 1, 1, name='conv3_1_p', c_i=128)
                 .conv(3, 3, 256, 1, 1, name='conv3_2_p', c_i=256)
                 .conv(3, 3, 256, 1, 1, name='conv3_3_p', c_i=256)
                 .max_pool(2, 2, 2, 2, name='pool3_p')
                 .conv(3, 3, 512, 1, 1, name='conv4_1_p', c_i=256)
                 .conv(3, 3, 512, 1, 1, name='conv4_2_p', c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv4_3_p', c_i=512)
                 .max_pool(2, 2, 2, 2, name='pool4_p')
                 .conv(3, 3, 512, 1, 1, name='conv5_1_p', c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv5_2_p', c_i=512)
                 .conv(3, 3, 512, 1, 1, name='conv5_3_p', c_i=512))

            (self.feed('conv5_3', 'conv5_3_p')
                 .concat(3, name='concat_conv5')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=1024)
                 .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', trainable=False))

            (self.feed('conv4_3', 'conv4_3_p')
                 .concat(3, name='concat_conv4')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=1024))
        else:
            (self.feed('conv5_3')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv5', c_i=512)
                 .deconv(4, 4, self.num_units, 2, 2, name='upscore_conv5', trainable=False))

            (self.feed('conv4_3')
                 .conv(1, 1, self.num_units, 1, 1, name='score_conv4', c_i=512))

        (self.feed('score_conv4', 'upscore_conv5')
             .add(name='add_score')
             .dropout(self.keep_prob_queue, name='dropout')
             .deconv(int(16*self.scale), int(16*self.scale), self.num_units, int(8*self.scale), int(8*self.scale), name='upscore', trainable=False)
             .conv(1, 1, self.num_classes, 1, 1, name='score', c_i=self.num_units)
             .log_softmax_high_dimension(self.num_classes, name='prob'))

        (self.feed('score')
             .softmax_high_dimension(self.num_classes, name='prob_normalized')
             .argmax_2d(name='label_2d'))
