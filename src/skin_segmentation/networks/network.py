import numpy as np
from math import ceil
import tensorflow as tf

DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            print op_name
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        print op_name + ' ' + param_name + ' assigned'
                    except ValueError:
                        if not ignore_missing:
                            raise
            # try to assign dual weights
            with tf.variable_scope(op_name+'_p', reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        print op_name + '_p ' + param_name + ' assigned'
                    except ValueError:
                        if not ignore_missing:
                            raise

            with tf.variable_scope(op_name+'_d', reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def make_deconv_filter(self, name, f_shape, trainable=True):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=trainable)
        return var

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True, biased=True, c_i=-1):
        self.validate_padding(padding)
        if isinstance(input, tuple):
            input = input[0]
        if c_i == -1:
            c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)

            if group==1:
                output = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)    
        return output

    @layer
    def conv3d(self, input, k_d, k_h, k_w, c_i, c_o, s_d, s_h, s_w, name, reuse=None, relu=True, padding=DEFAULT_PADDING, trainable=True):
        self.validate_padding(padding)
        if isinstance(input, tuple):
            input = input[0]
        with tf.variable_scope(name, reuse=reuse) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_d, k_h, k_w, c_i, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)
            conv = tf.nn.conv3d(input, kernel, [1, s_d, s_h, s_w, 1], padding=padding)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def deconv(self, input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, padding=DEFAULT_PADDING, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        with tf.variable_scope(name, reuse=reuse) as scope:
            # Compute shape out of input
            in_shape = tf.shape(input)
            h = in_shape[1] * s_h
            w = in_shape[2] * s_w
            new_shape = [in_shape[0], h, w, c_o]
            output_shape = tf.stack(new_shape)

            # filter
            f_shape = [k_h, k_w, c_o, c_i]
            weights = self.make_deconv_filter('weights', f_shape, trainable)
        return tf.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding, name=scope.name)

    @layer
    def backproject(self, input, grid_size, kernel_size, threshold, name):
        return backproject_op.backproject(input[0], input[1], input[2], input[3], input[4], grid_size, kernel_size, threshold, name=name)

    @layer
    def compute_flow(self, input, kernel_size, threshold, max_weight, name):
        return compute_flow_op.compute_flow(input[0], input[1], input[2], input[3], input[4], kernel_size, threshold, max_weight, name=name)

    @layer
    def triplet_loss(self, input, margin, name):
        return triplet_loss_op.triplet_loss(input[0], input[1], tf.cast(input[2], tf.int32), margin, name=name)

    @layer
    def matching_loss(self, input, filename_model, name):
        return matching_loss_op.matching_loss(input[0], input[1], input[2], input[3], input[4], input[5], input[6], filename_model, name=name)

    @layer
    def project(self, input, kernel_size, threshold, name):
        return project_op.project(input[0], input[1], input[2], kernel_size, threshold, name=name)

    @layer
    def compute_label(self, input, name):
        return compute_label_op.compute_label(input[0], input[1], input[2], name=name)

    @layer
    def hough_voting(self, input, is_train, name):
        return hough_voting_op.hough_voting(input[0], input[1], input[2], input[3], input[4], is_train, name=name)

    @layer
    def rnn_gru2d(self, input, num_units, channels, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            gru2d = GRU2DCell(num_units, channels)
            return gru2d(input[0], input[1][0], input[1][1], scope)

    @layer
    def rnn_gru2d_original(self, input, num_units, channels, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            gru2d = GRUCell(num_units, channels)
            return gru2d(input[0], input[1][0], input[1][1], scope)

    @layer
    def rnn_gru3d(self, input, num_units, channels, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            gru3d = GRU3DCell(num_units, channels)
            return gru3d(input[0][0], input[0][2], input[1], scope)

    @layer
    def rnn_vanilla2d(self, input, num_units, channels, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            vanilla2d = Vanilla2DCell(num_units, channels)
            return vanilla2d(input[0], input[1], scope)
    
    @layer
    def rnn_add2d(self, input, num_units, channels, step, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            add2d = Add2DCell(num_units, channels)
            return add2d(input[0], input[1], step, scope)

    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def lrelu(self, input, name, leak=0.2):
        return tf.maximum(input, leak * input)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, pool_channel, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        return roi_pool_op.roi_pool(input[0], input[1],
                              pooled_height,
                              pooled_width,
                              spatial_scale,
                              pool_channel,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        if isinstance(inputs[0], tuple):
            inputs[0] = inputs[0][0]

        if isinstance(inputs[1], tuple):
            inputs[1] = inputs[1][0]

        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        if isinstance(inputs[0], tuple):
            inputs[0] = inputs[0][0]

        if isinstance(inputs[1], tuple):
            inputs[1] = inputs[1][0]

        return tf.add_n(inputs, name=name)

    @layer
    def multiply(self, input, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        return tf.multiply(input[0], input[1], name=name)

    @layer
    def multiply_sum(self, inputs, num_classes, name):
        prob = tf.reshape(inputs[0], [-1, num_classes])
        image = tf.matmul(prob, inputs[1])
        input_shape = tf.shape(inputs[0])
        return tf.reshape(image, [input_shape[0], input_shape[1], input_shape[2], 3])

    @layer
    def l2_normalize(self, input, dim, name):
        return tf.nn.l2_normalize(input, dim, name=name)

    @layer
    def fc(self, input, num_out, name, num_in=-1, height=-1, width=-1, channel=-1, reuse=None, relu=True, trainable=True):
        with tf.variable_scope(name, reuse=reuse) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            if height > 0 and width > 0 and channel > 0:
                input_shape = tf.shape(input)
                input = tf.reshape(input, [input_shape[0], height, width, channel])

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                if num_in == -1:
                    feed_in, dim = (input, int(input_shape[-1]))
                else:
                    feed_in, dim = (input, int(num_in))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def reshape(self, input, shape, name):
        return tf.reshape(input, shape, name)

    @layer
    def argmax_3d(self, input, name):
        return tf.argmax(input, 4, name)

    @layer
    def argmax_2d(self, input, name):
        return tf.to_int32(tf.argmax(input, 3, name))

    @layer
    def tanh(self, input, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        return tf.nn.tanh(input, name)

    @layer
    def softmax(self, input, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        return tf.nn.softmax(input, name)

    @layer
    def log_softmax(self, input, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        return tf.nn.log_softmax(input, name=name)

    @layer
    def tile(self, input, num, name):
        if isinstance(input, tuple):
            input = input[0]

        input_shape = input.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)

        return tf.tile(input, multiples)

    @layer
    def softmax_high_dimension(self, input, num_classes, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        input_shape = input.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num_classes

        m = tf.reduce_max(input, reduction_indices=[ndims-1], keep_dims=True)
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)
        e = tf.exp(tf.subtract(input, tf.tile(m, multiples)))
        s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
        return tf.div(e, tf.tile(s, multiples))


    @layer
    def log_softmax_high_dimension(self, input, num_classes, name):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        input_shape = input.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num_classes

        m = tf.reduce_max(input, reduction_indices=[ndims-1], keep_dims=True)
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)
        d = tf.subtract(input, tf.tile(m, multiples))
        e = tf.exp(d)
        s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
        return tf.subtract(d, tf.log(tf.tile(s, multiples)))

    @layer
    def batch_normalization(self, input, name, scale_offset=False, relu=False, reuse=None, trainable=True):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name, reuse=reuse) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape, trainable=trainable)
                offset = self.make_var('offset', shape=shape, trainable=trainable)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape, initializer=tf.constant_initializer(0.0), trainable=trainable),
                variance=self.make_var('variance', shape=shape, initializer=tf.constant_initializer(1.0), trainable=trainable),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def batch_norm(self, input, name, c_i=-1, momentum=0.9, epsilon=1e-5, is_training=True, relu=False, reuse=None):

        if c_i != -1:
            input_shape = tf.shape(input)
            input = tf.reshape(input, [input_shape[0], input_shape[1], input_shape[2], c_i])

        with tf.variable_scope(name, reuse=reuse) as scope:
            output = tf.contrib.layers.batch_norm(input,
                      decay=momentum, 
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=is_training,
                      scope=scope)
            if relu:
                output = tf.nn.relu(output)

            return output

    @layer
    def dropout(self, input, keep_prob, name):
        if isinstance(input, tuple):
            input = input[0]
        return tf.nn.dropout(input, keep_prob, name=name)


    def make_3d_spatial_filter(self, name, size, channel, theta):
        depth = size
        height = size
        width = size
        kernel = np.zeros([size, size, size])
        c = size / 2
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    kernel[d, h, w] = np.exp( -1 * ((d - c) * (d - c) + (h - c) * (h - c) + (w - c) * (w - c)) / (2.0 * theta * theta) )
        kernel[c, c, c] = 0

        weights = np.zeros([size, size, size, channel, channel])
        for i in range(channel):
            weights[:, :, :, i, i] = kernel

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=False)
        return var


    @layer
    def meanfield_3d(self, input, num_classes, name, reuse=None, trainable=True):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]
        with tf.variable_scope(name, reuse=reuse) as scope:
            # softmax
            '''
            input_shape = input.get_shape()
            ndims = input_shape.ndims
            array = np.ones(ndims)
            array[-1] = num_classes

            m = tf.reduce_max(input, reduction_indices=[ndims-1], keep_dims=True)
            multiples = tf.convert_to_tensor(array, dtype=tf.int32)
            e = tf.exp(tf.sub(input, tf.tile(m, multiples)))
            s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
            Q = tf.div(e, tf.tile(s, multiples))
            '''
            # message passing
            weights_message = self.make_3d_spatial_filter('weights_message', 3, num_classes, 0.8)
            message = tf.nn.conv3d(input, weights_message, [1, 1, 1, 1, 1], padding=DEFAULT_PADDING)

            # compatibility transform
            kernel = np.zeros([1, 1, 1, num_classes, num_classes])
            for i in range(num_classes):
                kernel[0, 0, 0, i, i] = 1
            init_weights = tf.constant_initializer(value=kernel, dtype=tf.float32)
            weights_comp = self.make_var('weights_comp', [1, 1, 1, num_classes, num_classes], init_weights, trainable)
            compatibility = tf.nn.conv3d(message, weights_comp, [1, 1, 1, 1, 1], padding=DEFAULT_PADDING)

            # add unary potential
            return input + compatibility

    def make_2d_spatial_filter(self, name, size, channel, theta):
        height = size
        width = size
        kernel = np.zeros([size, size])
        c = size / 2
        for h in range(height):
            for w in range(width):
                kernel[h, w] = np.exp( -1 * ((h - c) * (h - c) + (w - c) * (w - c)) / (2.0 * theta * theta) )
        kernel[c, c] = 0

        weights = np.zeros([size, size, channel, channel])
        for i in range(channel):
            weights[:, :, i, i] = kernel

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=False)
        return var


    @layer
    def meanfield_2d(self, input, num_steps, num_classes, name, reuse=None, trainable=True):
        # only use the first input
        if isinstance(input, tuple):
            input = input[0]

        input_shape = input.get_shape()
        ndims = input_shape.ndims
        array = np.ones(ndims)
        array[-1] = num_classes
        multiples = tf.convert_to_tensor(array, dtype=tf.int32)
        unary = input

        for i in range(num_steps):
            if i > 0:
                reuse = True
            with tf.variable_scope(name, reuse=reuse) as scope:
                # softmax
                m = tf.reduce_max(unary, reduction_indices=[ndims-1], keep_dims=True)
                e = tf.exp(tf.sub(unary, tf.tile(m, multiples)))
                s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
                Q = tf.div(e, tf.tile(s, multiples))

                # message passing
                weights_message = self.make_2d_spatial_filter('weights_message', 3, num_classes, 0.8)
                message = tf.nn.conv2d(Q, weights_message, [1, 1, 1, 1], padding=DEFAULT_PADDING)

                # compatibility transform
                kernel = np.zeros([1, 1, num_classes, num_classes])
                for i in range(num_classes):
                    kernel[0, 0, i, i] = 1            
                init_weights = tf.constant_initializer(value=kernel, dtype=tf.float32)
                weights_comp = self.make_var('weights_comp', [1, 1, num_classes, num_classes], init_weights, trainable)
                compatibility = tf.nn.conv2d(message, weights_comp, [1, 1, 1, 1], padding=DEFAULT_PADDING)

                # add unary potential
                unary = unary + compatibility

        return unary
