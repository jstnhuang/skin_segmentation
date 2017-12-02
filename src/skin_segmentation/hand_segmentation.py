import cv2
import networks
import numpy as np
import tensorflow as tf
import utils.blob


def _get_image_blob(im, im_depth, pixel_means):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    # RGB
    rgb = im.astype(np.float32)
    rgb -= pixel_means

    # depth
    depth = im_depth.astype(np.float32)
    depth = depth / depth.max() * 255
    depth = np.tile(depth[:, :, np.newaxis], (1, 1, 3))
    depth -= pixel_means

    # Create a blob to hold the input images
    blob = utils.blob.im_to_blob(rgb, 3)
    blob_depth = utils.blob.im_to_blob(depth, 3)

    return blob, blob_depth


class HandSegmentation(object):
    """HandSegmentation segments hands in an RGBD image.
    """
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

    def __init__(self, checkpoint_path):
        self._checkpoint_path = checkpoint_path

        _network_input = 'RGBD'
        _num_classes = 2
        _num_units = 64
        _scales_base = (1.0, )
        _trainable = False

        self._network = networks.vgg16_convs(
            _network_input, _num_classes, _num_units, _scales_base, _trainable)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self._sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options))
        saver = tf.train.Saver()
        saver.restore(self._sess, checkpoint_path)

    def segment(self, rgb, depth):
        """Perform segmentation on the RGB and depth images.

        Args:
            rgb: An OpenCV image with BGR format, UC3 encoding
            depth: An OpenCV image, with 16UC1 encoding, units in millimeters

        Returns:
            labels: An OpenCV image with 8UC1 encoding, 1 = hand, 0 = not hand
        """
        # You can comment this out if you know RGB/Depth are multiples of 16
        if (rgb.shape[0] % 16 != 0 or rgb.shape[1] % 16 != 0
                or depth.shape[0] % 16 != 0 or depth.shape[1] % 16 != 0):
            # read color image
            im = utils.blob.pad_im(rgb, 16)

            # read depth image
            im_depth = utils.blob.pad_im(depth, 16)

            labels, probs = self.im_segment_single_frame(im, im_depth)
            labels = utils.blob.unpad_im(labels, 16)
            return labels
        labels, probs = self.im_segment_single_frame(rgb, depth)
        return labels

    def im_segment_single_frame(self, im, im_depth):
        num_classes = 2

        # compute image blob
        im_blob, im_depth_blob = _get_image_blob(im, im_depth,
                                                 HandSegmentation.PIXEL_MEANS)

        # use a fake label blob of ones
        height = im_depth.shape[0]
        width = im_depth.shape[1]
        label_blob = np.ones((1, height, width, num_classes), dtype=np.float32)

        # forward pass
        data_blob = im_blob
        data_p_blob = im_depth_blob

        feed_dict = {
            self._network.data: data_blob,
            self._network.data_p: data_p_blob,
            self._network.gt_label_2d: label_blob,
            self._network.keep_prob: 1.0
        }
        self._sess.run(self._network.enqueue_op, feed_dict=feed_dict)

        labels_2d, probs = self._sess.run([
            self._network.get_output('label_2d'),
            self._network.get_output('prob_normalized')
        ])

        return labels_2d[0, :, :].astype(np.int32), probs[0, :, :, :]
