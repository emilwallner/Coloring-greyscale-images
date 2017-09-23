# Import libraries to list and create directories
import sys
import tarfile
import urllib
import urllib.request
from os.path import join, isfile, expanduser

import tensorflow as tf

from dataset.shared import dir_root

checkpoint_url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'


def maybe_download_inception(checkpoint_source):
    """
    Ensure that the checkpoint for Inception Resnet v2 exists
    :param checkpoint_source: if a link it gets downloaded, if an archive it
    gets extracted, if a path in the filesystem, it just check it exists
    :return: the [downloaded] [uncompressed] ready-for-use file path
    """

    # If the source is a link download it
    if checkpoint_source.startswith('http://'):
        print('Using urllib.request for the checkpoint file is extremely',
              'slow, it\'s better to download the tgz archive manualy',
              'and pass its path to this constructor', file=sys.stderr)
        checkpoint_source, _ = urllib.request.urlretrieve(
            checkpoint_source,
            join(dir_root, 'inception_resnet_v2_2016_08_30.ckpt.tgz'))

    # If the source is an archive extract it
    if checkpoint_source.endswith('.tgz'):
        with tarfile.open(checkpoint_source, 'r:gz') as tar:
            tar.extractall(path=dir_root)
            checkpoint_source = join(
                dir_root, 'inception_resnet_v2_2016_08_30.ckpt')

    checkpoint_source = expanduser(checkpoint_source)
    if not isfile(checkpoint_source):
        raise Exception('Checkpoint not valid: {}'.format(checkpoint_source))

    return checkpoint_source


def prepare_image_for_inception(input_tensor):
    """
    Pre-processes an image tensor ``(int8, range [0, 255])``
    to be fed into inception ``(float32, range [-1, +1])``

    :param input_tensor:
    :return:
    """
    res = tf.cast(input_tensor, dtype=tf.float32)
    res = 2 * res / 255 - 1
    res = tf.reshape(res, [-1, 299, 299, 3])
    return res
