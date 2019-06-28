# coding=utf-8
from utils import load_cifar10_data
from utils import extract_DenseSift_descriptors
from utils import build_codebook
from utils import input_vector_encoder
from classifier import svm_classifier

import numpy as np


VOC_SIZE = 100
PYRAMID_LEVEL = 2
DSIFT_STEP_SIZE = 4
# DSIFT_STEP_SIZE is related to the function
# extract_DenseSift_descriptors in utils.py
# and build_spatial_pyramid in spm.py

def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    from utils import DSIFT_STEP_SIZE as s
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
    h = image.shape[0] / step_size
    w = image.shape[1] / step_size

    idx_crop = np.array(range(len(descriptor))).reshape(h, w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height/bh, width/bw, bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
        idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid


def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))





if __name__ == '__main__':

    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    print "Dense SIFT feature extraction"
    x_train_feature = [extract_DenseSift_descriptors(img) for img in x_train]
    x_test_feature = [extract_DenseSift_descriptors(img) for img in x_test]
    x_train_kp, x_train_des = zip(*x_train_feature)
    x_test_kp, x_test_des = zip(*x_test_feature)

    print "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test))
    print "Codebook Size: {:d}".format(VOC_SIZE)
    print "Pyramid level: {:d}".format(PYRAMID_LEVEL)
    print "Building the codebook, it will take some time"
    codebook = build_codebook(x_train_des, VOC_SIZE)
    import cPickle
    with open('./spm_lv1_codebook.pkl', 'w') as f:
        cPickle.dump(codebook, f)

    print "Spatial Pyramid Matching encoding"
    x_train = [spatial_pyramid_matching(x_train[i],
                                        x_train_des[i],
                                        codebook,
                                        level=PYRAMID_LEVEL)
               for i in xrange(len(x_train))]

    x_test = [spatial_pyramid_matching(x_test[i],
                                       x_test_des[i],
                                       codebook,
                                       level=PYRAMID_LEVEL) for i in xrange(len(x_test))]

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    svm_classifier(x_train, y_train, x_test, y_test)
