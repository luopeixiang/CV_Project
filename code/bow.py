# coding=utf-8

import cPickle

from utils import load_cifar10_data
from utils import extract_sift_descriptors
from utils import build_codebook
from utils import input_vector_encoder
from classifier import svm_classifier

import numpy as np


VOC_SIZE = 100


if __name__ == '__main__':

    # 加载数据集 x表示图片，y是其对应的label
    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    # 抽取sift特征
    print "抽取 SIFT 特征..."
    x_train = [extract_sift_descriptors(img) for img in x_train]
    x_test = [extract_sift_descriptors(img) for img in x_test]

    # 移除没有抽取出sift特征的图片
    x_train = [each for each in zip(x_train, y_train) if each[0] is not None]
    x_train, y_train = zip(*x_train)
    x_test = [each for each in zip(x_test, y_test) if each[0] is not None]
    x_test, y_test = zip(*x_test)

    print "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test))
    print "Vocab Size: {:d}".format(VOC_SIZE)

    # 使用Kmeans聚类算法对抽取出来的特征进行聚类，VOC_SIZE表示k-means中的k
    # 这一步可能需要很长时间...
    print "构建词典..."
    codebook = build_codebook(x_train, voc_size=VOC_SIZE)
    with open('./bow_codebook.pkl', 'w') as f:
        cPickle.dump(codebook, f)

    # 将图片转成它的向量化表示
    print "图片向量化表示..."
    x_train = [input_vector_encoder(x, codebook) for x in x_train]
    x_train = np.asarray(x_train)
    x_test = [input_vector_encoder(each, codebook) for each in x_test]
    x_test = np.asarray(x_test)

    # 输入到分类器进行分类
    print("训练与测试分类器...")
    svm_classifier(x_train, y_train, x_test, y_test)
