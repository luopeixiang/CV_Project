# coding=utf-8

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
import numpy as np


def get_label(path='../cifar10/train/labels.txt'):
    """ Get cifar10 class label"""
    with open(path, 'r') as f:
        names = f.readlines()
    names = [n.strip() for n in names]
    return names


def svm_classifier(x_train, y_train, x_test, y_test):

    # 超参数调优
    # C_range = 10.0 ** np.arange(-3, 3)
    # gamma_range = 10.0 ** np.arange(-3, 3)
    # param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
    # print "网格搜索最优超参数...."
    # clf = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-2)
    # clf = svm.SVC()
    # clf.fit(x_train, y_train)
    # print(clf.best_estimator_)
    # print("\n其在验证集上的score为:\n")
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() * 2, params))

    # 使用默认的超参数
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    print("\n分类结果如下:\n")
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred, target_names=get_label()))
