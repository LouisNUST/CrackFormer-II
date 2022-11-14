import numpy as np
import cv2
import os
import math
def make_dataset():
    pred_imgs, gt_imgs = [], []
    path="/home/nlg/yj/mxy/code/crack/datasets/CrackLS315/test/"

    ls=os.listdir(path)
    for each in ls:
        if each.endswith("jpg"):
            img=cv2.imread(path+each)
            # gt=img[512:,:,:]
            # img=img[:512,:,:]
            gt=cv2.imread(path+each[:-3]+"bmp")
            print(img.shape, gt.shape)

            pred_imgs.append(img)
            gt_imgs.append(gt)
    return pred_imgs,gt_imgs
def get_statistics( pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]
def cal_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_acc_all = []
    for pred, gt in zip(pred_list, gt_list):
        statistics = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r_acc = tp / (tp + fn)

            if p_acc + r_acc == 0:
                f1 = 0
            else:
                f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            statistics.append([thresh, f1])
        max_f = np.amax(statistics, axis=0)
        final_acc_all.append(max_f[1])
    return np.mean(final_acc_all)
