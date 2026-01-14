from medpy import metric
import numpy as np
# def calculate_metric_percase(pred, gt):
#     # pred[pred > 0] = 1
#     # gt[gt > 0] = 1
#     pred_bin = (pred > 0).astype(np.uint8)
#     gt_bin   = (gt > 0).astype(np.uint8)
#     if pred.sum()>0 and gt.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         jc = metric.binary.jc(pred, gt)
#         asd = metric.binary.asd(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#     else:
#         #print('wrong')
#         dice, jc, hd95, asd = 0.0,0.0,0.0,0.0
#     return dice, jc, hd95, asd
def calculate_metric_percase_multiclass(pred, gt, num_classes=4):
    dice_list, jc_list, hd95_list, asd_list = [], [], [], []

    for c in range(1, num_classes):  # 跳过背景0
        pred_c = (pred == c).astype(np.uint8)
        gt_c   = (gt   == c).astype(np.uint8)

        if pred_c.sum() > 0 and gt_c.sum() > 0:
            dice = metric.binary.dc(pred_c, gt_c)
            jc   = metric.binary.jc(pred_c, gt_c)
            asd  = metric.binary.asd(pred_c, gt_c)
            hd95 = metric.binary.hd95(pred_c, gt_c)
        else:
            dice, jc, hd95, asd = 0.0, 0.0, 0.0, 0.0

        dice_list.append(dice)
        jc_list.append(jc)
        hd95_list.append(hd95)
        asd_list.append(asd)

    return dice_list, jc_list, hd95_list, asd_list


def calculate_metric_percase(pred_bin, gt_bin):
    pred = pred_bin.astype(np.uint8)
    gt   = gt_bin.astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc   = metric.binary.jc(pred, gt)
        asd  = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    else:
        dice, jc, hd95, asd = 0.0, 0.0, 0.0, 0.0

    return dice, jc, hd95, asd