import os
import math
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm

from my_model import MyModel
from test_utils import calculate_metric_percase
from torch.optim import AdamW
from acdc_dataset import SemiDataset

dino_path = '/vip_media/SharedData/Dataset-NII/SEG_code/weight/dinov2_small.pth'
model = MyModel(model_type='small', nclass=4, weights_path=dino_path).cuda()

model_ema = deepcopy(model)
model_ema.eval()
for param in model_ema.parameters():
    param.requires_grad = False

# ACDC dataset paths
l_root_dir = '/vip_media/SharedData/Dataset-NII/vqseg_ysc/acdc/ACDC_png_split_5percent/train-label'
u_root_dir = '/vip_media/SharedData/Dataset-NII/vqseg_ysc/acdc/ACDC_png_split_5percent/train-unlabel'
v_root_dir = '/vip_media/SharedData/Dataset-NII/vqseg_ysc/acdc/ACDC_png_split_5percent/test'


# ----------------- Dataset & Dataloader ----------------- #

img_size = 224
l_dataset = SemiDataset(root_dir=l_root_dir, mode='train_l', size=img_size)
u_dataset = SemiDataset(root_dir=u_root_dir, mode='train_u', size=img_size)
v_dataset = SemiDataset(root_dir=v_root_dir, mode='val', size=img_size)

trainloader_l = DataLoader(l_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=4)
trainloader_u = DataLoader(u_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
valloader    = DataLoader(v_dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=1)


# ----------------- Loss & Optimizer ----------------- #

criterion_l = nn.CrossEntropyLoss(ignore_index=255)
criterion_u = nn.CrossEntropyLoss(reduction='none', ignore_index=255)

ori_lr = 0.000005
lr_multi = 40.0

optimizer = AdamW(
    [
        {'params': [p for p in model.model.backbone.parameters() if p.requires_grad], 'lr': ori_lr},
        {'params': [param for name, param in model.model.named_parameters()
                    if 'backbone' not in name], 'lr': ori_lr * lr_multi},
    ],
    lr=ori_lr, betas=(0.9, 0.999), weight_decay=0.01
)


# ----------------- Training Config (Iter-based) ----------------- #

iters_per_epoch = min(len(trainloader_l), len(trainloader_u))
total_iters = 100000
num_epochs = math.ceil(total_iters / iters_per_epoch)

print("ITERS_PER_EPOCH:", iters_per_epoch)
print("TOTAL ITERS:", total_iters)

pth_folder = "acdc_pth_5per"
os.makedirs(pth_folder, exist_ok=True)

best_dice = 0.0
best_metrics_path = os.path.join(pth_folder, "best_metrics.txt")

global_iter = 0


# ----------------- Function: Evaluation ----------------- #

def evaluate(model, global_iter):
    model.eval()
    print(f"\n========== Evaluation at iter {global_iter} ==========")
    dice_c1, jc_c1, hd95_c1, asd_c1 = [], [], [], []
    dice_c2, jc_c2, hd95_c2, asd_c2 = [], [], [], []
    dice_c3, jc_c3, hd95_c3, asd_c3 = [], [], [], []

    with torch.no_grad():
        for images, labels, img_name in tqdm(valloader, desc="eval", total=len(valloader)):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)   # [B, H, W]

            for i in range(predicted.shape[0]):
                pred_np = predicted[i].cpu().numpy()
                gt_np   = labels[i].cpu().numpy()

                pred_c1 = (pred_np == 1).astype(np.uint8)
                gt_c1   = (gt_np == 1).astype(np.uint8)
                if gt_c1.sum() > 0: 
                    dice, jc, hd95, asd = calculate_metric_percase(pred_c1, gt_c1)
                    dice_c1.append(dice)
                    jc_c1.append(jc)
                    hd95_c1.append(hd95)
                    asd_c1.append(asd)

                pred_c2 = (pred_np == 2).astype(np.uint8)
                gt_c2   = (gt_np == 2).astype(np.uint8)
                if gt_c2.sum() > 0:
                    dice, jc, hd95, asd = calculate_metric_percase(pred_c2, gt_c2)
                    dice_c2.append(dice)
                    jc_c2.append(jc)
                    hd95_c2.append(hd95)
                    asd_c2.append(asd)

                pred_c3 = (pred_np == 3).astype(np.uint8)
                gt_c3   = (gt_np == 3).astype(np.uint8)
                if gt_c3.sum() > 0:
                    dice, jc, hd95, asd = calculate_metric_percase(pred_c3, gt_c3)
                    dice_c3.append(dice)
                    jc_c3.append(jc)
                    hd95_c3.append(hd95)
                    asd_c3.append(asd)

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    mean_dice_c1 = safe_mean(dice_c1)
    mean_dice_c2 = safe_mean(dice_c2)
    mean_dice_c3 = safe_mean(dice_c3)

    mean_jc_c1   = safe_mean(jc_c1)
    mean_jc_c2   = safe_mean(jc_c2)
    mean_jc_c3   = safe_mean(jc_c3)

    mean_hd95_c1 = safe_mean(hd95_c1)
    mean_hd95_c2 = safe_mean(hd95_c2)
    mean_hd95_c3 = safe_mean(hd95_c3)

    mean_asd_c1  = safe_mean(asd_c1)
    mean_asd_c2  = safe_mean(asd_c2)
    mean_asd_c3  = safe_mean(asd_c3)

    mean_dice = (mean_dice_c1 + mean_dice_c2 + mean_dice_c3) / 3.0

    print("------- Per-class metrics -------")
    print(f"Class 1 Dice: {mean_dice_c1:.4f}, JC: {mean_jc_c1:.4f}, HD95: {mean_hd95_c1:.2f}, ASD: {mean_asd_c1:.2f}")
    print(f"Class 2 Dice: {mean_dice_c2:.4f}, JC: {mean_jc_c2:.4f}, HD95: {mean_hd95_c2:.2f}, ASD: {mean_asd_c2:.2f}")
    print(f"Class 3 Dice: {mean_dice_c3:.4f}, JC: {mean_jc_c3:.4f}, HD95: {mean_hd95_c3:.2f}, ASD: {mean_asd_c3:.2f}")
    print("Overall Mean Dice (3 classes avg): {:.4f}".format(mean_dice))
    return mean_dice, (mean_dice_c1, mean_dice_c2, mean_dice_c3)


# ----------------- Training Loop ----------------- #

for epoch in range(num_epochs):

    loader = zip(trainloader_l, trainloader_u)

    epoch_loss = []
    epoch_loss_x = []
    epoch_loss_u = []
    epoch_loss_pfa = []
    epoch_loss_usa = []

    for i, ((img_x, mask_x, img_path),
            (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):

        if global_iter >= total_iters:
            break

        global_iter += 1

        # ----------------- Forward & Loss ----------------- #

        model.train()
        img_x, mask_x = img_x.cuda(), mask_x.cuda()
        img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
        ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()

        # 1) Weak pseudo label (EMA)
        with torch.no_grad():
            pred_u_w = model_ema(img_u_w).detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

        # 2) CutMix on strong aug
        img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
            img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
        img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
            img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

        # 3) Forward supervised + unsupervised
        pred_x, loss_pfa_x,  loss_usage_x = model(img_x, use_pfa=True, use_usage=True)
        pred_u, loss_pfa_u, loss_usage_u = model(
            torch.cat((img_u_s1, img_u_s2)), use_qpm=True, use_pfa=True, use_usage=True
        )
        loss_pfa = loss_pfa_u + loss_pfa_x
        loss_usa = loss_usage_x + loss_usage_u
        pred_u_s1, pred_u_s2 = pred_u.chunk(2)

        # 4) Construct CutMix pseudo labels
        mask_u_w_c1, conf_u_w_c1 = mask_u_w.clone(), conf_u_w.clone()
        mask_u_w_c2, conf_u_w_c2 = mask_u_w.clone(), conf_u_w.clone()
        ignore_c1, ignore_c2 = ignore_mask.clone(), ignore_mask.clone()

        mask_u_w_c1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
        conf_u_w_c1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
        ignore_c1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

        mask_u_w_c2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
        conf_u_w_c2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
        ignore_c2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

        # Supervised loss
        loss_x = criterion_l(pred_x, mask_x)

        # Unsupervised loss
        loss_u1 = criterion_u(pred_u_s1, mask_u_w_c1)
        loss_u1 = loss_u1 * ((conf_u_w_c1 >= 0.95) & (ignore_c1 != 255))
        loss_u1 = loss_u1.sum() / (ignore_c1 != 255).sum().item()

        loss_u2 = criterion_u(pred_u_s2, mask_u_w_c2)
        loss_u2 = loss_u2 * ((conf_u_w_c2 >= 0.95) & (ignore_c2 != 255))
        loss_u2 = loss_u2.sum() / (ignore_c2 != 255).sum().item()

        loss_u = (loss_u1 + loss_u2) / 2

        # Total loss
        # Note: The weighting factors are task-dependent and may require tuning.
        loss = (5 * loss_x + loss_u) / 2 + 0.1 * (loss_pfa + loss_usa)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_loss_x.append(loss_x.item())
        epoch_loss_u.append(loss_u.item())
        epoch_loss_pfa.append(loss_pfa.item())
        epoch_loss_usa.append(loss_usa.item())

        # ----------------- LR schedule ----------------- #
        lr = ori_lr * (1 - global_iter / total_iters) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * lr_multi

        # ----------------- EMA update ----------------- #
        ema_ratio = min(1 - 1 / (global_iter), 0.996)
        for p, p_ema in zip(model.parameters(), model_ema.parameters()):
            p_ema.data.copy_(p_ema * ema_ratio + p.detach() * (1 - ema_ratio))

        if global_iter % 1000 == 0:
            ckpt_path = os.path.join(pth_folder, f"segmentation_model_iter{global_iter}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Iter {global_iter}] Saved ckpt: {ckpt_path}")

            # --- eval ----
            mean_dice, (d1, d2, d3) = evaluate(model, global_iter)
            if mean_dice > best_dice:
                best_dice = mean_dice
                with open(best_metrics_path, "w") as f:
                    f.write(f"Best iter: {global_iter}\n")
                    f.write(f"Mean Dice (3cls): {best_dice:.4f}\n")
                    f.write(f"Class1 Dice: {d1:.4f}\n")
                    f.write(f"Class2 Dice: {d2:.4f}\n")
                    f.write(f"Class3 Dice: {d3:.4f}\n")
                print(f"New Best Dice: {best_dice:.4f}")

    mean_total = np.mean(epoch_loss)
    mean_x     = np.mean(epoch_loss_x)
    mean_u     = np.mean(epoch_loss_u)
    mean_pfa   = np.mean(epoch_loss_pfa)
    mean_usa   = np.mean(epoch_loss_usa)

    print(f"\n===== [Epoch {epoch+1}/{num_epochs}] Loss Summary =====")
    # print(f" Total Loss      : {mean_total:.6f}")
    print(f" Supervised (x)  : {mean_x:.6f}")
    print(f" Unsupervised (u): {mean_u:.6f}")
    print(f" PFA Loss        : {mean_pfa:.6f}")
    # print(f" USE Loss        : {mean_usa:.6f}")
    print("=======================================================\n")

    if global_iter >= total_iters:
        break

print("Training finished!")
