import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from medpy import metric

from my_model import MyModel
from transform import normalize

def save_multiclass_mask_color_rgb(mask, save_path):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    color[mask == 1] = [255, 0, 0]   # red
    color[mask == 2] = [0, 255, 0]   # green
    color[mask == 3] = [0, 0, 255]   # blue

    Image.fromarray(color).save(str(save_path))


def calculate_metric_percase_binary(pred_bin, gt_bin):
    pred_bin = pred_bin.astype(np.uint8)
    gt_bin   = gt_bin.astype(np.uint8)

    if pred_bin.sum() > 0 and gt_bin.sum() > 0:
        dice     = metric.binary.dc(pred_bin, gt_bin)
        jaccard  = metric.binary.jc(pred_bin, gt_bin)
        asd      = metric.binary.asd(pred_bin, gt_bin)
        hd95     = metric.binary.hd95(pred_bin, gt_bin)
    else:
        dice, jaccard, hd95, asd = 0.0, 0.0, 0.0, 0.0

    return dice, jaccard, hd95, asd


def sliding_window_predict(img_np, model, device,
                           window_size=224,
                           stride=224,
                           num_classes=4):
    model.eval()
    H, W, _ = img_np.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        pad_img = np.pad(
            img_np,
            pad_width=((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0
        )
    else:
        pad_img = img_np

    pH, pW, _ = pad_img.shape

    score_map = np.zeros((num_classes, pH, pW), dtype=np.float32)
    count_map = np.zeros((pH, pW), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, pH - window_size + 1, stride):
            for x in range(0, pW - window_size + 1, stride):
                patch_np = pad_img[y:y + window_size, x:x + window_size, :]
                patch_pil = Image.fromarray(patch_np)
                patch_tensor = normalize(patch_pil)  # (3,win,win)
                patch_tensor = patch_tensor.unsqueeze(0).to(device)  # (1,3,win,win)

                logits = model(patch_tensor)  # (1,C,win,win)
                logits_np = logits.squeeze(0).cpu().numpy()  # (C,win,win)

                score_map[:, y:y + window_size, x:x + window_size] += logits_np
                count_map[y:y + window_size, x:x + window_size] += 1.0

    count_map[count_map == 0] = 1.0
    score_map = score_map / count_map[None, :, :]

    pred_full = np.argmax(score_map, axis=0).astype(np.uint8)  # (pH,pW)
    pred = pred_full[:H, :W]
    return pred


def get_case_id(img_name: str) -> str:
    base = os.path.splitext(img_name)[0]
    case_id = base.split('_')[0]
    return case_id



def build_case_index(root_dir):
    img_dir = os.path.join(root_dir, "image")
    mask_dir = os.path.join(root_dir, "mask")

    case_dict = {}

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.endswith(".png"):
            continue
        case_id = get_case_id(img_name)
        img_path = os.path.join(img_dir, img_name)
        mask_name = img_name.replace(".png", ".npy")
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue

        if case_id not in case_dict:
            case_dict[case_id] = []
        case_dict[case_id].append({
            "name": img_name,
            "img_path": img_path,
            "mask_path": mask_path
        })

    for cid in case_dict:
        case_dict[cid] = sorted(case_dict[cid], key=lambda x: x["name"])

    return case_dict

def test_single_case(case_id, slice_list, model, device, save_dir=None):
    sample = slice_list[0]
    img0 = Image.open(sample["img_path"]).convert('RGB')
    H, W = np.array(img0).shape[:2]
    Z = len(slice_list)

    prediction = np.zeros((Z, H, W), dtype=np.uint8)
    label = np.zeros((Z, H, W), dtype=np.uint8)
    if save_dir is not None:
        case_save_dir = os.path.join(save_dir, case_id)
        os.makedirs(case_save_dir, exist_ok=True)
    else:
        case_save_dir = None

    for z, item in enumerate(slice_list):
        img = Image.open(item["img_path"]).convert('RGB')
        img_np = np.array(img).astype(np.uint8)  # (H,W,3)
        gt_np = np.load(item["mask_path"]).astype(np.uint8)  # (H,W)

        pred_2d = sliding_window_predict(
            img_np, model, device,
            window_size=224,
            stride=224,
            num_classes=4
        )

        prediction[z] = pred_2d
        label[z] = gt_np

        if case_save_dir is not None:
            name_base = os.path.splitext(item["name"])[0]
            Image.fromarray(img_np).save(
                os.path.join(case_save_dir, f"{name_base}_img.png")
            )
            save_multiclass_mask_color_rgb(
                gt_np,
                os.path.join(case_save_dir, f"{name_base}_gt.png")
            )
            save_multiclass_mask_color_rgb(
                pred_2d,
                os.path.join(case_save_dir, f"{name_base}_pred.png")
            )

    first_metric  = calculate_metric_percase_binary(prediction == 1, label == 1)
    second_metric = calculate_metric_percase_binary(prediction == 2, label == 2)
    third_metric  = calculate_metric_percase_binary(prediction == 3, label == 3)

    return first_metric, second_metric, third_metric


def main():
    dino_path = "/vip_media/SharedData/Dataset-NII/SEG_code/weight/dinov2_small.pth"
    test_root = "/vip_media/SharedData/Dataset-NII/vqseg_ysc/acdc/ACDC_png_split/test"
    iters = 14000
    ckpt_path = f"/vip_media/SharedData/Dataset-NII/vqseg_fin/acdc_pth_usage_5per/segmentation_model_iter{iters}.pth"
    print(f"test with {iters} ckpt.")
    save_dir = "acdc_test"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(model_type="small", nclass=4, weights_path=dino_path)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.reset_codebook_stats()
    case_dict = build_case_index(test_root)
    case_ids = sorted(case_dict.keys())
    print(f"Found {len(case_ids)} cases.")
    first_total  = np.zeros(4, dtype=np.float64)
    second_total = np.zeros(4, dtype=np.float64)
    third_total  = np.zeros(4, dtype=np.float64)
    for cid in tqdm(case_ids, desc="eval-cases"):
        slice_list = case_dict[cid]
        first_metric, second_metric, third_metric = test_single_case(
            cid, slice_list, model, device, save_dir=save_dir
        )
        first_total  += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total  += np.asarray(third_metric)
        # print(cid, "cls1:", first_metric, "cls2:", second_metric, "cls3:", third_metric)
        # case_metrics.append((cid, first_metric, second_metric, third_metric))

    n_case = len(case_ids)
    first_avg  = first_total / n_case   # (dice1, jaccard1, hd951, asd1)
    second_avg = second_total / n_case  # (dice2, jaccard2, hd952, asd2)
    third_avg  = third_total / n_case   # (dice3, jaccard3, hd953, asd3)

    print("Class-1 (mean dice, jaccard, hd95, asd):", first_avg)
    print("Class-2 (mean dice, jaccard, hd95, asd):", second_avg)
    print("Class-3 (mean dice, jaccard, hd95, asd):", third_avg)

    overall = (first_avg + second_avg + third_avg) / 3.0
    print("Overall (mean dice, jaccard, hd95, asd) over 3 classes:", overall)

    if hasattr(model, "get_codebook_usage"):
        usage = model.get_codebook_usage()
        print(
            f"Codebook usage: {usage['used_codes']}/{usage['total_codes']} "
            f"({usage['usage_ratio']*100:.2f}%)"
        )
        print(
            f"Avg tokens per used code: "
            f"{usage['avg_tokens_per_used_code']:.1f}"
        )

if __name__ == "__main__":
    main()
