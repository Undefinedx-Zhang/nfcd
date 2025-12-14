import argparse
import shutil
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from utils.metrics import eval_metrics, AverageMeter

from matplotlib import pyplot as plt
from utils.helpers import DeNormalize
import time
import PIL

import torch
from PIL import Image
import numpy as np

import torch
import torchvision
import numpy as np






def save_image(data, path, cmap='hot'):  # Draw heatmap
    """
    Save image data to specified path

    Parameters:
    - data: Image data, can be NumPy array or PyTorch tensor, usually shaped [H, W] or [H, W, C].
    - path: Path to save the image (including filename and extension, e.g., 'output.png').
    - cmap: Color map to use (only valid for grayscale images), default is 'viridis'.

    Returns:
    - None
    """
    # If input data is PyTorch tensor, convert to NumPy array first
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Check if data dimension is 2D (grayscale) or 3D (color)
    if data.ndim == 2:  # Grayscale
        plt.imshow(data, cmap=cmap)
    elif data.ndim == 3:  # Color
        plt.imshow(data)
    else:
        raise ValueError("Input data dimension must be 2 (grayscale) or 3 (color)")

    # Remove axes
    plt.axis('off')
    # Path setting
    path = os.path.join(path, 'hotMap.png')
    # Save image
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    plt.close()  # Close figure to prevent memory leaks


def visualize_prediction(output, prediction, path=None):
    # Create an empty RGB image with size [3, 256, 256]
    result_image = torch.zeros(3, 256, 256, dtype=torch.uint8).cuda()

    # Calculate TP, FN, TN and FP regions
    TP = (output == 1) & (prediction == 1)
    FN = (output == 1) & (prediction == 0)
    TN = (output == 0) & (prediction == 0)
    FP = (output == 0) & (prediction == 1)

    # Fill corresponding colors to RGB image
    result_image[0] = torch.where(FP, torch.tensor(255, dtype=torch.uint8).cuda(), result_image[0])  # Red: [255, 0, 0]
    result_image[1] = torch.where(FN, torch.tensor(255, dtype=torch.uint8).cuda(), result_image[1])  # Green: [0, 255, 0]
    result_image[0] = torch.where(TP, torch.tensor(255, dtype=torch.uint8).cuda(),
                                  result_image[0])  # White: [255, 255, 255]
    result_image[1] = torch.where(TP, torch.tensor(255, dtype=torch.uint8).cuda(),
                                  result_image[1])  # White: [255, 255, 255]
    result_image[2] = torch.where(TP, torch.tensor(255, dtype=torch.uint8).cuda(),
                                  result_image[2])  # White: [255, 255, 255]

    # Transfer from GPU to CPU and convert to NumPy array
    result_image_np = result_image.cpu().numpy()

    # Convert NumPy array to PIL image
    result_image_pil = Image.fromarray(result_image_np.transpose(1, 2, 0))  # Transpose to [H, W, C]

    path_name = os.path.join(path, 'error.png')
    # Save image
    result_image_pil.save(path_name)


def mySave(image_A_path, image_B_path, label, path_name, prediction):
    shutil.copy(image_A_path, os.path.join(path_name, 'image_A.png'))
    shutil.copy(image_B_path, os.path.join(path_name, 'image_B.png'))

    # Convert prediction from Tensor to PIL image
    prediction1 = prediction[0].cpu().numpy() * 255  # Convert to [256, 256] shape
    prediction1 = Image.fromarray(prediction1.astype(np.uint8), mode='L')
    prediction1.save(os.path.join(path_name, 'prediction.png'))

    # Convert label from Tensor to PIL image
    label1 = label[0].cpu().numpy() * 255  # Convert to [256, 256] shape
    label1 = Image.fromarray(label1.astype(np.uint8), mode='L')
    label1.save(os.path.join(path_name, 'label.png'))


def multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=False):
    H, W = (image_A.size(2), image_A.size(3))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image_A = F.pad(image_A, pad=(0, pad_w, 0, pad_h), mode='reflect')
    image_B = F.pad(image_B, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image_A.shape[2], image_A.shape[3]))

    for scale in scales:
        scaled_img_A = F.interpolate(image_A, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_img_B = F.interpolate(image_B, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(A_l=scaled_img_A, B_l=scaled_img_B))
        scaled_prediction = F.softmax(scaled_prediction, dim=1)

        if flip:
            fliped_img_A = scaled_img_A.flip(-1)
            fliped_img_B = scaled_img_B.flip(-1)
            fliped_predictions = upsample(model(A_l=fliped_img_A, B_l=fliped_img_B))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)
    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

name_lst = ['/sd1/kzq/nfcd/DATA/CDD/label/test_01392.jpg', '/sd1/kzq/nfcd/DATA/CDD/label/train_00727.jpg',\
            '/sd1/kzq/nfcd/DATA/CDD/label/train_05136.jpg', '/sd1/kzq/nfcd/DATA/CDD/label/train_05793.jpg',\
            '/sd1/kzq/nfcd/DATA/LEVIR/label/test_49_2.png', '/sd1/kzq/nfcd/DATA/LEVIR/label/test_50_11.png',\
            '/sd1/kzq/nfcd/DATA/LEVIR/label/test_76_12.png','/sd1/kzq/nfcd/DATA/LEVIR/label/test_81_2.png',\
            '/sd1/kzq/nfcd/DATA/WHU/label/whucd_02295.png','/sd1/kzq/nfcd/DATA/WHU/label/whucd_02473.png',\
            '/sd1/kzq/nfcd/DATA/WHU/label/whucd_04727.png','/sd1/kzq/nfcd/DATA/WHU/label/whucd_04874.png']
def main():
    args = parse_arguments()
    method = args.method
    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [1.0, 1.25]

    # DATA LOADER
    config['val_loader']["batch_size"] = 1
    config['val_loader']["num_workers"] = 1
    config['val_loader']["split"] = "test"
    config['val_loader']["shuffle"] = False
    config['val_loader']['data_dir'] = args.Dataset_Path
    config['val_loader']["aug_type"] = 'all'
    loader = dataloaders.CDDataset(config['val_loader'])
    num_classes = 2
    palette = get_voc_pallete(num_classes)

    # MODEL
    dataset = args.Dataset_Path.split('/')[-1]
    percent = config['percent']
    backbone = config['model']['backbone']
    if backbone == 'ResNet50':
        model = models.NF_ResNet50_CD(num_classes=num_classes, config=config, testing=True)
    elif backbone == 'ResNet101':
        model = models.NF_ResNet101_CD(num_classes=num_classes, config=config, testing=True)
    elif backbone == 'HRNet':
        model = models.NF_HRNet_CD(num_classes=num_classes, config=config, testing=True)
    elif backbone == 'NF':  # backward compatibility with older configs
        model = models.NF_ResNet50_CD(num_classes=num_classes, config=config, testing=True)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    print(f'\n{model}\n')
    checkpoint = torch.load(args.model)
    model = torch.nn.DataParallel(model)
    try:
        print("Loading the state dictionery of {} ...".format(args, model))
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.cuda()

    print(config)

    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=120)
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    os.makedirs(f'visuals/{method}/{dataset}20', exist_ok=True)
    for index, data in enumerate(tbar):
        image_A, image_B, label, [image_A_path, image_B_path, label_path] = data
        # if label_path == '/sd1/kzq/nfcd/DATA/CDD/label/test_01392.jpg':
        #     print("6666")
        if label_path[0] not in name_lst:
            continue
        image_A = image_A.cuda()
        image_B = image_B.cuda()
        label = label.cuda()
        image_A_path = image_A_path[0]
        image_B_path = image_B_path[0]
        label_path = label_path[0]
        image_id = image_A_path.split('/')[-1].split('\\')[-1].strip('.jpg').strip('.png')

        # PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image_A, image_B, scales, num_classes)

        output = torch.from_numpy(output).cuda()
        tmp = output.cpu()

        label[label >= 1] = 1
        output = torch.unsqueeze(output, 0)

        _, prediction = torch.max(output, 1)
        path_name = os.path.join(f'visuals/{method}/{dataset}20', image_id)
        os.makedirs(path_name, exist_ok=True)

        mySave(image_A_path, image_B_path, label, path_name, prediction)  # A B Label
        visualize_prediction(label, prediction, path_name)  # error
        save_image(tmp[1], path_name, cmap='viridis')  # hot map

        label = torch.unsqueeze(label, 0)
        correct, labeled, inter, union, tp, fp, tn, fn = eval_metrics(output, label, num_classes)

        total_inter, total_union = total_inter + inter, total_union + union
        total_correct, total_label = total_correct + correct, total_label + labeled
        total_tp, total_fp = total_tp + tp, total_fp + fp
        total_tn, total_fn = total_tn + tn, total_fn + fn

        IOUC_single = (inter / (np.spacing(1) + union))[1]
        with open(f'visuals/{method}/{dataset}20/IoU.txt', "a") as f:
            f.write(f"{image_id} {IOUC_single}\n")





def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config_LEVIR.json', type=str,
                        help='Path to the config file')
    parser.add_argument('--model', default='/sd1/kzq/RCR/outputs/LEVIR/s4GAN_semi_20/best_model.pth', type=str,
                        help='Path to the trained .pth model')
    parser.add_argument('--save', action='store_true', help='Save images')
    parser.add_argument('--Dataset_Path', default="/sd1/kzq/nfcd/DATA/LEVIR", type=str,
                        help='Path to dataset WHU-CD')
    parser.add_argument('--method', default="s4GAN", type=str,
                        help='method')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    main()

