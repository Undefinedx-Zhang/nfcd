import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from webencodings.tests import assert_raises


def save_batch_prob_and_labels(A_l, B_l, prob_tensor, true_labels, threshold, save_dir="./output", cmap='viridis'):
    """
    Save the probability maps, predicted label maps, true label maps, and input images (A_l, B_l) as comparison images.

    Parameters:
        A_l (torch.Tensor): Batch of images A, shape (batch_size, 3, height, width).
        B_l (torch.Tensor): Batch of images B, shape (batch_size, 3, height, width).
        prob_tensor (torch.Tensor): The probability map tensor of shape (batch_size, height, width).
        true_labels (torch.Tensor): The true labels tensor of shape (batch_size, height, width), containing ground truth.
        threshold (float): The threshold used to convert probability maps into binary label maps.
        save_dir (str): Directory where the images will be saved.
        cmap (str): Colormap for the probability map.
    """
    if not isinstance(prob_tensor, torch.Tensor) or not isinstance(true_labels, torch.Tensor):
        raise ValueError("Input prob_tensor and true_labels must be PyTorch tensors (torch.Tensor).")

    if prob_tensor.dim() != 3 or true_labels.dim() != 3:
        raise ValueError("prob_tensor and true_labels should be 3D tensors (batch, height, width).")

    if prob_tensor.size() != true_labels.size():
        raise ValueError("prob_tensor and true_labels must have the same size.")

    if A_l.dim() != 4 or B_l.dim() != 4 or A_l.shape != B_l.shape:
        raise ValueError("A_l and B_l should be 4D tensors (batch, 3, height, width) and have the same shape.")

    assert torch.max(prob_tensor) <= 3, "Prediction map out of bounds"

    batch_size, _, height, width = A_l.shape

    # Convert tensors to numpy arrays
    A_l = A_l.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Convert to (batch_size, height, width, 3)
    B_l = B_l.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Convert to (batch_size, height, width, 3)

    A_l = np.clip(A_l, 0, 255).astype(np.uint8)  # If integer values, ensure in [0, 255] range
    B_l = np.clip(B_l, 0, 255).astype(np.uint8)  # If integer values, ensure in [0, 255] range

    prob_maps = prob_tensor.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()

    os.makedirs(save_dir, exist_ok=True)

    for i in range(batch_size):
        prob_map = prob_maps[i]
        label_map = (prob_map >= threshold).astype(int)
        true_label_map = true_labels[i].astype(int)
        img_A = A_l[i]
        img_B = B_l[i]

        # Create a 1x5 subplot layout
        fig, axs = plt.subplots(1, 5, figsize=(25, 6))

        # Subplot 1: A_l (original image A)
        axs[0].imshow(img_A)
        axs[0].set_title("Image A")
        axs[0].axis('off')

        # Subplot 2: B_l (original image B)
        axs[1].imshow(img_B)
        axs[1].set_title("Image B")
        axs[1].axis('off')

        # Subplot 3: Probability Map
        im1 = axs[2].imshow(prob_map, cmap=cmap)
        axs[2].set_title("Probability Map")
        fig.colorbar(im1, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].axis('off')

        # Subplot 4: Predicted Label Map
        im2 = axs[3].imshow(label_map, cmap='gray')
        axs[3].set_title(f"Predicted Label (Threshold = {threshold})")
        fig.colorbar(im2, ax=axs[3], fraction=0.046, pad=0.04)
        axs[3].axis('off')

        # Subplot 5: True Label Map
        im3 = axs[4].imshow(true_label_map, cmap='gray')
        axs[4].set_title("True Label Map")
        fig.colorbar(im3, ax=axs[4], fraction=0.046, pad=0.04)
        axs[4].axis('off')

        # Save image
        save_path = os.path.join(save_dir, f"sample_{i + 1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    print(f"Images saved to directory: {save_dir}")
