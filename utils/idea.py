import torch
import numpy as np
from scipy.ndimage import label


def refine_pseudo_labels(nf_prop_map: torch.Tensor,
                         target_ul: torch.Tensor,
                         threshold: float = 0.5,
                         connectivity: int = 1) -> torch.Tensor:
    """
    Optimize confidence of connected regions in pseudo labels (supports GPU-CPU hybrid computing)

    Parameters:
    - nf_prop_map: Foreground probability map [B,H,W], value range [0,1]
    - target_ul: Original pseudo labels [B,H,W], binary matrix {0,1}
    - threshold: Confidence filtering threshold, default 0.5
    - connectivity: Connected domain determination method (1=4-neighborhood, 2=8-neighborhood)

    Returns:
    - refined_target: Optimized pseudo labels [B,H,W]
    """
    assert nf_prop_map.shape == target_ul.shape, "Input shapes must be consistent"
    device = target_ul.device
    refined_target = target_ul.clone()

    # Process sample by sample
    for batch_idx in range(target_ul.shape[0]):
        # Transfer to CPU for connected domain processing
        mask_np = target_ul[batch_idx].byte().cpu().numpy()  # uint8 type
        prob_np = nf_prop_map[batch_idx].cpu().numpy()

        # Label connected domains (scipy uses C order by default)
        labeled, n_components = label(mask_np,
                                      structure=np.ones((3, 3)) if connectivity == 2 else None)

        # Iterate through each connected region
        for comp_id in range(1, n_components + 1):
            # Extract pixel coordinates of current connected domain
            y, x = np.where(labeled == comp_id)
            if len(y) == 0:  # Skip empty region
                continue

            # Calculate average confidence of this region
            region_probs = prob_np[y, x]
            mean_confidence = np.mean(region_probs)

            # Confidence filtering logic
            if mean_confidence < threshold:
                # Get binary mask and transfer back to GPU
                comp_mask = torch.from_numpy(labeled == comp_id).to(device)
                # Set low confidence region to 1 (can be changed to 0 based on requirements)
                refined_target[batch_idx][comp_mask] = 1

    return refined_target