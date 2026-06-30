import numpy as np

def clDice(predict, target):
    """
    Calculate centerline Dice coefficient between prediction and target.

    Args:
        predict: Binary prediction array
        target: Binary target array

    Returns:
        Centerline Dice score (float between 0 and 1)
    """
    if np.sum(target) == 0:
        return 1.0 if np.sum(predict) == 0 else 0.0

    # Get skeleton/centerline of predictions and targets
    predict_skeleton = _get_skeleton(predict)
    target_skeleton = _get_skeleton(target)

    # Topology Precision: intersection of predicted skeleton and target mask
    precision = np.sum(predict_skeleton * target) / (np.sum(predict_skeleton) + 1e-10)
    
    # Topology Sensitivity (Recall): intersection of target skeleton and predicted mask
    recall = np.sum(target_skeleton * predict) / (np.sum(target_skeleton) + 1e-10)

    # clDice is the harmonic mean of topology precision and topology sensitivity
    cl_dice = 2 * precision * recall / (precision + recall + 1e-10)

    return cl_dice


def _get_skeleton(binary_mask):
    """
    Extract skeleton/centerline from binary mask using medial axis.

    Args:
        binary_mask: Binary input array

    Returns:
        Skeleton array
    """
    from scipy import ndimage

    if np.sum(binary_mask) == 0:
        return binary_mask.astype(np.float32)

    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary_mask)
    except ImportError:
        # Fallback: use distance transform for medial axis
        dist = ndimage.distance_transform_edt(binary_mask)
        # Skeleton is where distance is local maximum
        skeleton = ndimage.maximum_filter(dist, size=3) == dist
        skeleton = skeleton & (binary_mask > 0)

    return skeleton.astype(np.float32)
