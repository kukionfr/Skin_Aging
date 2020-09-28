import numpy as np
from scipy import linalg
from skimage.util import dtype

def rgb2hed(rgb):
    return separate_stains(rgb, hed_from_rgb)


# Hematoxylin + Eosin + DAB
# rgb_from_hed = np.array([[0.650, 0.704, 0.286],
#                          [0.072, 0.990 0.105],
#                          [0.268, 0.570, 0.776]])
# hed_from_rgb = linalg.inv(rgb_from_hed)

# Hematoxylin + Eosin
rgb_from_hed = np.array([[0.650, 0.704, 0.286],
                         [0.072, 0.990, 0.105],
                         [0.0, 0.0, 0.0]])
rgb_from_hed[2, :] = np.cross(rgb_from_hed[0, :], rgb_from_hed[1, :])
hed_from_rgb = linalg.inv(rgb_from_hed)

# Hematoxylin + DAB
rgb_from_hdx = np.array([[0.650, 0.704, 0.286],
                         [0.268, 0.570, 0.776],
                         [0.0, 0.0, 0.0]])
rgb_from_hdx[2, :] = np.cross(rgb_from_hdx[0, :], rgb_from_hdx[1, :])
hdx_from_rgb = linalg.inv(rgb_from_hdx)

# Hematoxylin + AEC
rgb_from_hax = np.array([[0.650, 0.704, 0.286],
                         [0.2743, 0.6796, 0.6803],
                         [0.0, 0.0, 0.0]])
rgb_from_hax[2, :] = np.cross(rgb_from_hax[0, :], rgb_from_hax[1, :])
hax_from_rgb = linalg.inv(rgb_from_hax)

# Hematoxylin + PAS
rgb_from_hpx = np.array([[0.644211, 0.716556, 0.266844],
                         [0.175411, 0.972178, 0.154589],
                         [0.0, 0.0, 0.0]])
rgb_from_hpx[2, :] = np.cross(rgb_from_hpx[0, :], rgb_from_hpx[1, :])
hpx_from_rgb = linalg.inv(rgb_from_hpx)

def separate_stains(rgb, conv_matrix):
    # Examples
    # --------
    # >>> from skimage import data
    # >>> from skimage.color import separate_stains, hdx_from_rgb
    # >>> ihc = data.immunohistochemistry()
    # >>> ihc_hdx = separate_stains(ihc, hdx_from_rgb)
    rgb = _prepare_colorarray(rgb, force_copy=True)
    np.maximum(rgb, 1E-6, out=rgb)  # avoiding log artifacts
    log_adjust = np.log(1E-6)  # used to compensate the sum above

    stains = (np.log(rgb) / log_adjust) @ conv_matrix

    return stains

def _prepare_colorarray(arr, force_copy=False):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.shape[-1] != 3:
        raise ValueError("Input array must have a shape == (..., 3)), "
                         f"got {arr.shape}")

    return dtype.img_as_float(arr, force_copy=force_copy)


