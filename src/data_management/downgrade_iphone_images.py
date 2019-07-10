"""
File Name: downgrade_iphone_images.py

Authors: Kyle Seidenthal

Date: 09-07-2019

Description: A module containing functionality to match the high quality IPhone images with the quality of the GroPro
images

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import io
from skimage import data, exposure, img_as_float
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import rescale

REFERENCE_PATH = "./GrowPro_Annotations/cropped_images/5/G0016261.JPG"
IMAGE_PATH = "./IPhone_Annotations/cropped_images/5/IMG_4436.JPG"




def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array

    src_quantiles = np.cumsum(src_counts).astype(np.float64) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float64) / template.size


    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)



def match_histograms(image, reference, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape
    image_dtype = image.dtype

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched

ref_img = io.imread(REFERENCE_PATH)
img = io.imread(IMAGE_PATH)

scaling_factor = 0.074


img = rescale(img, scaling_factor)

img = rgb2hsv(img)
ref_img = rgb2hsv(ref_img)

ref_img_hist_R = exposure.histogram(ref_img[:, :, 0])
ref_img_hist_G = exposure.histogram(ref_img[:, :, 1])
ref_img_hist_B = exposure.histogram(ref_img[:, :, 2])

img_hist_R = exposure.histogram(img[:, :, 0])
img_hist_G = exposure.histogram(img[:, :, 1])
img_hist_B = exposure.histogram(img[:, :, 2])

ref_img_cdf_R = exposure.cumulative_distribution(ref_img[:, :, 0])
ref_img_cdf_G = exposure.cumulative_distribution(ref_img[:, :, 1])
ref_img_cdf_B = exposure.cumulative_distribution(ref_img[:, :, 2])

img_cdf_R = exposure.cumulative_distribution(img[:, :, 0])
img_cdf_G = exposure.cumulative_distribution(img[:, :, 1])
img_cdf_B = exposure.cumulative_distribution(img[:, :, 2])

matched = match_histograms(img, ref_img, multichannel=True)

matched_hist_R = exposure.histogram(matched[:, :, 0])
matched_hist_G = exposure.histogram(matched[:, :, 1])
matched_hist_B = exposure.histogram(matched[:, :, 2])

matched_cdf_R = exposure.cumulative_distribution(matched[:, :, 0])
matched_cdf_G = exposure.cumulative_distribution(matched[:, :, 1])
matched_cdf_B = exposure.cumulative_distribution(matched[:, :, 2])


f, axes = plt.subplots(4, 3)
axes[0, 0].imshow(hsv2rgb(img))
axes[0, 0].set_title("Source")
axes[0, 1].imshow(hsv2rgb(ref_img))
axes[0, 1].set_title("Target")
axes[0, 2].imshow(hsv2rgb(matched))
axes[0, 2].set_title("Matched")

axes[1, 0].plot(img_hist_R[1], img_hist_R[0]/img_hist_R[0].max(), lw=2)
axes[1, 0].plot(img_cdf_R[1], img_cdf_R[0], 'r')

axes[1, 1].plot(ref_img_hist_R[1], ref_img_hist_R[0]/ref_img_hist_R[0].max(), lw=2)
axes[1, 1].plot(ref_img_cdf_R[1], ref_img_cdf_R[0], 'r')

axes[1, 2].plot(matched_hist_R[1], matched_hist_R[0]/matched_hist_R[0].max(), lw=2)
axes[1, 2].plot(matched_cdf_R[1], matched_cdf_R[0], 'r')

axes[2, 0].plot(img_hist_G[1], img_hist_G[0]/img_hist_G[0].max(), lw=2)
axes[2, 0].plot(img_cdf_G[1], img_cdf_G[0], 'r')

axes[2, 1].plot(ref_img_hist_G[1], ref_img_hist_G[0]/ref_img_hist_G[0].max(), lw=2)
axes[2, 1].plot(ref_img_cdf_G[1], ref_img_cdf_G[0], 'r')

axes[2, 2].plot(matched_hist_G[1], matched_hist_G[0]/matched_hist_G[0].max(), lw=2)
axes[2, 2].plot(matched_cdf_G[1], matched_cdf_G[0], 'r')

axes[3, 0].plot(img_hist_B[1], img_hist_B[0]/img_hist_B[0].max(), lw=2)
axes[3, 0].plot(img_cdf_B[1], img_cdf_B[0], 'r')

axes[3, 1].plot(ref_img_hist_B[1], ref_img_hist_B[0]/ref_img_hist_B[0].max(), lw=2)
axes[3, 1].plot(ref_img_cdf_B[1], ref_img_cdf_B[0], 'r')

axes[3, 2].plot(matched_hist_B[1], matched_hist_B[0]/matched_hist_B[0].max(), lw=2)
axes[3, 2].plot(matched_cdf_B[1], matched_cdf_B[0], 'r')



plt.show()



# TODO: Match  each of H, S, and V channels


