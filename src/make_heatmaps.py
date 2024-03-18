import os
import math
import copy
import itertools
import numpy as np
from matplotlib import cm
from PIL import Image, ImageFilter
from _treecnn_utils import crops2matrix
from wsi_manager import SectionManager, CropList

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'


def _is_background(image_array):
    # White crops are removed at runtime.
    mean = np.mean(image_array)

    if mean > 210:
        return True
    else:
        return False


def _batch_iterator(iterable_images, iterable_labels, batch_size=1):
    """
    Batch iterator
    :param iterable_images:
    :param iterable_labels:
    :param batch_size:
    :return:
    """
    iterable_images = iter(iterable_images)
    iterable_labels = iter(iterable_labels)

    while True:
        batch_images = list(itertools.islice(iterable_images, batch_size))
        batch2_labels = list(itertools.islice(iterable_labels, batch_size))

        if len(batch_images) > 0:
            yield batch_images, batch2_labels
        else:
            break


def _crops2slide(slide, crop_obj, channel, mapping_key, colormap=cm.get_cmap('Blues'), lv=-1):
    """
    Builds a 3 channel map.
    The first three channels represent the sum of all the probabilities of the crops which contain that pixel
    belonging to classes 0-1, the fourth hold the number of crops which contain it.
    :param slide: OpenSlide object
    :param crop_obj: CropList object
    :param lv: Slide level
    :param mapping_key: Key mapped into heatmaps
    :return:
    """

    level_downsample = 1 / slide.level_downsamples[lv]

    if 'openslide.bounds-width' in slide.properties.keys():
        # Here to consider only the rectangle bounding the non-empty region of the slide, if available.
        # This properties are in the level 0 reference frame.
        bounds_width = int(slide.properties['openslide.bounds-width'])
        bounds_height = int(slide.properties['openslide.bounds-height'])
        bounds_x = int(slide.properties['openslide.bounds-x'])
        bounds_y = int(slide.properties['openslide.bounds-y'])

        region_lv0 = (bounds_x,
                      bounds_y,
                      bounds_width,
                      bounds_height)
    else:
        # If bounding box of the non-empty region of the slide is not available
        size = slide.level_dimensions[lv]
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, size[0] / level_downsample, size[1] / level_downsample)

    region_lv0 = [round(x) for x in region_lv0]
    region_lv_selected = [round(x * level_downsample) for x in region_lv0]
    if mapping_key and channel != "slide":
        probabilities = np.zeros((region_lv_selected[3], region_lv_selected[2], 3))
        n_classes = 2
        for crop in crop_obj.indexes:
            top = math.ceil(crop['top'] * level_downsample)
            left = math.ceil(crop['left'] * level_downsample)
            side = math.ceil((crop['size'] * slide.level_downsamples[crop['level']]) * level_downsample)

            top -= region_lv_selected[1]
            left -= region_lv_selected[0]
            side_x = side
            side_y = side
            if top < 0:
                side_y += top
                top = 0
            if left < 0:
                side_x += left
                left = 0

            if side_x > 0 and side_y > 0:
                val = probabilities[top:top + side_y, left:left + side_x, 0:n_classes]
                probabilities[top:top + side_y, left:left + side_x, 0:n_classes] = val + np.array(crop[mapping_key])
                probabilities[top:top + side_y, left:left + side_x, n_classes] = \
                    probabilities[top:top + side_y, left:left + side_x, n_classes] + 1

        # Dividing the first three channels by the fourth, every channel now becomes a probability map.
        # Each value is between 0 and 1 and represents the average of all votes for that pixel.
        # The map is then converted to uint8 to be saved as an 8bpc image.
        probabilities = np.divide(probabilities[:, :, 0:n_classes], probabilities[:, :, n_classes:n_classes+1],
                                  where=probabilities[:, :, n_classes:n_classes+1] != 0)
        probabilities = probabilities * 255
        probabilities = probabilities.astype('uint8')

        map_ = probabilities[:, :, channel]
        map_ = Image.fromarray(map_).filter(ImageFilter.GaussianBlur(3))
        map_ = np.array(map_) / 255

        alpha = Image.fromarray(((map_ + 0.01) * 255 * 1.5).astype('uint8'))
        map_[map_ < 0.5] = 0

        map_ = colormap(np.array(map_))
        roi_map = Image.fromarray((map_ * 255).astype('uint8'))
        roi_map.putalpha(50)

        slide_image = slide.read_region((region_lv0[0], region_lv0[1]), lv,
                                        (region_lv_selected[2], region_lv_selected[3]))
        slide_image.alpha_composite(roi_map)
        slide_image.convert('RGBA')

        roi_map.putalpha(alpha)
        return slide_image, roi_map

    else:
        # Slide
        slide_image = slide.read_region((region_lv0[0], region_lv0[1]), lv,
                                        (region_lv_selected[2], region_lv_selected[3]))
        return False, slide_image


def make_heatmap(crop_obj,
                 slide,
                 output_dir,
                 channels,
                 colormaps,
                 mapping_key,
                 tissue_key='tissue',
                 filename_root='',
                 resolution_level=3,
                 single_heatmap=False,
                 mask=False,
                 save_original=False):
    """
    Function to make heatmaps from a CropList object elaborated yet.
    With mapping key a specific prediction may be selected to generate heatmaps on.
    """
    print("Doing heatmap(s) for {}".format(crop_obj.indexes[0]['filepath_slide']))

    # Take filtering mask from indexes[0]
    if mask:
        masks = [crop_obj.indexes[0][mask][0], crop_obj.indexes[0][mask][1]]
        crop_obj_original = copy.deepcopy(crop_obj)

    else:
        masks = None
        crop_obj_original = None

    # Make slide image for background
    if single_heatmap or save_original:
        _, slide_image_singe_heatmap = _crops2slide(slide,
                                                    crop_obj,
                                                    "slide",
                                                    lv=resolution_level,
                                                    mapping_key=None)
        slide_image_singe_heatmap.save(os.path.join(output_dir, filename_root + '_original.bmp'))
    else:
        slide_image_singe_heatmap = None

    # Make masks
    for channel, colormap in zip(channels, colormaps):
        # Optional filter on mask
        if mask:
            crop_obj = crop_obj_original
            label2refine = mapping_key.split('_')[0]
            # This change crop_obj
            crops2matrix(slide, crop_obj, label2refine, mask=masks[channel],
                         flag_to_crop_index=channel)
            crop_obj_filt = CropList(list(filter(lambda x: "keep" in x, crop_obj.indexes)))
        else:
            crop_obj_filt = crop_obj
        # Remove background crops
        crop_obj_filt = CropList(list(filter(lambda x: tissue_key in x, crop_obj_filt.indexes)))
        (slide_image, heatmap) = _crops2slide(slide,
                                              crop_obj_filt,
                                              channel,
                                              mapping_key,
                                              colormap=colormap,
                                              lv=resolution_level)
        if single_heatmap:
            slide_image_singe_heatmap.alpha_composite(heatmap)
        else:
            if len(filename_root) > 0:
                filename_root = filename_root + " "
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if slide_image:
                slide_image.save(os.path.join(output_dir, filename_root + f'{channel}.bmp'))
            filepath_final = os.path.join(output_dir, filename_root + "{}.png".format(channel))
            heatmap.save(filepath_final)

    if single_heatmap:
        slide_image_singe_heatmap.save(os.path.join(output_dir, filename_root + '_heatmap.bmp'))

