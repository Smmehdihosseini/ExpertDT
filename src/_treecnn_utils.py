import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def crops2matrix(slide, crop_obj, label2refine, mask=None, flag_to_crop_index=None, verbose=""):
    """
    Map crops argmax prediction to matrix super-pixels.
    Key XY is added to each element in crop_obj. Such key univocally maps matrix coordiantes into crops.
    """
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
        size = slide.level_dimensions[0]
        # Slide dimensions of given level reported to level 0
        region_lv0 = (0, 0, size[0], size[1])
    crop_size = np.floor(crop_obj.indexes[0]['size'] / slide.level_downsamples[0])
    region_lv0 = [int(x) for x in np.floor(np.array(region_lv0) / crop_size)]
    n_crops = region_lv0[2] * region_lv0[3]
    if n_crops != len(crop_obj):
        import warnings
        warnings.warn("Shape error. Check mask.")
        # raise ValueError("Shape of matrix not coherent with len(crop_obj)")
    output = np.ones((region_lv0[3], region_lv0[2])) * -1
    if mask is None:
        mask = np.ones_like(output)
    else:
        assert mask.shape == output.shape, "Shape error. Check mask."

    # Each crop is mapped into x, y super-pixel in mask
    for j, crop in enumerate(crop_obj.indexes):
        x, y = np.unravel_index(j, output.shape, 'C')
        if f'{label2refine}_argmax' in crop and mask[x][y] == 1:
            output[x, y] = np.argmax(crop[f'{label2refine}_prediction'])
            if flag_to_crop_index is not None:
                # Following lines will affect crop object.
                # This is to plot heatmap governed by refined masks
                crop['keep'] = True
                if flag_to_crop_index == 0:
                    crop[f'{label2refine}_prediction'] = np.array([1.0, 0.0])
                else:
                    crop[f'{label2refine}_prediction'] = np.array([0.0, 1.0])

    # if never_found:
    #     print(f"{label2refine} never found in crop_obj")
    # Optional plot
    if len(verbose) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        ax.imshow(output, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(label2refine)
        fig.savefig(os.path.join(verbose, label2refine + '-crop2matrix.png'))
    return output


def _refine(input_image, verbose="", size=(3, 3), label2refine=""):

    def _most_frequent(x):
        central = x[x.size // 2]
        values, counts = np.unique(x, return_counts=True)
        max_freq = counts.max()
        modes = values[counts == max_freq]
        if central in modes:
            return central
        else:
            return modes[0]

    refined = ndimage.generic_filter(input_image, _most_frequent, size=size)
    if len(verbose) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        ax.imshow(refined, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(label2refine)
        fig.savefig(os.path.join(verbose, label2refine + '-refined.png'))
    return refined


def _make_masks(slide, crop_obj, label, size=(5, 5), mask=None, do_refine=False, verbose=""):
    superpixels = crops2matrix(slide, crop_obj, label, mask=mask, verbose=verbose)
    if do_refine:
        superpixels_refined = _refine(superpixels, size=size, verbose=verbose, label2refine=label)
    else:
        superpixels_refined = superpixels
    mask_0 = np.zeros_like(superpixels)
    mask_1 = np.zeros_like(superpixels)
    mask_0[superpixels_refined == 0] = 1
    mask_1[superpixels_refined == 1] = 1
    return mask_0, mask_1


def predict_by_mask(slide,
                    crop_obj,
                    size=(5, 5),
                    selective_refine=(False, False, False),
                    do_refine_not_tree=False,
                    leaf_given_cancer=None,
                    ignore_root=False,
                    leaf_given_node_by_patient=False,
                    leaf_given_node_by_patient_forced=False,
                    is_tree=True,
                    verbose=""):
    if is_tree:
        y_true = crop_obj.indexes[0]['slide_label']
        # Root: Cancer Binary
        mask_cancer, mask_not_cancer = _make_masks(slide, crop_obj, "CancerBinary", size=size,
                                                   mask=None,
                                                   do_refine=selective_refine[0],
                                                   verbose=verbose)
        # Node
        mask_node_CO, mask_node_ccp = _make_masks(slide, crop_obj, "Node", size=size,
                                                  mask=mask_cancer,
                                                  do_refine=selective_refine[1],
                                                  verbose=verbose)
        if leaf_given_cancer:
            mask = mask_cancer
        else:
            mask = mask_node_ccp
        if ignore_root:
            mask = np.ones_like(mask_cancer)

        # Leaf1
        mask_leaf1_cc_given_CO, mask_leaf1_p_given_CO = _make_masks(slide,
                                                                    crop_obj,
                                                                    "Leaf1",
                                                                    mask=mask,
                                                                    verbose="")

        mask_leaf1_cc_given_ccp, mask_leaf1_p_given_ccp = _make_masks(slide,
                                                                      crop_obj,
                                                                      "Leaf1",
                                                                      size=size,
                                                                      mask=mask,
                                                                      do_refine=selective_refine[2],
                                                                      verbose=verbose)
        # Leaf2
        if leaf_given_cancer:
            mask = mask_cancer
        else:
            mask = mask_node_CO
        if ignore_root:
            mask = np.ones_like(mask_cancer)
        mask_leaf2_C_given_CO, mask_leaf2_O_given_CO = _make_masks(slide,
                                                                   crop_obj,
                                                                   "Leaf2",
                                                                   size=size,
                                                                   mask=mask,
                                                                   do_refine=selective_refine[2],
                                                                   verbose=verbose)

        mask_leaf2_C_given_ccp, mask_leaf2_O_given_ccp = _make_masks(slide,
                                                                     crop_obj,
                                                                     "Leaf2",
                                                                     mask=mask,
                                                                     verbose="")

        leaves_counter = dict()
        leaves_counter['ccRCC'] = np.sum(mask_leaf1_cc_given_ccp, axis=(0, 1))
        leaves_counter['pRCC'] = np.sum(mask_leaf1_p_given_ccp, axis=(0, 1))
        leaves_counter['CHROMO'] = np.sum(mask_leaf2_C_given_CO, axis=(0, 1))
        leaves_counter['ONCO'] = np.sum(mask_leaf2_O_given_CO, axis=(0, 1))

        if leaf_given_node_by_patient:
            # Root decides if patient is cc+p or O+C
            # If so, all crops belonging to opposite leaf are set to zero.
            print("leaf_given_node_by_patient set to True")
            if np.sum(mask_node_CO, axis=(0, 1)) < np.sum(mask_node_ccp, axis=(0, 1)):
                leaves_counter['CHROMO'] = 0
                leaves_counter['ONCO'] = 0
            elif np.sum(mask_node_CO, axis=(0, 1)) > np.sum(mask_node_ccp, axis=(0, 1)):
                leaves_counter['ccRCC'] = 0
                leaves_counter['pRCC'] = 0
            else:
                print(crop_obj.indexes[0]['filepath_slide'])
                print("NB: found {} crops predicted as ccRCC+pRCC same as ONCO+CHROMO. "
                      "Root label assigned to ccRCC+pRCC".format(np.sum(mask_node_ccp, axis=(0, 1))))
                leaves_counter['CHROMO'] = 0
                leaves_counter['ONCO'] = 0

        if leaf_given_node_by_patient_forced:
            # Root decides if patient is cc+p or O+C
            # If so, all crops are fed to the winner leaf.
            if np.sum(mask_node_CO, axis=(0, 1)) < np.sum(mask_node_ccp, axis=(0, 1)):
                leaves_counter['ccRCC'] = np.sum(mask_leaf1_cc_given_ccp, axis=(0, 1)) + \
                                          np.sum(mask_leaf1_cc_given_CO, axis=(0, 1)) - \
                                          np.sum(np.logical_and(mask_leaf1_cc_given_ccp,
                                                                mask_leaf1_cc_given_CO), axis=(0, 1))
                leaves_counter['pRCC'] = \
                    np.sum(mask_leaf1_p_given_ccp, axis=(0, 1)) + \
                    np.sum(mask_leaf1_p_given_CO, axis=(0, 1)) - \
                    np.sum(np.logical_and(mask_leaf1_p_given_ccp,
                                          mask_leaf1_p_given_CO), axis=(0, 1))
                leaves_counter['CHROMO'] = 0
                leaves_counter['ONCO'] = 0
            elif np.sum(mask_node_CO, axis=(0, 1)) > np.sum(mask_node_ccp, axis=(0, 1)):
                if np.sum(mask_node_CO, axis=(0, 1)) < np.sum(mask_node_ccp, axis=(0, 1)):
                    leaves_counter['CHROMO'] = np.sum(mask_leaf2_C_given_CO, axis=(0, 1)) + \
                                               np.sum(mask_leaf2_C_given_ccp, axis=(0, 1)) - \
                                               np.sum(np.logical_and(mask_leaf2_C_given_CO,
                                                                     mask_leaf2_C_given_ccp), axis=(0, 1))
                    leaves_counter['ONCO'] = \
                        np.sum(mask_leaf2_O_given_CO, axis=(0, 1)) + \
                        np.sum(mask_leaf2_O_given_ccp, axis=(0, 1)) - \
                        np.sum(np.logical_and(mask_leaf2_O_given_ccp,
                                              mask_leaf2_O_given_ccp), axis=(0, 1))
                    leaves_counter['ccRCC'] = 0
                    leaves_counter['pRCC'] = 0
            else:
                print(crop_obj.indexes[0]['filepath_slide'])
                print("NB: found {} crops predicted as ccRCC+pRCC same as ONCO+CHROMO. "
                      "Root label assigned to ccRCC+pRCC".format(np.sum(mask_node_ccp, axis=(0, 1))))
                leaves_counter['CHROMO'] = 0
                leaves_counter['ONCO'] = 0

        # make prediction
        pred = max(leaves_counter,
                   key=leaves_counter.get) if max(leaves_counter,
                                                  key=leaves_counter.get) != 'ONCO' else 'ONCOCYTOMA'

        # Save masks in crop_obj
        crop_obj.indexes[0]['mask_root'] = []
        crop_obj.indexes[0]['mask_node'] = []
        crop_obj.indexes[0]['mask_leaf1'] = []
        crop_obj.indexes[0]['mask_leaf2'] = []
        crop_obj.indexes[0]['mask_root'].append(mask_cancer)
        crop_obj.indexes[0]['mask_root'].append(mask_not_cancer)
        crop_obj.indexes[0]['mask_node'].append(mask_node_CO)
        crop_obj.indexes[0]['mask_node'].append(mask_node_ccp)
        crop_obj.indexes[0]['mask_leaf1'].append(mask_leaf1_cc_given_ccp)
        crop_obj.indexes[0]['mask_leaf1'].append(mask_leaf1_p_given_ccp)
        crop_obj.indexes[0]['mask_leaf2'].append(mask_leaf2_C_given_CO)
        crop_obj.indexes[0]['mask_leaf2'].append(mask_leaf2_O_given_CO)

        return pred, y_true, leaves_counter
    else:

        superpixels = crops2matrix(slide, crop_obj, "NotCancerAllTumors", mask=None, verbose=verbose)
        if do_refine_not_tree:
            superpixels_refined = _refine(superpixels, size=size, verbose=verbose)
        else:
            superpixels_refined = superpixels

        leaves_counter = dict()
        labels_categorical = ['CHROMO', 'ONCO', 'ccRCC', 'not_cancer', 'pRCC']
        for j in range(5):
            mask = np.zeros_like(superpixels)
            mask[superpixels_refined == j] = 1

            leaves_counter[labels_categorical[j]] = np.sum(mask, axis=(0, 1))

        pred = max(leaves_counter,
                   key=leaves_counter.get) if max(leaves_counter,
                                                  key=leaves_counter.get) != 'ONCO' else 'ONCOCYTOMA'

        y_true = crop_obj.indexes[0]['slide_label']

        return pred, y_true, leaves_counter
