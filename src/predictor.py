import os
import json
import click
import pickle
import numpy as np
import skimage.io as io
from skimage.transform import resize
from make_heatmaps import _is_background
from _segmentation_cnn import SegmnentationNet


def _grayscale_to_bw(img, threshold=0.2):
    img[img > threshold] = 1
    img[img <= threshold] = 0
    return img


def save_result(save_path, prediction, original, filename, uint8=True, _resize=None):
    if len(prediction.shape) == 3:
        prediction = np.squeeze(prediction, axis=-1)
    img = _grayscale_to_bw(prediction)
    if uint8:
        img = img * 255
        img = img.astype(np.uint8)
    if resize:
        img_resized = resize(img, (_resize, _resize),
                             anti_aliasing=True)
        img_resized = img_resized * 255
        img_resized = img_resized.astype(np.uint8)
    else:
        img_resized = img
    filepath = os.path.join(save_path, filename)
    io.imsave(filepath, img_resized)
    if resize:
        img_resized = resize(original, (_resize, _resize),
                             anti_aliasing=True)
        img_resized = img_resized * 255
        img_resized = img_resized.astype(np.uint8)
    else:
        img_resized = img
    filename = filename.split(".")[0] + "-orignal.png"
    filepath = os.path.join(save_path, filename)
    io.imsave(filepath, img_resized)


@click.command()
@click.argument('crop_obj_file', type=click.Path(exists=False, file_okay=True), default=None)
@click.argument("parameters_file", type=click.Path(exists=False, file_okay=True), default=None)
@click.argument('output_folder', type=click.Path(exists=False, file_okay=False), default=None)
@click.option('--device', type=str, default="0")
@click.option('--save_images', type=bool, default=False)
def main(crop_obj_file,
         parameters_file,
         output_folder,
         device,
         save_images):

    # Load crop obj
    with open(crop_obj_file, "rb") as handle:
        crop_obj = pickle.load(handle)
    print("qui")
    # Load model
    with open(parameters_file) as f:
        params = json.load(f)

    net = SegmnentationNet(input_shape=params["input_shape"],
                           initial_learning_rate=1e-3,
                           loss=params["loss"],
                           model_type=params["model_type"],
                           backbone=params["backbone"],
                           computing_device=device,
                           encoder_freeze=params["encoder_freeze"],
                           model_filename="segmentation_network_ccRCC+pRCC.h5",
                           output_folder=output_folder)

    net.model.load_weights(os.path.join(params["dir_experiment"],
                                        params["model_name"]))
    j = 0
    for image, index in zip(crop_obj, crop_obj.indexes):
        if not _is_background(image):
            mask = np.squeeze(net.model.predict(np.array([image]) / 255), axis=0)
            index['mask'] = mask

            if save_images:
                save_result(output_folder,
                            mask,
                            image,
                            "{}-{}.png".format(crop_obj_file.split('.')[0],
                                               j), _resize=500)
            j += 1

    # Save the crop obj
    with open(crop_obj_file, "wb") as handle:
        pickle.dump(crop_obj, handle)


if __name__ == '__main__':
    main()
