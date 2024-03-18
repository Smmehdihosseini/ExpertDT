import os
import json
import click
import numpy as np
import pandas as pd
import tensorflow as tf
import skimage.io as io
from skimage.transform import resize
from _segmentation_cnn import SegmnentationNet
# from models._model_factory import get_network_by_name


def grayscale_to_bw(img, threshold=0.2):
    img[img > threshold] = 1
    img[img <= threshold] = 0
    return img


def save_result(save_path, file_npy, filepaths_test, uint8=True, _resize=None):
    filepaths_predictions = list()
    for i, item in enumerate(file_npy):
        if len(item.shape) == 3:
            item = np.squeeze(item, axis=-1)
        img = grayscale_to_bw(item)
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
        filename = "{}_predict.png".format(i)
        filepath = os.path.join(save_path, filename)
        filepaths_predictions.append(filepath)
        io.imsave(filepath, img_resized)

    # Save dataframe for image - segmentation correspondence
    df = pd.DataFrame(list(zip(filepaths_test, filepaths_predictions)),
                      columns=['Original Image', 'Predicted Mask'])
    df.to_csv(os.path.join(save_path, "image-segmentation.csv"))


@click.command()
@click.option("--dir_experiment", type=click.Path(exists=False, file_okay=False), default=None)
@click.option('--prediction_folder_name', type=str, default="predictions")
@click.option('--model_type', type=str, default="Unet")
@click.option('--loss', type=str, default="binary_crossentropy")
@click.option('--backbone', type=str, default="resnet34")
@click.option('--encoder_freeze', type=str, default="True")
@click.option('--model_name', type=str, default="model.h5")
@click.option('--epochs', type=int, default=10)
@click.option('--batch_size', type=int, default=2)
@click.option('--dataframe_images_train', type=click.Path(exists=True, file_okay=True), default=None)
@click.option('--dataframe_masks_train', type=click.Path(exists=True, file_okay=True), default=None)
@click.option('--do_training', type=bool, default=True)
@click.option('--steps_by_epoch', type=int, default=10)
@click.option('--computing_device', type=str, default="0")
@click.option('--dataframe_test', type=click.Path(exists=True, file_okay=True), default=None)
@click.option('--input_shape', type=str, default="224 224 3")
def main(dir_experiment,
         prediction_folder_name,
         model_type,
         loss,
         backbone,
         encoder_freeze,
         model_name,
         epochs,
         batch_size,
         dataframe_images_train,
         dataframe_masks_train,
         do_training,
         steps_by_epoch,
         computing_device,
         dataframe_test,
         input_shape):

    # Convert encoder_freeze parameter to boolean
    encoder_freeze = True if encoder_freeze == 'True' else False

    parameters = locals()
    file_path = os.path.join(dir_experiment, 'parameters.json')
    with open(file_path, 'w') as fp:
        json.dump(parameters, fp)

    def init_net():
        _net = SegmnentationNet(input_shape=_input_shape,
                                initial_learning_rate=1e-3,
                                loss=loss,
                                model_type=model_type,
                                backbone=backbone,
                                computing_device=computing_device,
                                encoder_freeze=encoder_freeze,
                                model_filename=model_name,
                                output_folder=dir_experiment)
        return _net
    if dataframe_test == 'None':
        dataframe_test = None
    _input_shape = (int(input_shape.split(' ')[0]),
                    int(input_shape.split(' ')[1]),
                    int(input_shape.split(' ')[2]))
    if do_training:
        net = init_net()
        # Start training >>>>
        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')

        net.train(epochs,
                  batch_size,
                  pd.read_csv(dataframe_images_train),
                  pd.read_csv(dataframe_masks_train),
                  "path",
                  steps_per_epoch=steps_by_epoch,
                  aug_dict=data_gen_args,
                  aug_dir=None)
    else:
        print("Training skipped.")
    # <<<<

    # Test >>>>
    print("Testing: {}".format(dir_experiment))
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    if not os.path.exists(os.path.join(dir_experiment, prediction_folder_name)):
        os.makedirs(os.path.join(dir_experiment, prediction_folder_name))

    if dataframe_test:
        net = init_net()
        net.model.load_weights(os.path.join(dir_experiment, model_name))
        test_generator = test_datagen.flow_from_dataframe(pd.read_csv(dataframe_test),
                                                          directory=None,
                                                          x_col="path",
                                                          target_size=(int(input_shape.split(' ')[0]),
                                                                       int(input_shape.split(' ')[1])),
                                                          color_mode="rgb",
                                                          shuffle=False,
                                                          class_mode=None,
                                                          batch_size=1)
        nb_samples = len(test_generator.filenames)

        predict = net.model.predict_generator(test_generator, steps=nb_samples)
        save_result(os.path.join(dir_experiment, prediction_folder_name),
                    predict,
                    test_generator.filenames,
                    _resize=2000)
        np.save(os.path.join(dir_experiment, "predict.npy"), predict)
    else:
        print("Test skipped")
    # <<<<


if __name__ == '__main__':
    main()
