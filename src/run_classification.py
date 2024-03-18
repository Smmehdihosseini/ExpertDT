import os
import math
import json
import click
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from data_manager import DataManager
from vgg16 import Vgg16
from utils import plot_confusion_matrix, output_message


def _scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1


def _define_optimizer(optimizer_name, learning_rate):
    def _optimizer_error():
        raise ValueError("Invalid optimizer.")

    optimizers = {
                  "Adam": tf.keras.optimizers.Adam(learning_rate=learning_rate)
                  }
    optimizer = optimizers.get(optimizer_name, None)
    if optimizer is None:
        _optimizer_error()
    return optimizer


def prepare_model(my_model, first_trained_layer, learning_rate, optimizer_name, loss):
    """
    Compile and fit keras model.
    @param my_model: AbstractCnn object
    @param my_model:
    @param first_trained_layer: first trained layer (if fine tuning)
    @param learning_rate: learning_rate
    @param optimizer_name: optimizer
    @param loss: loss
    @return: AbstractCnn object compiled.
    """
    my_model.init(first_trained_layer=first_trained_layer)
    opt = _define_optimizer(optimizer_name, learning_rate)
    my_model.model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy'])
    return my_model


@click.command()
@click.argument("filepath_dataframe_train",
                type=click.Path(exists=True,
                                file_okay=True))
@click.argument("filepath_dataframe_test",
                type=click.Path(exists=True,
                                file_okay=True))
@click.argument("dirpath_model",
                type=click.Path(exists=False,
                                file_okay=False))
@click.argument('x_col', type=str)
@click.argument('y_col', type=str)
@click.option('--cnn_model', type=str, default="Vgg16")
@click.option('--training', type=bool, default=True)
@click.option('--testing', type=bool, default=True)
@click.option('--image_shape_original', nargs=3, default=(2000, 2000, 3), type=str)
@click.option('--image_shape_resized', nargs=2, default=(112, 112), type=str)
@click.option('--patch_size', nargs=2, default=(1000, 1000), type=str)
@click.option('--batch_size', default=128, type=int)
@click.option('--epochs', default=50, type=int)
@click.option('--learning_rate', default=0.0001, type=float)
@click.option('--optimizer', default="Adam", type=str)
@click.option('--loss', default="categorical_crossentropy", type=str)
@click.option('--val_split', default=0.2, type=float)
@click.option('--first_trained_layer', default=11, type=int)
@click.option('--label_coding', default="one-hot", type=str)
@click.option('--preprocessing', default=None)
@click.option('--augment', default=True, type=bool)
@click.option('--buffer_size', default=None, type=int)
@click.option('--test_reduced', default=False, type=bool)
@click.option('--cuda_device', default='0', type=str)
def main(filepath_dataframe_train,
         filepath_dataframe_test,
         dirpath_model,
         x_col,
         y_col,
         cnn_model,
         training,
         testing,
         image_shape_original,
         image_shape_resized,
         patch_size,
         batch_size,
         epochs,
         learning_rate,
         optimizer,
         loss,
         val_split,
         first_trained_layer,
         label_coding,
         preprocessing,
         augment,
         buffer_size,
         test_reduced,
         cuda_device):

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    # Set first_trained_layer to None if is < 0
    if first_trained_layer < 0:
        first_trained_layer = None

    # Convert to int
    image_shape_original = (int(image_shape_original[0]),
                            int(image_shape_original[1]),
                            int(image_shape_original[2]))
    image_shape_resized = (int(image_shape_resized[0]),
                           int(image_shape_resized[1]))
    patch_size = (int(patch_size[0]), int(patch_size[1]))

    # Dir to save some images as example
    if not os.path.exists(dirpath_model):
        os.makedirs(dirpath_model)

    # Save params
    parameters = locals()
    filepath_parameters = os.path.join(dirpath_model, "parameters.json")
    with open(filepath_parameters, 'w') as fp:
        json.dump(parameters, fp)

    dir_path_save_images = os.path.join(dirpath_model, "examples_images")

    def _step_decay(epoch, lr):
        drop = 0.5
        epochs_drop = 10.0
        lr = learning_rate * math.pow(drop,
                                      math.floor((1 + epoch) / epochs_drop))
        return lr

    # Preprocessing depends on fine tuning
    if first_trained_layer:
        _functions = {
                      "Vgg16": "tf.keras.applications.vgg16.preprocess_input"
                    }
        preprocessing = _functions[cnn_model]
    else:
        preprocessing = None

    # Make train and val datasets >>>>
    dataframe_train = pd.read_csv(filepath_dataframe_train)
    data_manager_train = DataManager(dataframe=dataframe_train,
                                     x_col=x_col,
                                     y_col=y_col,
                                     image_shape_original=image_shape_original,
                                     batch_size=batch_size,
                                     label_coding=label_coding,
                                     image_shape_resized=image_shape_resized,
                                     patch_size=patch_size,
                                     val_split=val_split,
                                     preprocessing=preprocessing,
                                     augment=augment,
                                     buffer_size=buffer_size,
                                     is_test_ds=False,
                                     dir_path_save_images=dir_path_save_images)
    train_ds = data_manager_train.ds_major
    val_ds = data_manager_train.ds_minor
    # <<<<

    # Get number of classes
    n_classes = len(set(dataframe_train[y_col]))
    output_message("Found {} classes in {}".format(n_classes, filepath_dataframe_train))

    # Make test dataset >>>>
    dataframe_test = pd.read_csv(filepath_dataframe_test)

    # Drop row corresponding to possible NaN values in y_col
    dataframe_test = dataframe_test[dataframe_test[y_col].notna()]
    if test_reduced:
        print("Reduced test dataset")
        dataframe_test = dataframe_test.sample(frac=0.15)

    data_manager_test = DataManager(dataframe=dataframe_test,
                                    x_col=x_col,
                                    y_col=y_col,
                                    image_shape_original=image_shape_original,
                                    batch_size=batch_size,
                                    label_coding=label_coding,
                                    image_shape_resized=image_shape_resized,
                                    patch_size=patch_size,
                                    val_split=0,
                                    preprocessing=preprocessing,
                                    augment=False,
                                    is_test_ds=True)
    test_ds = data_manager_test.ds_major
    y_test = data_manager_test.labels_numerical
    # <<<<
    color_channels = image_shape_original[-1]
    cnn_dict = {"Vgg16": Vgg16(n_classes=n_classes,
                               input_shape=(image_shape_resized[0],
                                            image_shape_resized[1],
                                            color_channels))
                }
    cnn = prepare_model(cnn_dict[cnn_model],
                        learning_rate=learning_rate,
                        first_trained_layer=first_trained_layer,
                        optimizer_name=optimizer,
                        loss=loss)

    # Callbacks >>>>
    # Set quantity for saving model
    if val_split != 0:
        monitor = "val_accuracy"
    else:
        monitor = "accuracy"
    filepath_model = os.path.join(dirpath_model,  "weights.hdf5")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath_model, monitor=monitor, verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto')

    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="auto",
                                                   patience=25, min_delta=0.005)

    lrs_callback = tf.keras.callbacks.LearningRateScheduler(_scheduler)

    sd_callback = tf.keras.callbacks.LearningRateScheduler(_step_decay)

    rop_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.1, patience=10, verbose=1, mode='auto',
        min_delta=0.005, cooldown=0, min_lr=learning_rate*0.0001)
    if training:
        history = cnn.model.fit(
            train_ds,
            validation_data=val_ds,
            validation_steps=int(tf.data.experimental.cardinality(val_ds).numpy()),
            epochs=epochs,
            callbacks=[cp_callback, es_callback, rop_callback]
        ).history

        # Save tran/val curves
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        filepath_plot = os.path.join(dirpath_model, "training.png")
        ax[0].set_title("Accuracy")
        epochs_actual = len(history['accuracy'])
        ax[0].plot([e for e in range(1, epochs_actual + 1)], history['accuracy'], label="train")
        ax[0].plot([e for e in range(1, epochs_actual + 1)], history['val_accuracy'], label="val")
        ax[0].set_xticks([e for e in range(1, epochs_actual + 1)])
        ax[0].legend()
        ax[1].set_title("Loss")
        ax[1].plot([e for e in range(1, epochs_actual + 1)], history['loss'], label="train")
        ax[1].plot([e for e in range(1, epochs_actual + 1)], history['val_loss'], label="val")
        ax[1].set_xticks([e for e in range(1, epochs_actual + 1)])
        ax[1].legend()
        fig.savefig(filepath_plot)

        # Save history
        filepath_history = os.path.join(dirpath_model, "history.pickle")
        with open(filepath_history, 'wb') as fp:
            pickle.dump(history, fp)
    # <<<<

    # Test from saved model >>>>
    # Load model
    if testing:
        cnn = prepare_model(cnn_dict[cnn_model],
                            learning_rate=learning_rate,
                            first_trained_layer=first_trained_layer,
                            optimizer_name=optimizer,
                            loss=loss)
        cnn.model.load_weights(filepath_model)

        # Predictions
        output_message("Doing predictions")
        predictions = cnn.model.predict(test_ds, batch_size=2000, verbose=1)

        # Save predictions and labels
        filepath_predictions = os.path.join(dirpath_model, "predictions.npy")
        filepath_y_test = os.path.join(dirpath_model, "y_test.npy")
        np.save(filepath_predictions, predictions)
        np.save(filepath_y_test, y_test)

        # Save confusion matrix
        fig, ax = plt.subplots(1, 1)
        filepath_plot = os.path.join(dirpath_model, "cfm.png")
        plot_confusion_matrix(y_test,
                              np.argmax(predictions, axis=1),
                              ax,
                              data_manager_test.labels_univocal,
                              filepath_plot=filepath_plot)
    # <<<<
    print("Done")


if __name__ == '__main__':
    main()
