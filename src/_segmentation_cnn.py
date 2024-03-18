import os
import json
# import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from segmentation_models import Unet, FPN, Linknet, PSPNet
from segmentation_models.losses import JaccardLoss, BinaryFocalLoss, DiceLoss
from segmentation_models.utils import set_trainable


class SegmnentationNet:
    def __init__(self,
                 input_shape,
                 initial_learning_rate,
                 output_folder,
                 loss='binary_crossentropy',
                 model_type="Unet",
                 backbone="vgg16",
                 encoder_freeze=True,
                 model_filename="Seg.h5",
                 computing_device="0"):
        """
        Code from https://segmentation-models.readthedocs.io/en/latest/tutorial.html
        """
        self.input_shape = input_shape
        self.initial_learning_rate = initial_learning_rate
        self. model_folder = output_folder
        self.computing_device = computing_device
        self.model_type = model_type
        self.backbone = backbone
        self.encoder_freeze = encoder_freeze
        self.loss = loss
        # Init model file_path
        self.model_file_path = os.path.join(self.model_folder, model_filename)
        # Init methods
        self._initialize_session()
        self._build_model()

        self.epochs = None
        self.batch_size = None
        self.history = None

    def _initialize_session(self):
        """
        Init tf session.
        :return:
        """
        if self.computing_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.computing_device

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)

        # Restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init_operation = tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer())
        self.sess.run(init_operation)

    def _build_model(self):
        models = {"Unet": Unet(backbone_name=self.backbone,
                               classes=1,
                               encoder_weights='imagenet',
                               encoder_freeze=self.encoder_freeze),
                  "LinkNet": Linknet(backbone_name=self.backbone,
                                     classes=1,
                                     encoder_weights='imagenet',
                                     encoder_freeze=self.encoder_freeze)}
        losses = {"JaccardLoss": JaccardLoss(),
                  "DiceLoss": DiceLoss(),
                  "BinaryFocalLoss": BinaryFocalLoss(),
                  "binary_crossentropy": "binary_crossentropy"}

        self.model = models.get(self.model_type)
        self.model.compile('Adam',
                           losses.get(self.loss),
                           ['accuracy'])

    def _save_params(self, verbose=True):
        """
        Save CNN paremeters into a json file.
        :return:
        """
        file_path = os.path.join(self.model_folder, "params.json")
        d = dict()
        d['learning_rate'] = self.initial_learning_rate
        d['epochs'] = self.epochs
        d['input_shape'] = self.input_shape
        d['batch_size'] = self.batch_size

        if verbose:
            for k, v in d.items():
                self._print_parameter_value(k, v)

        with open(file_path, 'w') as f:
            json.dump(d, f)

    def _unet_generator_df(self,
                           batch_size,
                           dataframe_images,
                           dataframe_masks,
                           x_col,
                           aug_dict,
                           image_color_mode="grayscale",
                           mask_color_mode="grayscale",
                           image_save_prefix="image",
                           mask_save_prefix="mask",
                           save_to_dir=None,
                           class_mode=None,
                           seed=42):

        image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
        mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)

        image_generator = image_datagen.flow_from_dataframe(
            dataframe=dataframe_images,
            directory=None,
            x_col=x_col,
            subset="training",
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            class_mode=class_mode,
            target_size=(self.input_shape[0], self.input_shape[0]),
            color_mode=image_color_mode,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix)

        mask_generator = mask_datagen.flow_from_dataframe(
            dataframe=dataframe_masks,
            directory=None,
            x_col=x_col,
            subset="training",
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            class_mode=class_mode,
            color_mode=mask_color_mode,
            target_size=(self.input_shape[0], self.input_shape[0]),
            save_to_dir=save_to_dir,
            save_prefix=mask_save_prefix)

        train_generator = zip(image_generator, mask_generator)
        for (img, mask) in train_generator:
            img, mask = self.adjust_data(img, mask)
            yield (img, mask)

    @staticmethod
    def adjust_data(img, mask):
        if np.max(img) > 1:
            img = img / 255
            mask = mask / 255
            mask[mask > 0.1] = 1
            mask[mask <= 0.5] = 0
        return img, mask

    def train(self,
              epochs,
              batch_size,
              dataframe_images,
              dataframe_masks,
              x_col,
              steps_per_epoch=None,
              aug_dict=None,
              aug_dir=None):

        print("Running Unet train method")
        self.epochs = epochs
        self.batch_size = batch_size
        self._save_params()
        if steps_per_epoch is None:
            steps_per_epoch = len(dataframe_images) / batch_size
        else:
            augmentation_factor = (steps_per_epoch * batch_size) / len(dataframe_images)
            print("Data augmentation factor: {:.2f}".format(augmentation_factor))
        # Define the generator
        generator_train = self._unet_generator_df(batch_size,
                                                  dataframe_images,
                                                  dataframe_masks,
                                                  x_col,
                                                  aug_dict,
                                                  image_color_mode="rgb",
                                                  mask_color_mode="grayscale",
                                                  image_save_prefix="image",
                                                  mask_save_prefix="mask",
                                                  save_to_dir=aug_dir,
                                                  class_mode=None,
                                                  seed=42)
        self._show_images_from_generator(generator_train)
        self.history = self.model.fit_generator(generator_train,
                                                epochs=epochs,
                                                steps_per_epoch=steps_per_epoch,
                                                callbacks=[tf.keras.callbacks.ModelCheckpoint(self.model_file_path,
                                                                                           monitor="accuracy",
                                                                                           verbose=1,
                                                                                           save_best_only=True,
                                                                                           save_weights_only=False),
                                                           tf.keras.callbacks.EarlyStopping(
                                                               monitor="accuracy",
                                                               verbose=1,
                                                               min_delta=0.01,
                                                               patience=20,
                                                               mode='auto',
                                                               baseline=None,
                                                               restore_best_weights=True),
                                                           tf.keras.callbacks.ReduceLROnPlateau(
                                                               monitor="accuracy",
                                                               min_delta=0.005,
                                                               verbose=1,
                                                               factor=0.5,
                                                               patience=10,
                                                               min_lr=self.initial_learning_rate*0.01),
                                                           ]
                                                ).history
        self._log_message("Training done")
        self._log_message("Model saved in {}".format(self.model_file_path))

    def _show_images_from_generator(self, generator, fold=None):
        if fold:
            dir_examples_images = os.path.join(self.model_folder, "images_from_generator_examples", fold)
        else:
            dir_examples_images = os.path.join(self.model_folder, "images_from_generator_examples")
        if not os.path.exists(dir_examples_images):
            os.makedirs(dir_examples_images)
        masks, imgs = next(generator)
        for i, img in enumerate(imgs):
            img = np.squeeze(img)*255
            image = Image.fromarray(img.astype(np.uint8))
            file_path = os.path.join(dir_examples_images, str(i) + "_mask" + ".png")
            image.save(file_path)
        for i, img in enumerate(masks):
            img = np.squeeze(img)*255
            image = Image.fromarray(img.astype(np.uint8))
            file_path = os.path.join(dir_examples_images, str(i) + "_img" + ".png")
            image.save(file_path)

    @staticmethod
    def _log_message(string_to_print):
        print('-' * len(string_to_print))
        print(string_to_print)
        print('-' * len(string_to_print))

    @staticmethod
    def _print_parameter_value(parameter_name, parameter_value):
        """
        Print couple of parameter name and value.
        :param parameter_name:
        :param parameter_value:
        :return:
        """
        string = "{} : {}".format(parameter_name, parameter_value)
        print('-' * len(string) + "\n{}".format(string) + '\n' + '-' * len(string))
