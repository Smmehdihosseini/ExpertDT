import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DataManager:

    def __init__(self, dataframe,
                 x_col,
                 y_col,  # categorical labels
                 image_shape_original,  # img_height, img_width, color_channels
                 batch_size,
                 is_test_ds=False,
                 label_coding="one-hot",
                 image_shape_resized=None,
                 val_split=0.2,
                 patch_size=None,
                 preprocessing=None,  # Is a function
                 augment=None,  # Possible data augmentation
                 buffer_size="auto",  # Number of elements shuffled
                 dir_path_save_images=None):

        # Input params
        self.df = dataframe
        self.x_col = x_col
        self.y_col = y_col  # Labels are categorical
        self.image_shape_original = image_shape_original
        self.batch_size = batch_size
        self.is_test_ds = is_test_ds
        self.val_split = val_split
        self.path_size = patch_size  # Must be squared patch
        self.data_augmentation = augment
        self.buffer_size = buffer_size
        self.dir_path_save = dir_path_save_images
        self.image_shape_resized = image_shape_resized
        if label_coding not in ["one-hot", "one-dim"]:
            raise ValueError("label_coding must be one-hot or one-dim")
        else:
            self._output_message("Labels are: {}".format(label_coding))
        self.label_coding = label_coding

        # Others
        
        if image_shape_resized:
            self.image_shape_resized = image_shape_resized
        else:
            self.image_shape_resized = image_shape_original
        
        image_count = len(dataframe[x_col])
        if patch_size:
            self.patch_count_per_image = int(image_shape_original[0] * image_shape_original[1] /
                                             (patch_size[0] * patch_size[1]))  # As ratio between areas
            self.image_count_with_patches = self.patch_count_per_image * image_count
            self._output_message("{} patches for {} images: {} total patches.".format(self.patch_count_per_image,
                                                                                      image_count,
                                                                                      self.image_count_with_patches
                                                                                      ))
            self.path_size = patch_size
        else:
            self.image_count_with_patches = image_count
            self.path_size = self.image_shape_resized
        self.image_count = image_count
        self.color_channels = image_shape_original[-1]
        self.class_names = sorted(list(set(dataframe[y_col])))
        self.n_classes = len(set(dataframe[y_col]))
        self.autotune = tf.data.experimental.AUTOTUNE
        self.labels_categorical = dataframe[y_col]
        self.images_path = None

        # Preprocessing
        if preprocessing is None:
            self._preprocessing_function = self._rescale
        else:
            self._preprocessing_function = preprocessing

        # Others filled later
        self.ds_major = None
        self.ds_minor = None
        self.labels_univocal = None
        self.labels_numerical = list()

        if self.is_test_ds:
            self._output_message("is_test_ds set to {} --> test dataset built. "
                                 "Following params are ignored: "
                                 "buffer size (no shuffle); "
                                 "val split (no validation data).".format(self.is_test_ds))

        else:
            self._output_message("Train and val datasets built")

        self._build_ds()

        # Optionally get test ground truth
        if self.is_test_ds:
            self._get_labels_numerical()
        self._output_message("Classes correspondence: {}".format([(i, self.class_names[i])
                                                                  for i in range(len(self.class_names))]))

    def _get_labels_numerical(self):
        labels_univocal = np.array(self.class_names)
        self.labels_univocal = labels_univocal
        for label in self.df[self.y_col]:
            self.labels_numerical += [np.argmax(label == labels_univocal) for _ in range(self.patch_count_per_image)]
        self.labels_numerical = np.array(self.labels_numerical)

    def _build_ds(self):
        # Make dir for plotting some examples of images
        if self.dir_path_save:
            if not os.path.exists(self.dir_path_save):
                os.makedirs(self.dir_path_save)

        list_ds = tf.data.Dataset.from_tensor_slices((self.df[self.x_col],
                                                      self.df[self.y_col]))

        # Filter out dataset entries where self.df[self.y_col] is None
        # list_ds = list_ds.filter(lambda x, label: self.df[self.x_col] is None)  # TODO

        # Optionally split into two folds (val and training)
        if self.is_test_ds:
            val_size = 0
        else:
            list_ds = list_ds.shuffle(self.image_count_with_patches, reshuffle_each_iteration=False)
            val_size = int(self.image_count * self.val_split)

        ds_major = list_ds.skip(val_size)
        ds_minor = list_ds.take(val_size)

        ds_major = ds_major.map(self._process_path, num_parallel_calls=self.autotune)
        if self.dir_path_save:
            self._plot_first_n_images(ds_major, 1, dirpath_save=self.dir_path_save, tag="original")
        ds_major = ds_major.map(self._patch_image, num_parallel_calls=self.autotune)
        ds_major = ds_major.map(self._resize, num_parallel_calls=self.autotune)
        if self.data_augmentation:
            ds_major = self._data_augmentation(ds_major)
        ds_major = ds_major.unbatch()
        if self.dir_path_save:
            self._plot_first_n_images(ds_major, self.patch_count_per_image, dirpath_save=self.dir_path_save, tag="crop")

        ds_minor = ds_minor.map(self._process_path, num_parallel_calls=self.autotune)
        ds_minor = ds_minor.map(self._patch_image, num_parallel_calls=self.autotune)
        ds_minor = ds_minor.map(self._resize, num_parallel_calls=self.autotune)
        ds_minor = ds_minor.unbatch()

        ds_major = ds_major.map(self._preprocess)
        ds_minor = ds_minor.map(self._preprocess)

        if self.is_test_ds:
            self._output_message("Test dataset: {} "
                                 "images".format(int(tf.data.experimental.cardinality(ds_major).numpy())))
        else:
            self._output_message("Train: {} images".format(int(tf.data.experimental.cardinality(ds_major).
                                                               numpy())))
            self._output_message("Validation: {} images".format(int(tf.data.experimental.cardinality(ds_minor).
                                                                    numpy())))

        # If your data is such that it cannot be all loaded into memory,
        # then you cannot perform a "perfect" shuffle
        # (i.e. using a shuffle buffer size that matches the cardinality of your dataset)
        # In case of test dataset shuffle is skipped
        if not self.is_test_ds:
            if self.buffer_size == "auto":
                buffer_size_train = int(tf.data.experimental.cardinality(ds_major).numpy())
                buffer_size_val = int(tf.data.experimental.cardinality(ds_minor).numpy())
            else:
                buffer_size_train = self.buffer_size
                buffer_size_val = self.buffer_size
        else:
            buffer_size_train = None
            buffer_size_val = None

        self.ds_major = self._configure_for_performance(ds_major, buffer_size=buffer_size_train)
        self.ds_minor = self._configure_for_performance(ds_minor, buffer_size=buffer_size_val)

    def _process_path(self, file_path, label):
        if self.label_coding == "one-hot":
            label = self._get_label_one_dim(label)
        else:
            label = self._get_label_one_hot(label)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        # Label is returned as not one-hot here
        return img, label

    def _decode_img(self, image):
        # Convert the compressed string to a 3D uint8 tensor
        image = tf.image.decode_jpeg(image, channels=self.color_channels)
        return image

    def _get_label_one_hot(self, label):
        one_hot = label == self.class_names
        return one_hot

    def _get_label_one_dim(self, label):
        one_hot = label == self.class_names
        return tf.argmax(one_hot)

    def _resize(self, image, label):
        return tf.image.resize(image, [self.image_shape_resized[0], self.image_shape_resized[1]]), label

    def _patch_image(self, image, label):
        image = tf.expand_dims(image, 0)
        extracted_patches = tf.image.extract_patches(images=image,
                                                     sizes=[1,
                                                            self.path_size[0],
                                                            self.path_size[1],
                                                            1],
                                                     strides=[1,
                                                              self.path_size[0],
                                                              self.path_size[1],
                                                              1],
                                                     rates=[1, 1, 1, 1],
                                                     padding="VALID")
        patches = tf.reshape(extracted_patches, [-1, self.path_size[0], self.path_size[1], self.color_channels])
        # Labels must be repeated
        label = tf.repeat(label, self.patch_count_per_image, axis=0)
        if self.label_coding == "one-hot":
            label = tf.one_hot(label, depth=self.n_classes)
        return patches, label

    def _configure_for_performance(self, ds, buffer_size=None):
        ds = ds.cache()
        if buffer_size is not None:
            ds = ds.shuffle(buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)
        return ds

    def _preprocess(self, x, y):
        """
        Wrapper to map dataset with this function.
        """
        x = self._preprocessing_function(x)
        return x, y

    @staticmethod
    def _data_augmentation(ds):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            # tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        ])

        aug_ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y))
        return aug_ds

    @staticmethod
    def _output_message(string_to_print):
        print('-' * len(string_to_print))
        print(string_to_print)
        print('-' * len(string_to_print))

    def _plot_first_n_images(self, ds, n_images, dirpath_save, tag=""):
        # Dataset must be unbatched
        j = 1
        for x, y in ds.take(n_images):
            x = x.numpy().astype("uint8")
            fig, ax = plt.subplots(1, 1)
            ax.set_axis_off()
            ax.imshow(x)
            if not np.isscalar(y.numpy()):
                y = np.argmax(y.numpy())
            else:
                y = y.numpy()
            ax.set_title(self.class_names[y])
            filename = tag + "_{}-{}.png".format(j, n_images) if len(tag) > 0 else "_{}-{}.png".format(j, n_images)
            fig.savefig(os.path.join(dirpath_save, filename))
            plt.close(fig)
            j += 1

    @staticmethod
    def _rescale(x):
        return x/255.
