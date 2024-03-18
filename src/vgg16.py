import tensorflow as tf
from _cnn import AbstractCnn


class Vgg16(AbstractCnn):

    def _init(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        """
        self.first_trained_layer = self._setter(None, kwargs.get("first_trained_layer"),
                                                "first trained layer")
        # self._config_gpu()
        self._build()

    @staticmethod
    def _restore_original_image_from_array(x,
                                           data_format='channels_last'):
        mean = [103.939, 116.779, 123.68]

        # Zero-center by mean pixel
        if data_format == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] += mean[0]
                x[1, :, :] += mean[1]
                x[2, :, :] += mean[2]
            else:
                x[:, 0, :, :] += mean[0]
                x[:, 1, :, :] += mean[1]
                x[:, 2, :, :] += mean[2]
        else:
            x[..., 0] += mean[0]
            x[..., 1] += mean[1]
            x[..., 2] += mean[2]

        if data_format == 'channels_first':
            # 'BGR'->'RGB'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'BGR'->'RGB'
            x = x[..., ::-1]
        return x

    def _build(self):
        if self.first_trained_layer is None:
            vgg16 = tf.keras.applications.vgg16.VGG16(weights=None,
                                                      include_top=False,
                                                      classes=self.n_classes,
                                                      input_shape=self.input_shape)
            x = tf.keras.layers.GlobalAveragePooling2D()(vgg16.output)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            predictions = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
            self.model = tf.keras.Model(inputs=vgg16.input, outputs=predictions)
        else:
            vgg16 = tf.keras.applications.vgg16.VGG16(weights="imagenet",
                                                      include_top=False,
                                                      classes=self.n_classes,
                                                      input_shape=self.input_shape)
            x = tf.keras.layers.GlobalAveragePooling2D()(vgg16.output)
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            predictions = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
            self.model = tf.keras.Model(inputs=vgg16.input, outputs=predictions)
            for layer in vgg16.layers[:self.first_trained_layer]:
                layer.trainable = False
            for layer in vgg16.layers[self.first_trained_layer:]:
                layer.trainable = True

        # Print layer.trainable
        self._output_message("Print layer.trainable")
        for layer in self.model.layers:
            print("{} - {} is trainable? {}".format(layer,
                                                    layer.name,
                                                    layer.trainable))
        self.model.summary()
