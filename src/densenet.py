import tensorflow as tf
from src._cnn import AbstractCnn


class DenseNet(AbstractCnn):

    def _init(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        """
        self.first_trained_layer = self._setter(None, kwargs.get("first_trained_layer"),
                                                "first trained layer")
        # self._config_gpu()
        self._build()

    def _build(self):
        if self.first_trained_layer is None:
            densenet = tf.keras.applications.densenet.DenseNet121(weights=None,
                                                                  include_top=False,
                                                                  classes=self.n_classes,
                                                                  input_shape=self.input_shape)
            x = tf.keras.layers.GlobalAveragePooling2D()(densenet.output)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            predictions = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
            self.model = tf.keras.Model(inputs=densenet.input, outputs=predictions)
        else:
            densenet = tf.keras.applications.densenet.DenseNet121(weights="imagenet",
                                                                  include_top=False,
                                                                  classes=self.n_classes,
                                                                  input_shape=self.input_shape)
            x = tf.keras.layers.GlobalAveragePooling2D()(densenet.output)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            predictions = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
            self.model = tf.keras.Model(inputs=densenet.input, outputs=predictions)
            for layer in densenet.layers[:self.first_trained_layer]:
                layer.trainable = False
            for layer in densenet.layers[self.first_trained_layer:]:
                layer.trainable = True

        # Print layer.trainable
        self._output_message("Print layer.trainable")
        for layer in self.model.layers:
            print("{} - {} is trainable? {}".format(layer,
                                                    layer.name,
                                                    layer.trainable))
        self.model.summary()
