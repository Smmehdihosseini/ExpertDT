import os
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import Callback
import numpy as np

class SaveCallback(Callback):

    def __init__(self, save_path, save_epoch=10, save_weights_only=True):

        super(SaveCallback, self).__init__()
        self.save_path = save_path
        self.save_weights_only = save_weights_only
        self.save_epoch = save_epoch

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_epoch == 0:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            filepath = os.path.join(self.save_path, f'model_epoch_{epoch + 1}.h5')
            if self.save_weights_only:
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath)
            print(f'\n >>> Saved model at Epoch {epoch + 1} to {filepath}')

class Vgg16:
    def __init__(self, input_shape, n_classes, first_trained_layer=None, save_dir='_weights'):

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.first_trained_layer = first_trained_layer
        self.model = self.build_model()
        self.save_dir = save_dir

    def build_model(self):

        vgg16 = VGG16(include_top=False, input_shape=self.input_shape, weights='imagenet')

        # Freeze the specified first layers
        if self.first_trained_layer is not None:
            for layer in vgg16.layers[:self.first_trained_layer]:
                layer.trainable = False

        x = vgg16.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x) 
        x = Dropout(0.5)(x)
        predictions = Dense(self.n_classes, activation='softmax')(x) 

        # Creating the final model
        model = Model(inputs=vgg16.input, outputs=predictions)

        return model

    def compile(self,
                learning_rate=1e-5,
                loss='categorical_crossentropy',
                metrics=['accuracy']):
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics)

    def fit(self, data, epochs, callbacks):

        self.model.fit(data,
                       epochs=epochs,
                       callbacks=[callbacks])

    def summary(self):
        self.model.summary()

    def save_weights(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Define the path for saving the weights
        weights_path = os.path.join(self.save_dir, 'vgg16_weights.h5')
        
        # Save the model's weights
        self.model.save_weights(weights_path)
        print(f'>>> Model Weights Saved To: {weights_path}')

    def load_weights(self, weights_path):

        self.model.load_weights(weights_path)
        print(f'>>> Succesfully Loaded Weights ...')

    def predict(self, input):

        if input.ndim == 3:
            input = np.expand_dims(input, axis=0)

        predictions = self.model.predict(input, verbose=0)

        return predictions