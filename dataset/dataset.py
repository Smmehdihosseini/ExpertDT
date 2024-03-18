import os
import platform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import hashlib
from tqdm import tqdm

if platform.system() == "Windows":
    OPENSLIDE_PATH = r'D:\Openslide\Openslide\openslide-win64-20231011\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
elif platform.system() == "Linux":
    import openslide
    

class CropDatasetCached:

    def __init__(self, dataframe, cache_dir, n_classes=2, level=0, crop_size=(112, 112), batch_size=128, augment=False):

        self.dataframe = dataframe
        self.crop_size = crop_size
        self.augment = augment
        self.level = level
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.dataframe)
    
    def _get_cached_path(self, slide_path, top, left, size):

        identifier = f"{os.path.basename(slide_path)}_{top}_{left}_{size}_{self.level}"
        hash_id = hashlib.md5(identifier.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_id}.npy")

    def _process_image(self, slide_path, top, left, size):

        cached_path = self._get_cached_path(slide_path, top, left, size)

        if os.path.exists(cached_path):
            return np.load(cached_path)
        
        slide = openslide.OpenSlide(slide_path)
        region = slide.read_region((left, top), self.level, (size, size))
        slide.close()
        region = region.convert('RGB').resize(self.crop_size)
        image_array = np.array(region) / 255.0
        np.save(cached_path, image_array)

        return image_array
    
    def _process_and_cache_image(self, crop):

        cached_path = self._get_cached_path(crop['path'],  crop['top'], crop['left'], crop['size'])

        # If the processed crop is already cached, Skip Processing
        if not os.path.exists(cached_path):
            slide = openslide.OpenSlide(crop['path'])
            region = slide.read_region((crop['left'],  crop['top']), self.level, (crop['size'], crop['size']))
            slide.close()
            region = region.convert('RGB').resize(self.crop_size)
            image_array = np.array(region) / 255.0
            np.save(cached_path, image_array)

    def process_and_cache_all_images(self):

        for _, crop in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0], desc="Caching Images"):
            self._process_and_cache_image(crop)

    def __getitem__(self, idx):

        crop = self.dataframe.iloc[idx]
        image = self._process_image(slide_path=crop['path'],
                                    top=crop['top'],
                                    left=crop['left'],
                                    size=crop['size'])
        
        label = crop['label']
        
        # Convert image and labels to tensors
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.int32)

        return image, label

    def _augmentation(self):

        return tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

    def _generator(self):
        for i in range(len(self)):
            yield self[i]

    def plot_sample(self, idx):

        crop = self.dataframe.iloc[idx]
        image = self._process_image(slide_path=crop['path'],
                                    top=crop['top'],
                                    left=crop['left'],
                                    size=crop['size'])
        
        label = crop['label']
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"Label: {crop['type']} ({label})")
        plt.axis('off')
        plt.show()

    def get_dataset(self):
        # Create a tf.data.Dataset from the generator
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((self.crop_size[0], self.crop_size[1], 3), ())
        )

        # Apply data augmentation if enabled
        if self.augment:
            augmentation_layer = self._augmentation()
            dataset = dataset.map(
                lambda x, y: (augmentation_layer(x, training=True), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        # Apply one-hot encoding to labels within the dataset
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, depth=self.n_classes)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
    
    def check_dataset(self):

        print("\n>>> Checking dataset ...")
        
        # Check for missing values in dataset
        missing_values = self.dataframe.isnull().sum().sum()

        if missing_values > 0:
            print(f"✕ [WARNING]: There Are {missing_values} Missing Values In The Dataset.")
        else:
            print("✓ 1. No Missing Values Found ...")
        
        # Verify image paths and ensure they can be opened
        invalid_image_paths = []
        for path in self.dataframe['path'].unique():
            try:
                with openslide.OpenSlide(path) as slide:
                    pass  # Just attempting to open the slide to ensure it's valid
            except Exception as e:
                invalid_image_paths.append(path)
        
        if invalid_image_paths:
            print(f"✕ [WARNING]: Unable to Open {len(invalid_image_paths)} Image Paths!")
        else:
            print("✓ 2. All image Paths Are Valid ...")
        
        # Check labels are within expected range (e.g., 0 or 1 for binary classification)
        if self.dataframe['label'].min() < 0 or self.dataframe['label'].max() > 1:
            print("✕ Warning: Labels are outside the expected range (0 or 1).")
        else:
            print("✓ 3. Labels Are Within The Expected Range ...")
        
        # Optionally, check for class balance
        label_counts = self.dataframe['label'].value_counts()
        # Simple heuristic check for balance: no class should be less than half the size of another
        if label_counts.min() < label_counts.max() / 2:
            print("✕ [WARNING]: Significant Class Imbalance Detected!")
        else:
            print("✓ 4. Class Distribution Appears Balanced ...")

class CropDataset:
    
    def __init__(self, dataframe, n_classes=2, level=0, crop_size=(112, 112), batch_size=128, augment=False):

        self.dataframe = dataframe
        self.crop_size = crop_size
        self.augment = augment
        self.level = level
        self.n_classes = n_classes
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataframe)

    def _process_image(self, slide_path, top, left, size):

        slide = openslide.OpenSlide(slide_path)
        region = slide.read_region((left, top), self.level, (size, size))
        slide.close()

        region = region.convert('RGB')
        region = region.resize(self.crop_size)
        region = np.array(region) / 255.0

        return region

    def __getitem__(self, idx):

        crop = self.dataframe.iloc[idx]
        image = self._process_image(slide_path=crop['path'],
                                    top=crop['top'],
                                    left=crop['left'],
                                    size=crop['size'])
        
        label = crop['label']
        
        # Convert image and labels to tensors
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.int32)

        return image, label

    def _augmentation(self):

        return tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

    def _generator(self):
        for i in range(len(self)):
            yield self[i]

    def plot_sample(self, idx):

        crop = self.dataframe.iloc[idx]
        image = self._process_image(slide_path=crop['path'],
                                    top=crop['top'],
                                    left=crop['left'],
                                    size=crop['size'])
        
        label = crop['label']
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"Label: {crop['type']} ({label})")
        plt.axis('off')
        plt.show()

    def get_dataset(self):
        # Create a tf.data.Dataset from the generator
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=((self.crop_size[0], self.crop_size[1], 3), ())
        )

        # Apply data augmentation if enabled
        if self.augment:
            augmentation_layer = self._augmentation()
            dataset = dataset.map(
                lambda x, y: (augmentation_layer(x, training=True), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        # Apply one-hot encoding to labels within the dataset
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, depth=self.n_classes)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
    
    def check_dataset(self):

        print("\n>>> Checking dataset ...")
        
        # Check for missing values in dataset
        missing_values = self.dataframe.isnull().sum().sum()

        if missing_values > 0:
            print(f"✕ [WARNING]: There Are {missing_values} Missing Values In The Dataset.")
        else:
            print("✓ 1. No Missing Values Found ...")
        
        # Verify image paths and ensure they can be opened
        invalid_image_paths = []
        for path in self.dataframe['path'].unique():
            try:
                with openslide.OpenSlide(path) as slide:
                    pass  # Just attempting to open the slide to ensure it's valid
            except Exception as e:
                invalid_image_paths.append(path)
        
        if invalid_image_paths:
            print(f"✕ [WARNING]: Unable to Open {len(invalid_image_paths)} Image Paths!")
        else:
            print("✓ 2. All image Paths Are Valid ...")
        
        # Check labels are within expected range (e.g., 0 or 1 for binary classification)
        if self.dataframe['label'].min() < 0 or self.dataframe['label'].max() > 1:
            print("✕ Warning: Labels are outside the expected range (0 or 1).")
        else:
            print("✓ 3. Labels Are Within The Expected Range ...")
        
        # Optionally, check for class balance
        label_counts = self.dataframe['label'].value_counts()
        # Simple heuristic check for balance: no class should be less than half the size of another
        if label_counts.min() < label_counts.max() / 2:
            print("✕ [WARNING]: Significant Class Imbalance Detected!")
        else:
            print("✓ 4. Class Distribution Appears Balanced ...")