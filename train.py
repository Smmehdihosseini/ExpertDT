import os
import platform
import json
import pandas as pd
import argparse

# Set The Desired GPU Device for Server
runtime_gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime_gpu_id)

# CPU Thread Limitations for Server
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import tensorflow as tf
from dataset.dataset import CropDatasetCached
from model.vgg16 import Vgg16, SaveCallback

if platform.system() == "Windows":
     DEFAULT_PATH = "_info/default_path_local.json"
elif platform.system() == "Linux":
     DEFAULT_PATH = "_info/default_path_server.json"

with open(DEFAULT_PATH, 'r') as file:
            default_dirs = json.load(file)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Training Stages of Binary Classification of ExpertDT")

    parser.add_argument("--stage", type=str, default="Leaf1", help='Training Stage of ExpertDT')
    parser.add_argument("--dfs_dir", type=str, default=default_dirs['stage_df_dir'], help='DataFrames Directory')
    parser.add_argument("--weights_dir", type=str, default=default_dirs['weights_dir'], help='DataFrames Directory')
    parser.add_argument("--cache_dir", type=str, default=default_dirs['cache_dir'], help='Temporary Image Cache')

    parser.add_argument("--level", type=int, default=0, help='WSIs Level of Magnification')
    parser.add_argument("--crop_size", type=int, default=112, help='Resize Crop Size')
    parser.add_argument("--batch_size", type=int, default=128, help='Dataset Batch Size')
    parser.add_argument("--n_classes", type=int, default=2, help='Crop Size Directory')

    parser.add_argument("--first_trained_layer", type=int, default=11, help='VGG16 First Trained Layer')
    parser.add_argument("--learning_rate", type=float, default=1e-5, help='Training Learning Rate')
    parser.add_argument("--epochs", type=int, default=150, help='Training Epochs')
    parser.add_argument("--loss", type=str, default='categorical_crossentropy', help='Training Loss Function')
    parser.add_argument("--monitor_metric", type=str, default='loss', help='Training Monitor Metric')
    parser.add_argument("--early_patience", type=int, default=20, help='Early Stopping Callback Patience')
    parser.add_argument("--save_epochs", type=int, default=10, help='Saving Weights Callback')

    parser.add_argument("--gpu_fraction", type=int, default=8, help="GPU Memory Usage for Runtime") 
    parser.add_argument("--gpu_policy", type=str, default="mixed_float16", help="GPU Mixed Precision Policy")

    args = parser.parse_args()

    tf.keras.mixed_precision.set_global_policy(args.gpu_policy)
    runtime_gpu_limit = args.gpu_fraction*1024
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[runtime_gpu_id], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=runtime_gpu_limit)])

    print(f"\n>>> Loading Data for {args.stage} ...")

    stage_data = pd.read_csv(f"{args.dfs_dir}/{args.stage}_data.csv")

    print(f">>> Labels Distribution of {args.stage}:")

    print(stage_data['type'].value_counts())

    print(">>> Preparing Training DataLoader ...")

    dataset = CropDatasetCached(dataframe=stage_data,
                          cache_dir=f"{args.cache_dir}{args.stage}",
                          level=args.level,
                          crop_size=(args.crop_size,args.crop_size),
                          batch_size=args.batch_size,
                          n_classes=args.n_classes)
    
    dataset.process_and_cache_all_images()
    tf_dataset = dataset.get_dataset()

    print(">>> Preparing Training Model ...")

    vgg16 = Vgg16(input_shape=(args.crop_size, args.crop_size, 3),
                  n_classes=args.n_classes,
                  first_trained_layer=args.first_trained_layer)
    
    print(">>> Compiling Training Model ...")

    vgg16.compile(learning_rate=args.learning_rate,
                        loss=args.loss,
                        metrics=['accuracy'])

    print(">>> Loading Training Callbacks ...")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=args.monitor_metric, patience=args.early_patience, verbose=1)
    save_callback = SaveCallback(save_path=f"{args.weights_dir}/{args.stage}", save_epoch=args.save_epochs, save_weights_only=True)

    print(">>> Starting Training ... \n")

    vgg16.fit(tf_dataset,
                epochs=args.epochs,
                callbacks=[save_callback, early_stopping])
