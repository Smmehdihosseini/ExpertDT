import os
import itertools
import subprocess

# ----------------------------------------------------------
ROOT_PATH = "/space/ponzio/Morpheme_v2"
DATASET_TYPE = "ccRCC+pRCC"
EXPERIMENT_ID = "{}_Vgg16_300epochs_lre-3_focal-loss_26-04-2021_benchmarking".format(DATASET_TYPE)
ROOT_PATH_EXPERIMENT = os.path.join(ROOT_PATH, "results_segmentation", EXPERIMENT_ID)
ROOT_PATH_DATASET = os.path.join(ROOT_PATH, "data")
DATASET_TYPE_DIR = os.path.join(ROOT_PATH_DATASET, DATASET_TYPE)
DIRPATH_DATAFRAME = os.path.join(ROOT_PATH_DATASET, "dataframes_vascular_segmentation_26-04-2021", DATASET_TYPE)
# LOSSES = ["JaccardLoss",
#           "DiceLoss",
#           "BinaryFocalLoss",
#           "binary_crossentropy"]
LOSSES = ["BinaryFocalLoss"]#, "binary_crossentropy"]
MODEL_TYPES = ["Unet"]#, "LinkNet"]
# BACKBONES = ["densenet121", "vgg16", "resnet50"]
BACKBONES = ["vgg16"]#, "resnet50", "densenet121"]
ENCODER_FREEZE = ["False"]
# ENCODER_FREEZE = ["True", "False"]
EPOCHS = 300
BATCH_SIZE = 8
TRAIN = False
STEPS_BY_EPOCH = 300
COMPUTING_DEVICE = "0"
# ----------------------------------------------------------
MODEL_NAME = "segmentation_network_{}.h5".format(DATASET_TYPE)
DF_IMAGES = os.path.join(DIRPATH_DATAFRAME, "train", "images.csv")
DF_MASKS = os.path.join(DIRPATH_DATAFRAME, "train", "masks.csv")
PREDS_FOLDER_NAME = 'test_preds'.format(DATASET_TYPE)
DF_TEST = os.path.join(DIRPATH_DATAFRAME, "test", "images.csv")
INPUT_SHAPE = "512 512 3"

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

for experiment in itertools.product(LOSSES, MODEL_TYPES, BACKBONES, ENCODER_FREEZE):
    print(experiment)
    EXPERIMENT = '_'.join([experiment[0], experiment[1], experiment[2], experiment[3]])
    EXPERIMENT_DIR = os.path.join(ROOT_PATH_EXPERIMENT,
                                  EXPERIMENT)
    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR)

    subprocess.run(["/space/ponzio/Morpheme/.venv/bin/python", "_run_segnet.py",
                    "--prediction_folder_name={}".format(PREDS_FOLDER_NAME),
                    "--dataframe_images_train={}".format(DF_IMAGES),
                    "--dataframe_masks_train={}".format(DF_MASKS),
                    "--dir_experiment={}".format(EXPERIMENT_DIR),
                    "--loss={}".format(experiment[0]),
                    "--model_type={}".format(experiment[1]),
                    "--backbone={}".format(experiment[2]),
                    "--encoder_freeze={}".format(ENCODER_FREEZE),
                    "--model_name={}".format(MODEL_NAME),
                    "--epochs={}".format(EPOCHS),
                    "--batch_size={}".format(BATCH_SIZE),
                    "--computing_device={}".format(COMPUTING_DEVICE),
                    "--do_training={}".format(TRAIN),
                    "--steps_by_epoch={}".format(STEPS_BY_EPOCH),
                    "--dataframe_test={}".format(DF_TEST),
                    "--input_shape={}".format(INPUT_SHAPE)
                    ])

