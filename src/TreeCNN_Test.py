import os
import cv2
import math
import pickle
import itertools
import openslide
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
import vgg16
from _treecnn_utils import *
import matplotlib.pyplot as plt
from wsi_manager import CropList
from matplotlib import cm as plt_cmap
from make_heatmaps import make_heatmap
from wsi_manager import SectionManager
from make_heatmaps import _is_background
from make_heatmaps import _batch_iterator
from sklearn.metrics import confusion_matrix, f1_score


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

# SETTINGS >>>>
FONTSIZE = 14
rootdir = "/space/ponzio/Morpheme_v2/"
# rootdir_output = "/space/ponzio/Morpheme_v2/results_classification/ExperDt_again_to_check" # Final DRCC
rootdir_output = "/space/ponzio/Morpheme_v2/results_classification/NaiveDT"
try:
    os.makedirs(rootdir_output)
except:
    pass
# rootdir_output = "/space/ponzio/Morpheme_v2/results_classification/RandomTree_cc+ONCO-CHROMO+pRCC_results" # Random tree
# paths_partial = ["RandomTree_cc+O-C+p_root/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
#                  "RandomTree_cc+O-C+p_node/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
#                  "RandomTree_cc+O-C+p_leaf1/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
#                  "RandomTree_cc+O-C+p_leaf2/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam"]
# Final RCC tree
# paths_partial = ["01_04_2021-14_52_42_Root/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
#                  "01_04_2021-14_54_12_Node/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
#                  "01_04_2021-15_27_47_Leaf1/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
#                  "02_04_2021-13_53_07_Leaf2_benchmarking/"\
#                  "fold_0/FTL7_IS(112, 112)_PS(1000, 1000)_LR1e-05_OPTRMSprop"]
# NaiveDT
paths_partial = ["RandomTree_cc+chr-ONCO+pap_root/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
                 "RandomTree_cc+chr-ONCO+pap_node/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
                 "RandomTree_cc+chr-ONCO+pap_leaf1/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam",
                 "RandomTree_cc+chr-ONCO+pap_leaf2/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam"]
# Load whole test dataframe
df_test = pd.read_csv("/space/ponzio/Morpheme_v2/data/RandomTree_cc+chr-ONCO+pap_dfs/df-test-whole.csv") # Final DRCC tree
# df_test = pd.read_csv("/space/ponzio/Morpheme_v2/data/TreeCNN_Final_DFs_2-folds/df-test-whole.csv") # Final DRCC tree
# df_test = pd.read_csv(
    # "/space/ponzio/Morpheme_v2/data/RandomTree_cc+ONCO-CHROMO+pRCC_dfs/df-test-whole.csv")  # Random tree
rootdir_wsi = os.path.join(rootdir, "data/RCC_WSIs")
if not os.path.exists(rootdir_output):
    os.makedirs(rootdir_output)
# <<<<


# LOAD TEST DATA >>>>
print("Patients in ROIs test dataset")
print(pd.pivot_table(df_test, values="Patient", index="CancerType", aggfunc=pd.Series.nunique))

patients_test_unique = df_test['Patient'].unique()
patients_test_unique_id = [''.join(p.split('.')[0:2]) for p in patients_test_unique]

# The following patients are not present in ROIs dataset (only WSIs avaiable)
# Possible further patents must be added here
patients_not_in_ROIs = [
    "HP195524",
    "HP1213588",
    "HP19008963",
    "HP131799",
    "HP193695",
    "HP145590",
    "HP123187",
    "HP197949",
    "HP51171",
    "HP70605",
]

patients_test_unique_id += patients_not_in_ROIs
# Look for test patients in WSIs rootdir
# Get all WSIs' filepaths
filepaths_wsi = [os.path.join(dp, f) for dp, dn, filenames in os.walk(rootdir_wsi) for f in filenames if
                 os.path.basename(f).split('.')[-1] == 'scn'
                 or os.path.basename(f).split('.')[-1] == 'svs'
                 or os.path.basename(f).split('.')[-1] == 'tif']
# filepaths_wsi = [path for path in filepaths_wsi if "not-in" not in path]
filepaths_wsi_not_in = [path for path in filepaths_wsi if "not-in" in path]

# Get Patients ID from WSIs
patient_wsi_ids = [''.join(os.path.basename(f).split('.')[0:2]) for f in filepaths_wsi]

# Get test patients WSIs
# N.B. Some patients have more than 1 wsi
# Label can be obtained from filepath
# '/space/ponzio/Morpheme_v2/data/RCC_WSIs/pre/pRCC/HP09.5392.A16.pRCC.scn' --> Label: pRCC
filepaths_wsi_test = list()
labels_wsi = list()
# Dict to map univocal patient in wsi to univocal patient in test set. Note that some patients have more than one wsi.
patient_wsi2test = dict()
for patient in patients_test_unique_id:
    for patient_wsi_id, filepath_wsi in zip(patient_wsi_ids, filepaths_wsi):
        if patient in patient_wsi_id:
            patient_wsi2test[patient_wsi_id] = patient
            filepaths_wsi_test.append(filepath_wsi)
            labels_wsi.append(filepath_wsi.split(os.sep)[-2])

print("\n{} further patients (only WSIs):".format(len(patients_not_in_ROIs)))
for patient in np.unique(patients_not_in_ROIs):
    for patient_wsi_id, filepath_wsi in zip(patient_wsi_ids, filepaths_wsi):
        if patient in patient_wsi_id:
            print(patient, filepath_wsi.split(os.sep)[-2])
            break
print("\n{} patients in total".format(len(patients_test_unique_id)))
# <<<<

# PREPARE CROPS >>>>
cropSize = 1000
crops_obj_list_tmp = list()
for filepath, slide_label in zip(filepaths_wsi_test, labels_wsi):
    print("Processing: {}".format(filepath))
    sections = SectionManager(cropSize, overlap=1)
    crops_obj_list_tmp.append(sections.crop(filepath, slide_label=slide_label, size=112, save_dir=None))
# <<<<

# LOAD NETWORK >>>>
vgg16 = Vgg16(n_classes=2,
              input_shape=(112,
                           112,
                           3))

vgg16.init(first_trained_layer=11)
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
vgg16.model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=['accuracy'])
# <<<<

# LOAD WEIGHTS
stages = ["CancerBinary", "Node", "Leaf1", "Leaf2"]
weights_paths = [os.path.join("/space/ponzio/Morpheme_v2/results_classification/NaiveDT", path, "weights.hdf5")
                 for path in paths_partial]
labels_categorical_array = [sorted(df_test[stage].dropna().unique()) for stage in stages]
# <<<<

# PREDICT WSIs >>>>
# # Predict
for j, crop_obj in enumerate(crops_obj_list_tmp):
    print("{}/{}".format(j + 1, len(crops_obj_list_tmp)))
    for batch_images, batch_indexes in _batch_iterator(crop_obj,
                                                       crop_obj.indexes,
                                                       1000):
        for weights_path, stage, labels_categorical in zip(weights_paths,
                                                           stages,
                                                           labels_categorical_array):
            vgg16.model.load_weights(weights_path)
            batch_predictions = vgg16.model.predict(tf.keras.applications.vgg16.preprocess_input(np.array(batch_images)))
            for image, index, prediction in zip(batch_images, batch_indexes, batch_predictions):
                if not _is_background(image):
                    index['tissue'] = True
                    index[f'{stage}_prediction'] = prediction
                    index[f'{stage}_argmax'] = labels_categorical[np.argmax(prediction)]
# Save all crop objs
filepath_crops_objs = os.path.join(rootdir_output, "crops.pickle")
with open(filepath_crops_objs, "wb") as handle:
    pickle.dump(crops_obj_list_tmp, handle)
# <<<<