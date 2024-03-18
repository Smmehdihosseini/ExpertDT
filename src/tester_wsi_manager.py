# import os
# import openslide
# import numpy as np
# from time import time
# # from vgg16 import Vgg16
# # import tensorflow as tf
# from wsi_manager import SectionManager
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
#
# def benchmark(crops_, my_model):
#     print("Found {} patches".format(len(crops_)))
#     start = 0
#     for i in range(1, 12):
#         if i == 2:
#             start = time()
#         image = crops_[i].convert('RGB')
#         image.thumbnail((112, 112))
#         x = [np.array(image)]
#         _ = my_model.model.predict_on_batch(np.array(x))
#     print("Time estimation to process all patches: {:.1f} [min]".format(((time() - start) / 10 * len(crops_)) / 60))
#
#
# cropSize = 2000
# rootpath = "/space/ponzio/Morpheme_v2/data/Annotations_onco/"
# rootpath_dest =os.path.join(rootpath, "Crops")
# filepaths = [os.path.join(rootpath, filepath) for filepath in os.listdir(rootpath)]
# for filepath in filepaths:
#     if filepath.endswith(".svs") or filepath.endswith(".tif"):
#         if not os.path.exists(rootpath_dest):
#             os.makedirs(rootpath_dest)
#         print("Processing: {}".format(filepath))
#         slide = openslide.OpenSlide(filepath)
#         sections = SectionManager(cropSize, overlap=1)
#         crops = sections.crop(slide)
#         rootdir_name = os.path.basename(filepath).split('.')[0]
#         dir_path = os.path.join(rootpath_dest, rootdir_name, "tumor", "Annotation", "subimages")
#         if not os.path.exists(dir_path):
#             os.makedirs(dir_path)
#             for j, crop in enumerate(crops):
#                 image = crop.convert('RGB')
#                 filepath_image = os.path.join(dir_path, "{}.png".format(j))
#                 image.save(filepath_image)
#
# # rootpath = "/space/ponzio/Morpheme_v2/data/Annotations_chromo/"
# # rootpath_dest = os.path.join(rootpath, "Crops")
# # filepaths = [os.path.join(rootpath, filepath) for filepath in os.listdir(rootpath)]
# # for filepath in filepaths:
# #     if filepath.endswith(".svs") or filepath.endswith(".tif"):
# #         if not os.path.exists(rootpath_dest):
# #             os.makedirs(rootpath_dest)
# #         print("Processing: {}".format(filepath))
# #         slide = openslide.OpenSlide(filepath)
# #         sections = SectionManager(cropSize, overlap=1)
# #         crops = sections.crop(slide)
# #         rootdir_name = os.path.basename(filepath).split('.')[0]
# #         dir_path = os.path.join(rootpath_dest, rootdir_name, "tumor", "Annotation", "subimages")
# #         if not os.path.exists(dir_path):
# #             os.makedirs(dir_path)
# #             for j, crop in enumerate(crops):
# #                 image = crop.convert('RGB')
# #                 filepath_image = os.path.join(dir_path, "{}.png".format(j))
# #                 image.save(filepath_image)
#
# # Load slide and create CropList object
#
# #
# # # print crops indexes data
# # print(crops.indexes[0])
#
# # Load model
# # filepath_model_weights = os.path.join(
# #     "/space/ponzio/Morpheme_v2/results_classification/hyperparams_analysis/TnT_VGG16_best/fold_0",
# #     "FTL11_IS(112, 112)_PS(500, 500)_LR0.0001_OPTAdam",
# #     "weights.hdf5")
# #
# # vgg16 = Vgg16(n_classes=2,
# #               input_shape=(112,
# #                            112,
# #                            3))
# #
# # vgg16.init(first_trained_layer=0)
# # opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# # vgg16.model.compile(
# #     optimizer=opt,
# #     loss="categorical_crossentropy",
# #     metrics=['accuracy'])
# #
# # vgg16.model.load_weights(filepath_model_weights)
# # benchmark(crops, vgg16)

import os
import itertools
import openslide
import numpy as np
import pandas as pd
from vgg16 import Vgg16
import tensorflow as tf
import matplotlib.pyplot as plt
from wsi_manager import SectionManager
from wsi_manager import CropList

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

rootdir = "/space/ponzio/Morpheme_v2/"

# # Load whole test dataframe
df_test = pd.read_csv("/space/ponzio/Morpheme_v2/data/TreeCNN_Final_DFs_2-folds/df-test-whole.csv")
print(pd.pivot_table(df_test, values="Patient", index="CancerType", aggfunc=pd.Series.nunique))

patients_test_unique = df_test['Patient'].unique()
patients_test_unique_id = [''.join(p.split('.')[0:2]) for p in patients_test_unique]

# Look for test patients in WSIs rootdir
rootdir_wsi = os.path.join(rootdir, "data/RCC_WSIs")

# Get all WSIs' filepaths
filepaths_wsi = [os.path.join(dp, f) for dp, dn, filenames in os.walk(rootdir_wsi) for f in filenames if
                 os.path.basename(f).split('.')[-1] == 'scn'
                 or os.path.basename(f).split('.')[-1] == 'svs'
                 or os.path.basename(f).split('.')[-1] == 'tif']

# Get Patients ID from WSIs
patient_wsi_ids = [''.join(os.path.basename(f).split('.')[0:2]) for f in filepaths_wsi]

# Get test patients WSIs
# N.B. Some patients have more than 1 wsi
# Label can be obtained from filepath
# '/space/ponzio/Morpheme_v2/data/RCC_WSIs/pre/pRCC/HP09.5392.A16.pRCC.scn' --> Label: pRCC
filepaths_wsi_test = list()
labels = list()
for patient in patients_test_unique_id:
    for patient_wsi_id, filepath_wsi in zip(patient_wsi_ids, filepaths_wsi):
        if patient in patient_wsi_id:
            filepaths_wsi_test.append(filepath_wsi)
            labels.append(filepath_wsi.split(os.sep)[-2])


cropSize = 500
crops_obj_list = list()
for filepath, label in zip(filepaths_wsi_test, labels):
    print("Processing: {}".format(filepath))
    # slide = openslide.OpenSlide(filepath)
    sections = SectionManager(cropSize, overlap=1)
    crops_obj_list.append(sections.crop(filepath, slide_label=label, size=112))

model_root = "/space/ponzio/Morpheme_v2/results_classification/"\
             "01_04_2021-14_52_42_Root/fold_0/FTL11_IS(112, 112)_PS(1000, 1000)_LR0.0001_OPTAdam/weights.hdf5"
labels_categorical = ["Cancer", "NotCancer"]
vgg16 = Vgg16(n_classes=2,
              input_shape=(112,
                           112,
                           3))

vgg16.init(first_trained_layer=11)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
vgg16.model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=['accuracy'])

vgg16.model.load_weights(model_root)

filepath = "/space/ponzio/Morpheme_v2/data/WSIs_benchmark/template_slide_rCC_cropped.svs"
from make_heatmaps import predict_and_make_heatmaps
predict_and_make_heatmaps(filepaths_wsi_test[4],
                          2000,
                          vgg16,
                          os.path.join(rootdir, "PROVA"),
                          tf.keras.applications.vgg16.preprocess_input,
                          os.path.join(rootdir, "crop_obj.pickle"),
                          resolution_level=3)
