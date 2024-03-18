import os
import pickle
import subprocess
from dataset import Dataset


src = "/space/ponzio/Morpheme_v2/data/RCC_WSIs/"
device = "0"
save_images = False
output_folder = "/space/ponzio/Morpheme_v2/results_segmentation/rudan_all"
keep_out = ("HP10.5813",
            "HP19.4372",
            "HP19.5254",
            "HP13.3201",
            "HP14.4279"
            "HP14.8231",
            "HP14.10122")
parameters_file = "/space/ponzio/Morpheme_v2/results_segmentation/" \
                  "ccRCC+pRCC_Vgg16_300epochs_lre-3_focal-loss_26-04-2021_" \
                  "benchmarking/BinaryFocalLoss_Unet_vgg16_False/parameters.json"
# Make dataset
dataset = Dataset(src,
                  crop_size=2000,
                  slide_extensions=(".scn", ".svs"),
                  keep_out=keep_out,
                  resize=512)

# Predict
for j, crop_obj in enumerate(dataset.crop_objs_original):
    patient_slide_id = '-'.join(os.path.basename(crop_obj.indexes[0]["filepath_slide"]).split('.')[:-1])
    filepath_crops_obj = os.path.join(output_folder, "{}_crop_obj.pickle".format(patient_slide_id))
    if os.path.exists(filepath_crops_obj):
        print("{} skipped".format(filepath_crops_obj))
        continue
    print("Crop obj {}/{}".format(j + 1, len(dataset.crop_objs_original)))
    with open(filepath_crops_obj, "wb") as handle:
        pickle.dump(crop_obj, handle)

    subprocess.run(["/space/ponzio/Morpheme_v2/.venv/bin/python", "predictor.py",
                    "{}".format(filepath_crops_obj),
                    "{}".format(parameters_file),
                    "{}".format(output_folder),
                    "--device={}".format(device),
                    "--save_images={}".format(save_images)
                    ])
