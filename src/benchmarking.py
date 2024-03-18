import os
import itertools
import subprocess
from datetime import datetime
from utils import output_message

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")

cuda_device = '1'
n_folds = 1
root_dir = "/space/ponzio/Morpheme_v2/results_classification/"
root_dir_dataframes = "/space/ponzio/Morpheme_v2/data/RandomTree_cc+chr-ONCO+pap_dfs/"
# experiment = f"{dt_string}_DEBUG"
# experiment = "08_04_2021-17_14_16_Baseline_NotCancerAllTumors_benchmarking"
experiment = "RandomTree_cc+chr-ONCO+pap_leaf2"
root_dir_results = os.path.join(root_dir, experiment)
x_col = "Path"
y_col = "Leaf2"
epochs = 100
cnn_model = "Vgg16"
test_reduced = False  # For hyperparams search

# Check if experiment done yet
if not os.path.exists(root_dir_results):
    os.makedirs(root_dir_results)
    answer = 'Y'
else:
    answer = input("{} exists! Overwrite (Y/N)?".format(root_dir_results))
if answer == 'Y':
    for fold in range(n_folds):
        if n_folds > 1:
            filepath_dataframe_train = os.path.join(root_dir_dataframes,
                                                    "df-train-balanced_{}_fold-{}.csv".format(y_col, fold))
            filepath_dataframe_test = os.path.join(root_dir_dataframes,
                                                   "df-test_{}_fold-{}.csv".format(y_col, fold))
            testing = True
        else:
            """
            Training on ALL train dataset in this case
            """
            filepath_dataframe_train = os.path.join(root_dir_dataframes,
                                                    "df-train-balanced_{}_whole.csv".format(y_col, fold))
            filepath_dataframe_test = os.path.join(root_dir_dataframes,
                                                   "df-test-whole.csv".format(y_col, fold))
            testing = True

        # VGG16 layers [None, 7, 11, 15]
        # DenseNet121 layers [None, 53, 80, 113]
        # ResNet layers [None, 15, 30, 45]
        first_trained_layer = [11]
        image_shape_resized = [(112, 112)]
        patch_size = [(1000, 1000)]
        learning_rate = [1e-4]
        optimizers = ["Adam"]

        hyperparameters_list = [first_trained_layer,
                                image_shape_resized,
                                patch_size,
                                learning_rate,
                                optimizers]
        hyperparameters_list = list(itertools.product(*hyperparameters_list))
        for j, hyperparameters in enumerate(hyperparameters_list):
            hyper_string = "FTL{}_IS{}_PS{}_LR{}_OPT{}".format(hyperparameters[0],  # first_trained_layer
                                                               hyperparameters[1],  # input_shape
                                                               hyperparameters[2],  # patch_size
                                                               hyperparameters[3],  # learning_rate
                                                               hyperparameters[4])  # optimizers
            dirpath_model = os.path.join(root_dir_results, "fold_{}".format(fold), hyper_string)
            print("Fold {}/{}".format(fold, n_folds-1))
            output_message("Training model {}/{} --> {}".format(j + 1,
                                                                len(hyperparameters_list),
                                                                hyper_string))

            if os.path.exists(os.path.join(dirpath_model, "cfm.png")):  # Means all done
                output_message("{} exists. Skip all".format(dirpath_model))
                continue
            elif os.path.exists(os.path.join(dirpath_model, "history.pickle")):
                output_message("{} exists. Skip training".format(dirpath_model))
                training = False
            else:
                training = True
            subprocess.run(["python", "run_classification.py",
                            filepath_dataframe_train,
                            filepath_dataframe_test,
                            dirpath_model,
                            x_col,
                            y_col,
                            "--cnn_model={}".format(cnn_model),
                            "--first_trained_layer={}".format(hyperparameters[0]),
                            "--image_shape_original={}".format(2000),
                            "2000",  # Click nargs > 1
                            "3",
                            "--training={}".format(training),
                            "--image_shape_resized={}".format(hyperparameters[1][0]),
                            str(hyperparameters[1][1]),
                            "--patch_size={}".format(hyperparameters[2][0]),
                            str(hyperparameters[2][1]),
                            "--testing={}".format(testing),
                            "--learning_rate={}".format(hyperparameters[3]),
                            "--optimizer={}".format(hyperparameters[4]),
                            "--epochs={}".format(epochs),
                            "--cuda_device={}".format(cuda_device),
                            "--test_reduced={}".format(test_reduced)])
else:
    print("Skipped.")
