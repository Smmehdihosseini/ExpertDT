import os
import platform
import json
import random
from glob import glob
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
from run.expertdt import ExpertDT

if platform.system() == "Windows":
     DEFAULT_PATH = "_info/default_path_local.json"
     DIV = "\\"
elif platform.system() == "Linux":
     DEFAULT_PATH = "_info/default_path_server.json"
     DIV = "/"

with open(DEFAULT_PATH, 'r') as file:
            default_dirs = json.load(file)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Test Stage of All Patients Using ExpertDT")

    parser.add_argument("--root_dir", type=str, default=default_dirs['root_dir'], help="Root WSIs Folder")
    parser.add_argument("--unknown_dir", type=str, default=default_dirs['unknown_dir'], help="Unknown WSIs Folder")
    parser.add_argument("--weights_dir", type=str, default=default_dirs['weights_dir'], help='DataFrames Directory')
    parser.add_argument("--ids_dir", type=str, default=default_dirs['ids_dir'], help="Path to Patient IDs File")
    parser.add_argument("--split_ids_dir", type=str, default=default_dirs['split_ids_dir'], help="Split Data IDs")
    parser.add_argument("--tree_pair_dir", type=str, default=default_dirs['tree_pair_dir'], help="Class Labels in Different Stages of Tree")
    parser.add_argument("--results_dir", type=str, default=default_dirs['results_dir'], help="Results Directory")
    parser.add_argument("--figures_dir", type=str, default=default_dirs['figures_dir'], help="Figures Directory")

    parser.add_argument("--verbose", type=bool, default=True, help="Verbose Print Details")

    parser.add_argument("--crop_size", type=int, default=1000, help="Patches Crop Size During Cropping Stage")
    parser.add_argument("--crop_resize", type=int, default=112, help="Patches Crop Size For Inference Stage")
    parser.add_argument("--wsi_level", type=int, default=0, help="Whole Slide Image Level of Magnification During Cropping Stage")
    parser.add_argument("--overlap", type=int, default=1, help="Patches Overlap During Cropping Stage; 1: No Overlap, 2: 50%, etc.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size For Inference Multiprocessing")
    parser.add_argument("--wsi_formats", nargs=3, type=str, default=['scn', 'svs', 'tif'], help="Possible Formats for Whole Slide Image")
    parser.add_argument("--save_plots", type=bool, default=True, help="Save Figures")

    parser.add_argument("--node_prune_threshold", type=int, default=50, help="Node Pruning Threshold")

    parser.add_argument("--gpu_fraction", type=int, default=8, help="GPU Memory Usage for Runtime") 
    parser.add_argument("--random_seed", type=int, default=42, help="Random Seed for Reproducibility")
    parser.add_argument("--test_note", type=str, default="50_threshold", help="thr_30")

    args = parser.parse_args()

    runtime_gpu_limit = args.gpu_fraction*1024
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[runtime_gpu_id], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=runtime_gpu_limit)])

    random.seed(args.random_seed)

    with open(f"{args.split_ids_dir}", 'r') as file:
        train_test_ids = json.load(file)

    ids_df = pd.read_csv(f"{args.ids_dir}")

    results_path = f"{args.results_dir}_{args.test_note}.csv"
    column_names = ['id', 'subtype', 'slide', 'format', 'node_pruned','node_count_1', 'node_count_2', 'ccRCC', 'pRCC', 'CHROMO', 'ONCOCYTOMA', 'result']

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        print(">>> Results Exists, Loading it ...")
    else:
        results_df = pd.DataFrame([], column_names=column_names)
        print(">>> Results Not Found, Creating New Results ...")

    for subtype in train_test_ids['Test'].keys():

        if args.verbose:
            print(f"\n")
            print(f"--"*15)
            print(f">>> Preparing Data for '{subtype}' Tests:")
            print(f"--"*15)
            print(f"\n")

        for index in train_test_ids['Test'][subtype]:
            
            if args.verbose:
                print(f"++"*15)
                print(f">>> Patient ID: {ids_df.iloc[index].id}")
                print(f"++"*15)
            
            glob_pattern = []
            
            if subtype != "Unknown":

                subtype_path = f"{args.root_dir}{DIV}{subtype}"

                if glob(f"{subtype_path}{DIV}*{ids_df.iloc[index].id}*"):
                    glob_pattern = glob(f"{subtype_path}{DIV}*{ids_df.iloc[index].id}*")

                elif glob(f"{args.root_dir}{DIV}pre{DIV}{subtype}{DIV}*{ids_df.iloc[index].id}*"):
                    glob_pattern = glob(f"{args.root_dir}{DIV}pre{DIV}{subtype}{DIV}*{ids_df.iloc[index].id}*")

            else:
                if glob(f"{args.unknown_dir}{DIV}*{ids_df.iloc[index].id}*"):
                    glob_pattern = glob(f"{args.unknown_dir}{DIV}*{ids_df.iloc[index].id}*")

            results_list = []

            for slide_dir in glob_pattern:

                slidename = slide_dir.split(f"{DIV}")[-1][:-4]
                slideformat = slide_dir.split(f"{DIV}")[-1][-3:]

                result_dict = {}
                
                if slidename in list(results_df['slide']):
                    print (f">>> Result Already Exists for {slidename} ...")
                else:
                    if slideformat in args.wsi_formats:

                        print(f"--->>> Slide {slidename} ---<<<")

                        result_dict['id'] = ids_df.iloc[index].id
                        result_dict['subtype'] = subtype
                        result_dict['slide'] = slidename
                        result_dict['format'] = slideformat

                        expertdt = ExpertDT(crop_size=args.crop_size,
                                            crop_resize=args.crop_resize,
                                            level=args.wsi_level,
                                            overlap=args.overlap,
                                            batch_size=args.batch_size,
                                            weights_dir=args.weights_dir,
                                            tree_pair_dir=args.tree_pair_dir,
                                            pruned_threshold=args.node_prune_threshold)
                        
                        expertdt.predict(slide_dir=slide_dir)
                        expertdt.save_figs(note_dir=args.test_note,
                                            save_dir=args.figures_dir,
                                            id=ids_df.iloc[index].id,
                                            slidename=slidename)
                        
                        result_dict['node_pruned'] = expertdt.node_cert
                        result_dict['node_count_1'] = expertdt.node_count_1
                        result_dict['node_count_2'] = expertdt.node_count_2
                        result_dict['ccRCC'] = expertdt.subtype_counts['ccRCC']
                        result_dict['pRCC'] = expertdt.subtype_counts['pRCC']
                        result_dict['CHROMO'] = expertdt.subtype_counts['CHROMO']
                        result_dict['ONCOCYTOMA'] = expertdt.subtype_counts['ONCOCYTOMA']
                        result_dict['result'] = expertdt.max_subtype

                        results_list.append(result_dict)

            results_df = pd.concat([results_df, pd.DataFrame(results_list)], ignore_index=True)
            results_df.to_csv(results_path, index=False)