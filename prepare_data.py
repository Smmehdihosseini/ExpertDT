import os
import platform
import random
import pandas as pd
import json
import argparse
import glob
import time
import numpy as np

# CPU Thread Limitations for Server
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from utils.data_split import patient_split
from wsi_manager.crop import CropIndexer
from dataset.balancer import Balancer

if platform.system() == "Windows":
     DEFAULT_PATH = "_info/default_path_local.json"
elif platform.system() == "Linux":
     DEFAULT_PATH = "_info/default_path_server.json"

with open(DEFAULT_PATH, 'r') as file:
            default_dirs = json.load(file)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Prepare Data for Binary Classification of ExpertDT")

    # Directory Arguments
    parser.add_argument("--root_dir", type=str, default=default_dirs['root_dir'], help="Root WSIs Folder")
    parser.add_argument("--ids_dir", type=str, default=default_dirs['ids_dir'], help="Path to Patient IDs File")
    parser.add_argument("--split_dir", type=str, default=default_dirs['split_dir'], help="Patient Level Train/Test Split")
    parser.add_argument("--split_ids_dir", type=str, default=default_dirs['split_ids_dir'], help="Split Data IDs")
    parser.add_argument("--train_df_dir", type=str, default=default_dirs['train_df_dir'], help="Training DataFrame")
    parser.add_argument("--tree_pair_dir", type=str, default=default_dirs['tree_pair_dir'], help="Class Labels in Different Stages of Tree")
    parser.add_argument("--stage_df_dir", type=str, default=default_dirs['stage_df_dir'], help="Path to Save Root, Node, Leaf1 and Leaf2 Dataset")

    # Verbose Argument
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose Print Details")

    # Load ID Split List
    parser.add_argument("--load_id_list", type=bool, default=True, help="Load ID Split List, False Results to Create and Save New List File")
    parser.add_argument("--load_train_df", type=bool, default=False, help="Load Training DataFrame, False Results to Create and Save New DataFrame")

    # Cropping Arguments
    parser.add_argument("--crop_size", type=int, default=1000, help="Patches Crop Size During Cropping Stage")
    parser.add_argument("--wsi_level", type=int, default=0, help="Whole Slide Image Level of Magnification During Cropping Stage")
    parser.add_argument("--overlap", type=int, default=1, help="Patches Overlap During Cropping Stage; 1: No Overlap, 2: 50%, etc.")
    parser.add_argument("--wsi_formats", nargs=3, type=str, default=['scn', 'svs', 'tif'], help="Possible Formats for Whole Slide Image")
    parser.add_argument("--balance_method", type=str, default="undersample", help="Balance Method for Each Stage of Tree")

    parser.add_argument("--random_seed", type=int, default=42, help="Random Seed for Reproducibility")

    args = parser.parse_args()

    random.seed(args.random_seed)
    
    if args.verbose:
        print("\n>>> Loading Patient IDs ...\n")

    ids_df = pd.read_csv(args.ids_dir)

    # Drop patients with no ROI and label info
    ids_df = ids_df.dropna()
    ids_df.roi_exist = ids_df.roi_exist.astype(bool)

    if args.verbose:
        print(ids_df.info())

    if args.verbose:
        print("\n>>> Loading Patient-level Split Ratio ...\n")

    try:
        with open(args.split_dir, 'r') as file:
            split_ratio = json.load(file)
    except:
        raise FileNotFoundError("No Split Ratio JSON File in './_info/' folder.")
    

    if args.verbose:
        print(pd.DataFrame(split_ratio))

    if args.verbose:
        print("\n>>> Loading/Creating Train Test ID Split File ...")

    train_test_ids = patient_split(data_df=ids_df,
                                   split_ratio=split_ratio,
                                   random_seed=args.random_seed,
                                   load=args.load_id_list,
                                   ids_dir=args.split_ids_dir)


    if not args.load_train_df:
        train_list = []

        # Creating Training Crops DataFrame
        for subtype in train_test_ids['Train'].keys():

            if args.verbose:
                print(f"\n")
                print(f"--"*15)
                print(f">>> Preparing data for {subtype}")
                print(f"--"*15)
                print(f"\n")

            for index in train_test_ids['Train'][subtype]:

                t_start = time.time()

                if ids_df.iloc[index].annot_type == 'SLIDE':
                    
                    if args.verbose:
                        print(f'>>> Get {ids_df.iloc[index].id} Image Patches ...')

                    # Read subtype patient correspondence file
                    patient_corr_path = f'{args.root_dir}\\{subtype}\\Annotations\\{subtype}_patients_correspondence.xlsx'
                    patients_corr = pd.read_excel(patient_corr_path, engine='openpyxl')
                    
                    # Check if there are multiple annotation slides for the patient
                    annot_slide_no = str(patients_corr[patients_corr['PATIENT']==ids_df.iloc[index].id]['ID'].values[0]).split('-')

                    # Create list of annotations to read crops
                    if len(annot_slide_no)>1:
                        annotations = list(str(i) for i in range(int(annot_slide_no[0]), int(annot_slide_no[1])+1))
                    else:
                        annotations = annot_slide_no

                    slide_section = CropIndexer(type='SLIDE', crop_size=args.crop_size, overlap=args.overlap)

                    # Crop all annotations and add them to train data list
                    for annot in annotations:
                        for fmt in args.wsi_formats:
                            slide_dir = f'{args.root_dir}\\{subtype}\\Annotations\\{annot}.{fmt}'
                            if os.path.exists(slide_dir):
                                crop_indexes = slide_section.crop(slide_dir=slide_dir)
                                break

                        # Show number of added crops
                        if args.verbose:
                            print(f"--------> +{len(crop_indexes)} Crops")

                        # Add metadata to each crop for further analysis
                        for crop in crop_indexes:

                            temp_crop = {}
                            temp_crop['subtype'] = subtype
                            temp_crop['annot_type'] = 'SLIDE'
                            temp_crop['id'] = ids_df.iloc[index].id
                            temp_crop['path'] = slide_dir
                            temp_crop['is_tumor'] = True
                            temp_crop['type'] = subtype
                            temp_crop['top'] = crop['top']
                            temp_crop['left'] = crop['left']
                            temp_crop['size'] = crop['size']

                            train_list.append(temp_crop)

                elif ids_df.iloc[index].annot_type == 'XML':
                    
                    if args.verbose:
                        print(f'>>> Get {ids_df.iloc[index].id} Image Patches ...')

                    subtype_path = f'{args.root_dir}\\{subtype}'

                    # Check for slide in main directory
                    if glob.glob(f"{subtype_path}\\*{ids_df.iloc[index].id}*"):
                        for slide_dir in glob.glob(f"{subtype_path}\\*{ids_df.iloc[index].id}*"):

                            # Creating path to corresponding XML annotation
                            slidename = slide_dir.split("\\")[-1][:-4]
                            xml_dir = f"{subtype_path}\\{subtype}_xml\\{slidename}.xml"

                            # Cropping slide into patches
                            slide_section = CropIndexer(type='XML', crop_size=args.crop_size, overlap=args.overlap)
                            crop_indexes = slide_section.crop(slide_dir=slide_dir, xml_dir=xml_dir)     

                            # Show number of added crops
                            if args.verbose:
                                print(f"--------> +{len(crop_indexes)} Crops")

                            # Add metadata to each crop for further analysis
                            for crop in crop_indexes:

                                temp_crop = {}
                                temp_crop['subtype'] = subtype
                                temp_crop['annot_type'] = 'XML'
                                temp_crop['id'] = ids_df.iloc[index].id
                                temp_crop['path'] = slide_dir
                                temp_crop['is_tumor'] = True if crop['label']=='tumor' else False
                                temp_crop['type'] = subtype if crop['label']=='tumor' else crop['label'].strip()
                                temp_crop['top'] = crop['top']
                                temp_crop['left'] = crop['left']
                                temp_crop['size'] = crop['size']

                                train_list.append(temp_crop)                     

                    # Check for slide in pre directory
                    elif glob.glob(f"{args.root_dir}\\pre\\{subtype}\\*{ids_df.iloc[index].id}*"):

                        if args.verbose:
                            print("---- Checking 'pre' Folder ...")

                        for slide_dir in glob.glob(f"{args.root_dir}\\pre\\{subtype}\\*{ids_df.iloc[index].id}*"):

                            # Creating path to corresponding XML annotation
                            slidename = slide_dir.split("\\")[-1][:-4]
                            xml_dir = f"{args.root_dir}\\pre\\{subtype}\\{subtype}_xml\\{slidename}.xml"

                            slide_section = CropIndexer(type='XML', crop_size=args.crop_size, overlap=args.overlap)
                            crop_indexes = slide_section.crop(slide_dir=slide_dir, xml_dir=xml_dir)

                            # Show number of added crops
                            if args.verbose:
                                print(f"--------> +{len(crop_indexes)} Crops")

                            # Add metadata to each crop for further analysis
                            for crop in crop_indexes:

                                temp_crop = {}

                                temp_crop['subtype'] = subtype
                                temp_crop['annot_type'] = 'XML'
                                temp_crop['id'] = ids_df.iloc[index].id
                                temp_crop['path'] = slide_dir
                                temp_crop['is_tumor'] = True if crop['label']=='tumor' else False
                                temp_crop['type'] = subtype if crop['label']=='tumor' else crop['label'].strip()
                                temp_crop['top'] = crop['top']
                                temp_crop['left'] = crop['left']
                                temp_crop['size'] = crop['size']

                                train_list.append(temp_crop)           

                t_end = time.time()

                if args.verbose:
                    print(f"+++ Finished Cropping '{ids_df.iloc[index].id}' Slides in {round(t_end-t_start, 2)}!")
    
        if args.verbose:
            print("\n>>> Saving Training DataFrame ...")

        train_df = pd.DataFrame(train_list)
        train_df.to_csv(args.train_df_dir, index=False)

    else:
        try:
            train_df = pd.read_csv(args.train_df_dir)
        except:
            raise FileNotFoundError(f"No Training .csv File '{args.train_df_dir}'")
        

    if args.verbose:
        print(">>> Loading Tree Pair Classes ...")

    try:
        with open(args.tree_pair_dir, 'r') as file:
            tree_pair_dict = json.load(file)
    except:
        raise FileNotFoundError(f"No Tree Pair .json File '{args.tree_pair_dir}'")

    if args.verbose:
        print(">>> Handling Dataset Imbalance Labels  ...")

    data_balancer = Balancer(method=args.balance_method, random_state=args.random_seed)
    balanced_dfs = data_balancer.apply(train_df, tree_pair_dict)

    if args.verbose:
        print(f">>> Saving Balanced Datasets to '{args.stage_df_dir}' ...")

    # Manage labels for binary classification in different stages
    balanced_dfs['Root']['label'] = np.where(balanced_dfs['Root']['is_tumor'] == False, 0, 1)
    balanced_dfs['Node']['label'] = np.where(balanced_dfs['Node']['type'].isin(tree_pair_dict['Node']['class_0']), 0, 1)
    balanced_dfs['Leaf1']['label'] = np.where(balanced_dfs['Leaf1']['type'].isin(tree_pair_dict['Leaf1']['class_0']), 0, 1)
    balanced_dfs['Leaf2']['label'] = np.where(balanced_dfs['Leaf2']['type'].isin(tree_pair_dict['Leaf2']['class_0']), 0, 1)

    for stage in tree_pair_dict.keys():

        # Shuffle Data Before Saving Them
        balanced_dfs[stage] = balanced_dfs[stage].sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        balanced_dfs[stage].to_csv(f"{args.stage_df_dir}/{stage}_data.csv", index=False)