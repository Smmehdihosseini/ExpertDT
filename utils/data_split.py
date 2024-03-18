import pandas as pd
import random
import json

def patient_split(data_df,
                  split_ratio,
                  random_seed,
                  load=True,
                  ids_dir=None,
                  default_patients = {"Train":
                                   {
                                       "ccRCC":[],
                                       "pRCC":[],
                                       "CHROMO":['HP20002300', 'HP19012316', 'HP20.2506'],
                                       "ONCOCYTOMA":['HP20.5602', 'HP18005453']
                                   },
                                   "Test":
                                   {
                                       "ccRCC":[],
                                       "pRCC":[],
                                       "CHROMO":[],
                                       "ONCOCYTOMA":[]
                                   },
                                   }):

    temp_data_df = data_df.copy()

    if not load:
        print(">>> Splitting Train/Test IDs ...")
        train_test_ids = {'Train':{}, 'Test': {}}

        for subtype, count in split_ratio['Train'].items():

            # Create empty trian and test lists
            train_test_ids['Train'][subtype] = []
            train_test_ids['Test'][subtype] = []

            # Check for no ROI patients to put in test list for each subtype
            st_temp = (data_df['subtype'] == subtype) & (~data_df['roi_exist'])
            if (st_temp).any():
                for index, patient in data_df[st_temp].iterrows():
                    train_test_ids['Test'][subtype].append(index)
                    temp_data_df = temp_data_df.drop(index=index)

            for id in default_patients['Train'][subtype]:
                train_test_ids['Train'][subtype].extend(temp_data_df[temp_data_df['id']==id].index.tolist())
                temp_data_df = temp_data_df.drop(index=temp_data_df[temp_data_df['id']==id].index[0])

            for id in default_patients['Test'][subtype]:
                train_test_ids['Test'][subtype].extend(temp_data_df[temp_data_df['id']==id].index.tolist())
                temp_data_df = temp_data_df.drop(index=temp_data_df[temp_data_df['id']==id].index[0])

            # Select randomly patients for train set
            st_df = temp_data_df[temp_data_df['subtype'] == subtype]
            st_train_indexes = st_df.sample(n=count-len(train_test_ids['Train'][subtype]), random_state=random_seed).index.tolist()

            # Allocate remaining patients to test set
            st_test_indexes = [idx for idx in st_df.index if idx not in st_train_indexes]

            # Add patient indexes to main train/test split JSON
            train_test_ids['Train'][subtype].extend(st_train_indexes)
            train_test_ids['Test'][subtype].extend(st_test_indexes)

        print(train_test_ids)

        # Save updated JSON to the the _info folder
        with open(ids_dir, 'w') as file:
            json.dump(train_test_ids, file, indent=4)

        print(f">>> Saving Split IDs to '{ids_dir}'")

    # Load train/test split JSON from _info folder
    else:

        try:
            with open(ids_dir, 'r') as file:
                train_test_ids = json.load(file)
                
            print(f">>> Loading Split IDs from '{ids_dir}'!")

        except:
            raise FileNotFoundError("")

    return train_test_ids