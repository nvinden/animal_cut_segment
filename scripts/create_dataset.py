# Description: This file is used to create the splits for the coco formatted cuts.
# Command Line Arguments: --data_path, --split, --output_path, --with_loins
# defaults: data_path = '../data/coco_formatted_cuts', split = 80:10:10 (80 for train, 10 for test and 10 for val)), 
#   output_path = '../data/coco_formatted_cuts_split', with_loins = False

import numpy as np
import argparse
import os

# Setting numpy seed
np.random.seed(41)

# Getting the directory path of this file's parent directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(data_path, split, output_path, with_loins):
    assert os.path.exists(data_path), "Data path does not exist"

    ttv_splits = [int(value) for value in split.split(":")]

    # Create the output folder and subfolders
    os.makedirs(output_path, exist_ok = True)
    os.makedirs(os.path.join(output_path, "train"), exist_ok = True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok = True)
    os.makedirs(os.path.join(output_path, "val"), exist_ok = True)  
    if with_loins == False: os.makedirs(os.path.join(output_path, "loins"), exist_ok = True)   

    # Get the list of all the files in the data_path
    file_list = [f.replace(".json", "") for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and ".json" in f]

    # Putting loins in a seperate list if with_loins is False
    if with_loins == False:
        loins = [f for f in file_list if "Loin" in f]
        file_list = [f for f in file_list if "Loin" not in f]

    # Shuffle the file list with numpy
    np.random.shuffle(file_list)

    # Splitting data into train, test and val with the ttv_splits ratio
    train_split = file_list[:int(len(file_list)*ttv_splits[0]/100)]
    test_split = file_list[int(len(file_list)*ttv_splits[0]/100):int(len(file_list)*(ttv_splits[0]+ttv_splits[1])/100)]
    val_split = file_list[int(len(file_list)*(ttv_splits[0]+ttv_splits[1])/100):]

    # Copying the files to the output folder
    for file in train_split:
        print(f"Copying {file} to train folder")
        os.system(f"cp {data_path}/{file}.json {output_path}/train/{file}.json")
        os.system(f"cp {data_path}/{file}.JPG {output_path}/train/{file}.JPG")
    for file in test_split:
        print(f"Copying {file} to test folder")
        os.system(f"cp {data_path}/{file}.json {output_path}/test/{file}.json")
        os.system(f"cp {data_path}/{file}.JPG {output_path}/test/{file}.JPG")
    for file in val_split:
        print(f"Copying {file} to val folder")
        os.system(f"cp {data_path}/{file}.json {output_path}/val/{file}.json")
        os.system(f"cp {data_path}/{file}.JPG {output_path}/val/{file}.JPG")

    if with_loins == False:
        for file in loins:
            print(f"Copying {file} to loins folder")
            os.system(f"cp {data_path}/{file}.json {output_path}/loins/{file}.json")
            os.system(f"cp {data_path}/{file}.JPG {output_path}/loins/{file}.JPG")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=f'{PARENT_DIR}/data/coco_formatted_cuts', help='Path to the data folder')
    parser.add_argument('--split', type=str, default="80:10:10", help='Split ratio for train, test and val')
    parser.add_argument('--output_path', type=str, default=f'{PARENT_DIR}/data/coco_formatted_cuts_split', help='Path to the output folder')
    parser.add_argument('--with_loins', type=bool, default=False, help='Whether to include loins in the dataset')
    args = parser.parse_args()

    main(args.data_path, args.split, args.output_path, args.with_loins)