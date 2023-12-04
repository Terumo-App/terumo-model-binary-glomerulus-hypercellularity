import random
from pathlib import Path
import sys
from typing import List
import re
import json
from tqdm import tqdm


def generate_data_folders(dataset_name: str, new_folder_name: str = 'processed', data_folder: str = './data', raw_data_folder: str = 'raw'):
    """
        Given class name A, returns data/raw and data/processed/A 
    """

    data_dir = Path(data_folder)
    raw_dir = data_dir / raw_data_folder
    processed_dir = data_dir / new_folder_name
    dataset_dir = processed_dir / dataset_name

    processed_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    return raw_dir, dataset_dir


def generate_binary_dataset(dataset_name: str, 
                            class_list: List[str],
                            target_class: str,
                            new_folder_name: str = 'processed', 
                            data_folder: str = './data'):
    def format_path(file_path):
        """
            data/raw/A/[...]/fname -> A/[...]/fname
        """
        folders_splitted = re.split(r'\\|/', str(file_path))
        path_suffix = "/".join(folders_splitted[2:])
        return path_suffix
    
    def get_all_image_files(pathlib_root_folder):
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
        image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
        return image_files
    

    raw_data_dir, dataset_dir = generate_data_folders(
         dataset_name,
         new_folder_name=new_folder_name, 
         data_folder=data_folder
         )

    print("#"*80)
    f_list = []
    for cls in tqdm(class_list, total=len(class_list)):
        # data/raw/A/[...]/filename
        origin_class_dir = raw_data_dir / cls
        files = get_all_image_files(origin_class_dir)

        print(f'Copying {cls} class..')
        print(f'Number of images: {len(files)}')

        if cls == target_class:
            ## -> data/processed/A/A
            target_dir = dataset_dir
        else:
            ## -> data/processed/A/_others
            ## the underscore guarantees this appears first and becomes class 0 (negative class)
            target_dir = dataset_dir / "_others"
        for f in tqdm(files, total=len(files)):
            dst = target_dir / format_path(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
            f_list.append(str(f))

    with open(f'{str(dataset_dir)}_{cls}_dataset.json', 'w') as f:
        json.dump({"images":f_list}, f)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_folder = sys.argv[1]
    else:
        data_folder = './data'
    
    classes_list = ['Hypercelularidade', 'Normal','Membranous', 'Sclerosis', 'Crescent', 'Podocitopatia']

    for target_cls in classes_list:
        generate_binary_dataset(
             dataset_name='binary_' + target_cls, 
             class_list=classes_list,
             target_class=target_cls,
             new_folder_name='binary',
             data_folder=data_folder)