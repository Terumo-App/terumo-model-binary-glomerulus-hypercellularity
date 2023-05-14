import random
from pathlib import Path
from typing import List
import re
import json


def generate_data_folders(dataset_name: str, new_folder_name: str = 'processed', data_folder: str = './data', raw_data_folder: str = 'raw'):

    data_dir = Path(data_folder)
    raw_dir = data_dir / raw_data_folder
    processed_dir = data_dir / new_folder_name
    dataset_dir = processed_dir / dataset_name

    processed_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)

    return raw_dir, dataset_dir


def generate_binary_dataset(dataset_name: str, class_list: List[str], new_folder_name: str = 'processed', data_folder: str = './data', val_split: int = 0.2):
    def format_path(file_path):
            folders_splitted = re.split(r'\\|/', str(file_path))
            path_suffix = "/".join(folders_splitted[2:])
            return path_suffix
    
    def get_all_image_files(pathlib_root_folder):
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
            image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
            return image_files
    

    raw_data_dir, dataset_dir = generate_data_folders(
         dataset_name,
         new_folder_name=new_folder_name, 
         data_folder=data_folder
         )

    print("#"*60)
    f_list = []
    for cls in class_list:

        class_dir = raw_data_dir / cls
        files = get_all_image_files(class_dir)


        len_files = len(files)


        print(f'Copying {cls} class..')
        print(f'Number of images: {len_files}')


        for f in files:

            dst = dataset_dir / format_path(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
            f_list.append(str(f))


    with open(f'{str(dataset_dir)}_{cls}_dataset.json', 'w') as f:
        json.dump({"images":f_list}, f)




if __name__ == "__main__":
    data_folder = './data'
    dataset_names = ['hiper_outros','membran_outros', 'scler_outros', 'normal_outros']
    classes_lists = [
         ['hipercellularity', 'normal','membranous','sclerosis'],
         ['hipercellularity', 'normal','membranous','sclerosis'],
         ['hipercellularity', 'normal','membranous','sclerosis'],
         ['hipercellularity', 'normal','membranous','sclerosis']
         
         ]

    for dataset_name, classes_list in zip(dataset_names, classes_lists):
        generate_binary_dataset(
             dataset_name, 
             classes_list, 
             new_folder_name='binary',
             data_folder=data_folder)