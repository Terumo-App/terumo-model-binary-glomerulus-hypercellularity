import random
from pathlib import Path
from typing import List
import re
import json


def generate_data_folders(dataset_name: str, data_folder: str = './data', raw_data_folder: str = 'raw'):

    data_dir = Path(data_folder)
    raw_dir = data_dir / raw_data_folder
    processed_dir = data_dir / 'processed'
    dataset_dir = processed_dir / dataset_name
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'

    processed_dir.mkdir(exist_ok=True)
    dataset_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    return raw_dir, train_dir, val_dir


def generate_binary_dataset(dataset_name: str, class_list: List[str], data_folder: str = './data', val_split: int = 0.2):
    def format_path(file_path):
            folders_splitted = re.split(r'\\|/', str(file_path))
            path_suffix = "/".join(folders_splitted[2:])
            return path_suffix
    
    def get_all_image_files(pathlib_root_folder):
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            img_regex = re.compile('|'.join(img_extensions), re.IGNORECASE)
            image_files = [f for f in pathlib_root_folder.glob('**/*') if f.is_file() and img_regex.search(f.suffix)]
            return image_files
    

    raw_data_dir, train_dir, val_dir = generate_data_folders(dataset_name, data_folder=data_folder)

    print("#"*60)
    for cls in class_list:

        class_dir = raw_data_dir / cls
        files = get_all_image_files(class_dir)

        random.seed(10)
        random.shuffle(files)
        len_files = len(files)
        val_size = int(len_files * val_split)

        print(f'Copying {cls} class..')
        print(f'Number of images: {len_files}')
        print(f'Test split percentage: {val_split*100}%')
        print(f'Number of images to train: {len_files - val_size}')
        print(f'Number of images to test: {val_size }',end='\n\n')

        f_train = []
        f_test = []
        for f in files[val_size:]:

            dst = train_dir / format_path(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
            f_train.append(str(f))

        for f in files[:val_size]:
            dst = val_dir / format_path(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(f.read_bytes())
            f_test.append(str(f))
        
        
        with open(f'{str(train_dir)}_{cls}_dataset.json', 'w') as f:
            json.dump({"images":f_train}, f)

        with open(f'{str(val_dir)}_{cls}_dataset.json', 'w') as f:
            json.dump({"images":f_test}, f)




if __name__ == "__main__":
    data_folder = './data'
    dataset_names = ['hiper_normal', 'memb_normal', 'scler_normal']
    classes_lists = [
         ['hipercellularity', 'normal'],
         ['membranous', 'normal'],
         ['sclerosis', 'normal']
         ]

    for dataset_name, classes_list in zip(dataset_names, classes_lists):
        generate_binary_dataset(dataset_name, classes_list, data_folder=data_folder)