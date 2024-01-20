import json

from pathlib import Path

# Set the path to the root directory of the image dataset
data_dir = Path('./raw/')

# Create a dictionary to hold the image data
json_data = {'data': []}

# Loop over the image files in the data directory and its subdirectories
for image_file in data_dir.glob('**/*.jpg'):
    # Get the path to the image file
    image_path = image_file.as_posix()
    index = image_path.rfind('/')
    file_name = image_path[index+1:]

    # Get the class label from the name of the parent directory
    subclass_label = image_file.parent.name
    class_label = image_file.parts[1]
    
    # Check if the image file already exists in the image data list
    existing_data = next((d for d in json_data['data'] if file_name in d['image_path'] ), None)
    
    # If the image file doesn't exist in the image data list, add it
    if existing_data is None:
        json_data['data'].append({
            'image_path': image_path,
            'subclass_label': [subclass_label],
            'class_labels': [class_label]
        })
    # If the image file already exists in the image data list, add the class label to the list
    else:
        existing_data['subclass_label'].append(subclass_label)

# Write the JSON data to a file
with open('dataset.json', 'w') as f:
    json.dump(json_data, f)