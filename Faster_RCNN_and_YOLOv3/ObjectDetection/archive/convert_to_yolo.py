import os
import pandas as pd
from PIL import Image
import numpy as np

# Create directory for YOLO annotations
output_dir = 'yolo_annotations'
os.makedirs(output_dir, exist_ok=True)

# First, read the annotation file and parse it correctly
annotations = []
with open('annotation.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 6:  # Ensure we have all expected parts
            image_path, x_min, y_min, x_max, y_max, class_name = parts
            annotations.append({
                'image_path': image_path,
                'x_min': int(x_min),
                'y_min': int(y_min),
                'x_max': int(x_max),
                'y_max': int(y_max),
                'class_name': class_name
            })

# Extract unique classes and create classes.txt
unique_classes = sorted(set(ann['class_name'] for ann in annotations))
with open('classes.txt', 'w') as f:
    for class_name in unique_classes:
        f.write(f"{class_name}\n")

print(f"Created classes.txt with {len(unique_classes)} classes")

# Create mapping from class name to class id
class_dict = {class_name: i for i, class_name in enumerate(unique_classes)}

# Group annotations by image
image_annotations = {}
for ann in annotations:
    if ann['image_path'] not in image_annotations:
        image_annotations[ann['image_path']] = []
    image_annotations[ann['image_path']].append(ann)

# Create train.txt file
with open('train.txt', 'w') as f:
    for image_path in image_annotations.keys():
        f.write(f"{os.path.abspath(image_path)}\n")

print(f"Created train.txt with {len(image_annotations)} images")

# Process each image and create YOLO format annotations
for image_path, anns in image_annotations.items():
    try:
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_path, 'w') as f:
            for ann in anns:
                # Get class_id
                class_id = class_dict[ann['class_name']]
                
                # Calculate bounding box in YOLO format
                x_min = ann['x_min']
                y_min = ann['y_min']
                x_max = ann['x_max']
                y_max = ann['y_max']
                
                # Calculate center coordinates and dimensions
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Write to file in YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

print(f"Converted annotations for {len(image_annotations)} images to YOLO format")