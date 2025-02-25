import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import xml.etree.ElementTree as ET
from main import image_segmentation


def ade_palette():
        """Part of ADE20K palette that maps each class to RGB values."""
        return [[120, 120, 120], [224, 5, 255], [4, 250, 7], [235, 255, 7],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]


def show_colormap(image, colormap):
    """Display image with overlayed colormap."""

    blended = cv2.addWeighted(image, 0.5, colormap, 0.5, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(blended_rgb)
    plt.axis('off')
    plt.show()


def show_colormap_with_labels(mask):
    """Display only the picture mask with class labels."""

    # Create a blank image with the same dimensions as the mask
    labeled_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    palette = np.array(ade_palette())
    
    # Apply the palette to the mask
    for label, color in enumerate(palette):
        labeled_image[mask == label] = color
    
    labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(labeled_image_rgb)
    plt.axis('off')
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label < len(palette):
            y, x = np.where(mask == label)
            if len(y) > 0 and len(x) > 0:
                plt.text(x[0], y[0], str(label), color='white', fontsize=12, backgroundcolor='black')
    plt.show()


def extract_label_with_filter(image, mask, label):
    """Extract the class with the given label from the image and mask, by cropping the image to class dimentions and setting all remaining pixels, that don't belong to class, to black.
    The function also filters out small disconnected regions that are smaller than 3% of the biggest region. (Not used because flood fill is time consuming)"""

    image = image.copy()
    regions = []

    # Find all pixels with the given label
    label_pixels = np.where(mask == label)

    # Create a blank mask to keep track of visited pixels
    visited = np.zeros_like(mask, dtype=bool)

    # Define a function to perform flood fill
    def flood_fill(y, x, label):
        stack = [(y, x)]
        region = []
        while stack:
            cy, cx = stack.pop()
            if visited[cy, cx] or mask[cy, cx] != label:
                continue
            visited[cy, cx] = True
            region.append((cy, cx))
            for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]:
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and not visited[ny, nx]:
                    stack.append((ny, nx))
        return region

    # Iterate over all label pixels and perform flood fill
    for y, x in zip(*label_pixels):
        if not visited[y, x]:
            region = flood_fill(y, x, label)
            if region:
                regions.append(np.array(region))

    # Calculate the size of each region
    region_sizes = [len(region) for region in regions]
    max_region_size = max(region_sizes)

    # Filter regions that are bigger than 3% of the biggest one
    filtered_regions = [region for region in regions if len(region) > 0.03 * max_region_size]

    # Create a blank mask for the filtered regions
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    # Set the pixels in the filtered regions to 1
    for region in filtered_regions:
        for y, x in region:
            filtered_mask[y, x] = 1

    # Find the bounding box of the filtered mask
    y_indices, x_indices = np.where(filtered_mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No regions found

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Crop the image and mask to the bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = filtered_mask[y_min:y_max+1, x_min:x_max+1]

    image[filtered_mask == 0] = [0, 0, 0]

    return cropped_image, cropped_mask, (y_min, y_max, x_min, x_max)

def extract_label(image, mask, label):
    """Extract the class with the given label from the image and mask, by cropping the image to class dimentions and setting all remaining pixels, that don't belong to class, to black."""

    image = image.copy()

    # Find all pixels with the given label
    filtered_mask = np.where(mask == label, 1, 0)

    # Find the bounding box of the filtered mask
    y_indices, x_indices = np.where(filtered_mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No regions found

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Crop the image and mask to the bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = filtered_mask[y_min:y_max+1, x_min:x_max+1]

    image[filtered_mask == 0] = [0, 0, 0]

    return cropped_image, cropped_mask, (y_min, y_max, x_min, x_max)\


def adjust_bounding_boxes(annotation_path, output_annotation_path, x_offset, y_offset):
    """Adjust object bounding boxes in annotation file by subtracting the given offsets."""

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text) - x_offset
        ymin = int(bndbox.find('ymin').text) - y_offset
        xmax = int(bndbox.find('xmax').text) - x_offset
        ymax = int(bndbox.find('ymax').text) - y_offset

        bndbox.find('xmin').text = str(max(0, xmin))
        bndbox.find('ymin').text = str(max(0, ymin))
        bndbox.find('xmax').text = str(max(0, xmax))
        bndbox.find('ymax').text = str(max(0, ymax))

    tree.write(output_annotation_path)

def process_images_and_annotations(image_dir, annotation_dir, output_image_dir, output_annotation_dir, seg_model):
    """Process all images in the given directory by segmenting the road and adjusting the annotations."""
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_annotation_dir):
        os.makedirs(output_annotation_dir)

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_name)
            annotation_path = os.path.join(annotation_dir, image_name.replace('.jpg', '.xml'))

            # Read and process the image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask, segmented_image = image_segmentation(image, seg_model)
            road_image, _, extract_bndbox = extract_label(image_rgb, mask, 0)
            x_offset = extract_bndbox[2]
            y_offset = extract_bndbox[0]

            # Save the processed image
            output_image_path = os.path.join(output_image_dir, image_name)
            cv2.imwrite(output_image_path, cv2.cvtColor(road_image, cv2.COLOR_RGB2BGR))

            # Adjust and save the annotation
            output_annotation_path = os.path.join(output_annotation_dir, image_name.replace('.jpg', '.xml'))
            adjust_bounding_boxes(annotation_path, output_annotation_path, x_offset, y_offset)


def load_bounding_box_data(xml_file):
    """Process the annotation file and return the image information with a list of objects with bounding boxes."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})

    return filename, width, height, depth, objects


def visualize_bounding_boxes(image_path, objects):
    """Visualize target bounding boxes of objects on the input image."""
    image = cv2.imread(image_path)
    for obj in objects:
        name = obj['name']
        xmin, ymin, xmax, ymax = obj['bbox']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()