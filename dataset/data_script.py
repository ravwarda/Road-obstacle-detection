import os
import xml.etree.ElementTree as ET
import random

def has_d40_object(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'D40':
            return True
    return False

def get_all_files(xml_folder, image_folder):
    xml_files = []
    image_files = []
    for root, _, files in os.walk(xml_folder):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    return xml_files, image_files

def main(xml_folder, image_folder):
    xml_files, image_files = get_all_files(xml_folder, image_folder)
    
    # Identify files with D40 objects
    d40_files = [xml_file for xml_file in xml_files if has_d40_object(xml_file)]
    d40_image_files = [os.path.join(image_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.jpg') for xml_file in d40_files if os.path.exists(os.path.join(image_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'))]
    
    # Identify remaining files
    remaining_files = [xml_file for xml_file in xml_files if xml_file not in d40_files]
    remaining_image_files = [os.path.join(image_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.jpg') for xml_file in remaining_files if os.path.exists(os.path.join(image_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'))]
    
    # Randomly select 10% of the remaining files
    num_to_keep = int(0.1 * len(remaining_files))
    random_files = random.sample(remaining_files, num_to_keep)
    random_image_files = [os.path.join(image_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.jpg') for xml_file in random_files if os.path.exists(os.path.join(image_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'))]
    
    # Combine files to keep
    files_to_keep = set(d40_files + random_files)
    image_files_to_keep = set(d40_image_files + random_image_files)
    
    # Delete other files
    for xml_file in xml_files:
        if xml_file not in files_to_keep:
            os.remove(xml_file)
    for image_file in image_files:
        if image_file not in image_files_to_keep:
            os.remove(image_file)

# Example usage
xml_folder = 'dataset/Czechtrain/annotations/xmls'
image_folder = 'dataset/Czechtrain/images'
main(xml_folder, image_folder)