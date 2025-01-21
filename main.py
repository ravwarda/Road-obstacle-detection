import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import xml.etree.ElementTree as ET
from shutil import copyfile

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerConfig
from segformer_train import segmentation_model_training
from image_processing import ade_palette, show_colormap_with_labels, show_colormap, extract_label
from cnn import cnn_model_training, SimpleCNN, sliding_window, visualize_detections, load_bounding_box_data, visualize_bounding_boxes


def image_segmentation(image, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    target_size = image.shape[:2]
    predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

    color_seg = np.zeros((predicted_segmentation_map.shape[0],
                        predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[predicted_segmentation_map == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = image * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    return predicted_segmentation_map, img


def adjust_bounding_boxes(annotation_path, output_annotation_path, x_offset, y_offset):
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


def compare_models(device, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").to(device)

    mask1, img1 = image_segmentation(image, model)
    ex1, _, _ = extract_label(image_rgb, mask1, 6)

    config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation(config=config).to(device)
    model.load_state_dict(torch.load("model_b0.pth", weights_only=True))

    mask2, img2 = image_segmentation(image, model)
    # show_colormap_with_labels(mask2)
    ex2, _, _ = extract_label(image_rgb, mask2, 0)

    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Obraz wejściowy")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Wynik segmentacji")
    
    plt.figure(figsize=(24, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Wynik segmentacji przed trenowaniem")
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Wynik segmentacji po trenowaniu")
    plt.subplot(2, 2, 3)
    plt.imshow(ex1)
    plt.title("Wyekstraktowany obraz drogi")
    plt.subplot(2, 2, 4)
    plt.imshow(ex2)
    plt.title("Wyekstraktowany obraz drogi")
    plt.show()

def main(seg_train=False, obj_train=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the model
    if seg_train:
        seg_model = segmentation_model_training(epochs=50, patience=5)
        torch.save(seg_model.state_dict(), "model_b0.pth")
    else:
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        seg_model = SegformerForSemanticSegmentation(config=config).to(device)
        seg_model.load_state_dict(torch.load("model_b0.pth", weights_only=True))

    if obj_train:
        det_model = cnn_model_training(epochs=50, patience=5)
        torch.save(det_model.state_dict(), 'cnn_model.pth')
    else:
        # Model load
        det_model = SimpleCNN()
        det_model.load_state_dict(torch.load('cnn_model.pth', weights_only=False))
        det_model = det_model.to(device)
        det_model.eval()

    # Test the model on a sample image

    # image = cv2.imread("dataset/Czech/images/Czech_003521.jpg")
    image = cv2.imread("dataset/Czech/images/Czech_002573.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask1, img1 = image_segmentation(image, seg_model)
    road_image, _, _ = extract_label(image_rgb, mask1, 0)

    # image_path = 'dataset/Czech/images/Czech_001697.jpg'
    # image = cv2.imread(image_path)
    windows = sliding_window(device, road_image, det_model, window_size=(64, 64), step_size=32)
    visualize_detections(road_image, windows, window_size=(64, 64))
    

if __name__ == "__main__":
    main(obj_train=True)



    # Test image segmentation
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image = cv2.imread("dataset/Czech/images/Czech_002573.jpg")
    # compare_models(device, image)



    # Train dataset segmentation
    # image_dir = "dataset/Czechtrain/images"
    # annotation_dir = "dataset/Czechtrain/annotations/xmls"
    # output_image_dir = "dataset/Czechtrain/processed_images"
    # output_annotation_dir = "dataset/Czechtrain/processed_annotations"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # model = SegformerForSemanticSegmentation(config=config).to(device)
    # model.load_state_dict(torch.load("model_b0.pth", weights_only=True))

    # # Assuming seg_model is already defined and loaded
    # process_images_and_annotations(image_dir, annotation_dir, output_image_dir, output_annotation_dir, model)



    # Nie wiem do czego to było, ale może jeszcze się przyda
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image = cv2.imread("dataset/segmentation_data/test/ae49bf6d-00000000.jpg")
    # config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # model = SegformerForSemanticSegmentation(config=config).to(device)
    # model.load_state_dict(torch.load("model_b0.pth", weights_only=True))

    # mask2, img2 = image_segmentation(image, model)
    # img_cut, mask_filtered = extract_label(image, mask2, 0)

    # fig = plt.figure(figsize=(24, 8))
    # plt.subplot(1, 3, 1)
    # plt.imshow(img2)
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask_filtered)
    # plt.subplot(1, 3, 3)
    # plt.imshow(img_cut)
    # plt.show()
