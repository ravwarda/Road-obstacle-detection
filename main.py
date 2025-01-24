import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import xml.etree.ElementTree as ET
import threading
from shutil import copyfile
import tempfile
import streamlit as st
import time

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


def execute_models(device, segmentation_model, detection_model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask1, img1 = image_segmentation(image, segmentation_model)
    road_image, _, (y_min, y_max, x_min, x_max) = extract_label(image_rgb, mask1, 0)

    windows = sliding_window(device, road_image, detection_model, window_size=(64, 64), step_size=32)
    processed_image = visualize_detections(image_rgb, windows, window_size=(64, 64), offset=(x_min, y_min))
    
    return processed_image, windows, (x_min, y_min)

def visualise_process(device, segmentation_model, detection_model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask1, img1 = image_segmentation(image, segmentation_model)
    road_image, _, (y_min, y_max, x_min, x_max) = extract_label(image_rgb, mask1, 0)

    windows = sliding_window(device, road_image, detection_model, window_size=(64, 64), step_size=32)

    processed_road = visualize_detections(road_image, windows, window_size=(64, 64))
    processed_image = visualize_detections(image_rgb, windows, window_size=(64, 64), offset=(x_min, y_min))
    
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 5, 1)
    plt.imshow(image_rgb)
    plt.title("Obraz wejściowy")
    plt.subplot(1, 5, 2)
    plt.imshow(img1)
    plt.title("Wynik segmentacji")
    plt.subplot(1, 5, 3)
    plt.imshow(road_image)
    plt.title("Wycięty obraz drogi")
    plt.subplot(1, 5, 4)
    plt.imshow(processed_road)
    plt.title("Wynik detekcji")
    plt.subplot(1, 5, 5)
    plt.imshow(processed_image)
    plt.title("Efekt końcowy")
    plt.show()


def process_video_rt(video_path, device, segmentation_model, detection_model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    processed_frame = None
    windows = None
    offset = (0, 0)
    lock = threading.Lock()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps

    def process_frame(frame):
        nonlocal processed_frame, windows, offset
        processed_frame, windows, offset = execute_models(device, segmentation_model, detection_model, frame)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Execute models once per 24 frames in a separate thread
        if frame_count % 24 == 0:
            thread = threading.Thread(target=process_frame, args=(frame,))
            thread.start()
        else:
            # Use the previously processed frame
            if processed_frame is not None:
                processed_frame = visualize_detections(frame, windows, window_size=(64, 64), offset=offset)

        # Display the frame
        if processed_frame is not None:
            cv2.imshow('Processed Video', processed_frame)

        # Calculate the time to wait to match the original frame rate
        elapsed_time = time.time() - start_time
        wait_time = max(1, int((frame_time - elapsed_time) * 1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path, output_path, device, segmentation_model, detection_model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    processed_frame = None
    windows = None
    offset = (0, 0)

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the processed video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Execute models once per 2 frames
        if frame_count % 3 == 0:
            processed_frame, windows, offset = execute_models(device, segmentation_model, detection_model, frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        else:
            # Use the previously processed frame
            if processed_frame is not None:
                processed_frame = visualize_detections(frame, windows, window_size=(64, 64), offset=offset)

        # Write the processed frame to the output video
        if processed_frame is not None:
            out.write(processed_frame)

        frame_count += 1

    cap.release()
    out.release()

def streamlit_function():
    st.title("Video Processing with Segformer and CNN")

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Load models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        seg_model = SegformerForSemanticSegmentation(config=config).to(device)
        seg_model.load_state_dict(torch.load("model_b0.pth", weights_only=True))

        det_model = SimpleCNN()
        det_model.load_state_dict(torch.load('cnn_model.pth', weights_only=False))
        det_model = det_model.to(device)
        det_model.eval()

        # Process the video
        st.text("Processing video...")
        output_path = "processed_video.mp4"
        process_video(tfile.name, output_path, device, seg_model, det_model)

        # Display the processed video
        st.text("Processed video:")
        st.video(output_path)


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
        det_model = cnn_model_training(epochs=100, patience=5)
        torch.save(det_model.state_dict(), 'cnn_model.pth')
    else:
        # Model load
        det_model = SimpleCNN()
        det_model.load_state_dict(torch.load('cnn_model.pth', weights_only=False))
        det_model = det_model.to(device)
        det_model.eval()

    # Test the model on a sample image
    image = cv2.imread("dataset/Czech/images/Czech_003521.jpg")
    image2 = cv2.imread("dataset/Czech/images/Czech_002573.jpg")
    image3 = cv2.imread("dataset/Czech/images/Czech_000082.jpg")
    visualise_process(device, seg_model, det_model, image)
    visualise_process(device, seg_model, det_model, image2)
    visualise_process(device, seg_model, det_model, image3)

    # Process video
    # video_path = "dataset/videos/video_short.mp4"
    # process_video_rt(video_path, device, seg_model, det_model)

    # video_path = "dataset/videos/video_short.mp4"
    # output_path = "dataset/videos/processed_video.mp4"
    # process_video(video_path, output_path, device, seg_model, det_model)


if __name__ == "__main__":
    main()
    # streamlit_function()
