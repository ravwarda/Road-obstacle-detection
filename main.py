import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from transformers import SegformerForSemanticSegmentation, SegformerConfig
from segformer_train import segmentation_model_training, image_segmentation
from image_processing import show_colormap_with_labels, show_colormap, extract_label
from cnn import cnn_model_training, SimpleCNN, sliding_window, visualize_detections


def compare_models(device, image):
    "Test function to compare default segformer model and the finetuned one."
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)

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
    """Perform full detection on the input image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask1, img1 = image_segmentation(image, segmentation_model)
    road_image, _, (y_min, y_max, x_min, x_max) = extract_label(image_rgb, mask1, 0)

    windows = sliding_window(device, road_image, detection_model, window_size=(64, 64), step_size=32)
    processed_image = visualize_detections(image_rgb, windows, window_size=(64, 64), offset=(x_min, y_min))
    
    return processed_image, windows, (x_min, y_min)

def visualise_process(device, segmentation_model, detection_model, image):
    """Visualize the process of image segmentation and object detection, on the input image."""
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


def process_video(video_path, output_path, device, segmentation_model, detection_model):
    """Perform full detection on a video file and save the processed video."""
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

        # Execute models once per 3 frames
        if frame_count % 3 == 0:
            processed_frame, windows, offset = execute_models(device, segmentation_model, detection_model, frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        else:
            if processed_frame is not None:
                processed_frame = visualize_detections(frame, windows, window_size=(64, 64), offset=offset)

        # Write the processed frame to the output video
        if processed_frame is not None:
            out.write(processed_frame)

        frame_count += 1

    cap.release()
    out.release()


def main(seg_train=False, obj_train=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the model
    if seg_train:
        root_dir = "dataset/segmentation_data"
        seg_model = segmentation_model_training(root_dir, epochs=50, patience=5)
        torch.save(seg_model.state_dict(), "model_b0.pth")
    else:
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        seg_model = SegformerForSemanticSegmentation(config=config).to(device)
        seg_model.load_state_dict(torch.load("model_b0.pth", weights_only=True, map_location=torch.device('cpu')))

    if obj_train:
        annotations_path = 'dataset/Czechtrain/processed_annotations'
        images_path = 'dataset/Czechtrain/processed_images'
        det_model = cnn_model_training(images_path, annotations_path, epochs=100, patience=5)
        torch.save(det_model.state_dict(), 'cnn_model.pth')
    else:
        # Model load
        det_model = SimpleCNN()
        det_model.load_state_dict(torch.load('cnn_model.pth', weights_only=False, map_location=torch.device('cpu')))
        det_model = det_model.to(device)
        det_model.eval()

    # Test the model on a sample image
    image = cv2.imread("dataset/Czech/images/Czech_003521.jpg")
    visualise_process(device, seg_model, det_model, image)

    # Process video
    video_path = "dataset/videos/video_short.mp4"
    output_path = "dataset/videos/processed_video.mp4"
    process_video(video_path, output_path, device, seg_model, det_model)


if __name__ == "__main__":
    main()
