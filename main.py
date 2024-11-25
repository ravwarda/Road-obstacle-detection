import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn
import evaluate


def segmentation_model_training(epochs=10):
    class SemanticSegmentationDataset(Dataset):
        """Image (semantic) segmentation dataset."""

        def __init__(self, root_dir, image_processor, train=True):
            """
            Args:
                root_dir (string): Root directory of the dataset containing the images + annotations.
                image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
                train (bool): Whether to load "training" or "validation" images + annotations.
            """
            self.root_dir = root_dir
            self.image_processor = image_processor
            self.train = train

            # sub_path = "training" if self.train else "validation"
            self.img_dir = os.path.join(self.root_dir, "image_2")
            self.ann_dir = os.path.join(self.root_dir, "gt_image_2")

            # read images
            image_file_names = []
            for root, dirs, files in os.walk(self.img_dir):
                image_file_names.extend(files)
            self.images = sorted(image_file_names)

            # read annotations
            annotation_file_names = []
            for root, dirs, files in os.walk(self.ann_dir):
                annotation_file_names.extend(files)
            self.annotations = sorted(annotation_file_names)

            assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):

            image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
            segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)

            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_() # remove batch dimension

            return encoded_inputs
        
    root_dir = "dataset/Road_seg_train"
    image_processor = SegformerImageProcessor(do_reduce_labels=True)

    dataset = SemanticSegmentationDataset(root_dir, image_processor, train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # config.num_labels = 3
    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", config=config, ignore_mismatched_sizes=True).to(device)
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    
    metric = evaluate.load("mean_iou")

    model.train()
    last_ma = 0
    not_improved_count = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(dataloader)):
                # get the inputs;
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss, logits = outputs.loss, outputs.logits

                loss.backward()
                optimizer.step()

                # evaluate
                with torch.no_grad():
                    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)

                    # note that the metric expects predictions + labels as numpy arrays
                    metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

                # let's print loss and metrics every 100 batches
                if idx % 100 == 0:
                    # currently using _compute instead of compute
                    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
                    metrics = metric._compute(
                            predictions=predicted.cpu(),
                            references=labels.cpu(),
                            num_labels=150,
                            ignore_index=255,
                            reduce_labels=False, # we've already reduced the labels ourselves
                        )

                    print("Loss:", loss.item())
                    print("Mean_iou:", metrics["mean_iou"])
                    print("Mean accuracy:", metrics["mean_accuracy"])

                    if metrics["mean_accuracy"] < last_ma:
                            not_improved_count += 1  
                    else:
                        not_improved_count = 0
                        last_ma = metrics["mean_accuracy"] 
        if not_improved_count > 4:
            break
    return model


def image_segmentation(image, model):
    def ade_palette():
        """ADE20K palette that maps each class to RGB values."""
        return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
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
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image[:,:,0].shape[::-1]])[0]
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

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()

    return predicted_segmentation_map


def main(train_model=False):
    image = cv2.imread("dataset/Czech/images/Czech_000019.jpg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)

    image_segmentation(image, model)

    if train_model:
        model = segmentation_model_training(epochs=100)
        torch.save(model.state_dict(), "model.pth")
    else:
        config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = SegformerForSemanticSegmentation(config=config).to(device)
        model.load_state_dict(torch.load("model.pth", weights_only=True))

    image_segmentation(image, model)

main(train_model=True)
