import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
import evaluate
import copy

def image_segmentation(image, model):
    """Perform image segmentation on the input image using the given model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
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

def segmentation_model_training(root_dir, epochs=100, patience=5):
    """Finetune a Segformer model with given dataset.
    root_dir should contain 'train' and 'val' folders with images, 'train_cm' and 'val_cm' with coresponding annotations."""

    class SemanticSegmentationDataset(Dataset):
        """Image (semantic) segmentation dataset."""

        def __init__(self, root_dir, image_processor, train=True):
            """
            Args:
                root_dir (string): Root directory of the dataset containing the images + annotations.
                image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            """
            self.root_dir = root_dir
            self.image_processor = image_processor
            self.train = train

            if self.train:
                self.img_dir = os.path.join(self.root_dir, "train")
                self.ann_dir = os.path.join(self.root_dir, "train_cm")
            else:
                self.img_dir = os.path.join(self.root_dir, "val")
                self.ann_dir = os.path.join(self.root_dir, "val_cm")

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
            # Normalize segmentation map to have classes starting from 0
            unique_labels = np.unique(segmentation_map)
            unique_labels = unique_labels[~np.isin(unique_labels, [0, 90])]
            global_label_map = {label: idx for idx, label in enumerate(sorted(unique_labels), start=2)}
            global_label_map[0] = 0
            global_label_map[90] = 1
            segmentation_map = np.vectorize(global_label_map.get)(segmentation_map)

            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_() # remove batch dimension

            return encoded_inputs
        
    image_processor = SegformerImageProcessor(do_reduce_labels=True)

    train_dataset = SemanticSegmentationDataset(root_dir, image_processor, train=True)
    val_dataset = SemanticSegmentationDataset(root_dir, image_processor, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    metric = evaluate.load("mean_iou")

    # EarlyStopping initialization
    best_iou = 0.0
    best_model_weights = None
    c_patience = patience

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("Epoch:", epoch)

        # train
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
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

        print("Loss:", loss.item())

        # evaluate
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits

                upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                # note that the metric expects predictions + labels as numpy arrays
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            metrics = metric.compute(
                num_labels=3,
                ignore_index=255,
                reduce_labels=False,  # we've already reduced the labels ourselves
            )

            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])

            # EarlyStopping
            if metrics["mean_iou"] < best_iou:
                c_patience -= 1
                if c_patience == 0:
                    model.load_state_dict(best_model_weights)
                    break
            else:
                best_iou = metrics["mean_iou"]
                best_model_weights = copy.deepcopy(model.state_dict())
                c_patience = patience

    return model