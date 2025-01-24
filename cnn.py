import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import copy

def load_bounding_box_data(xml_file):
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
    image = cv2.imread(image_path)
    for obj in objects:
        name = obj['name']
        xmin, ymin, xmax, ymax = obj['bbox']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: background, D40

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def cnn_model_training(epochs=10, patience=3):
    class ObjectDetectionDataset(Dataset):
        def __init__(self, xml_folder, image_folder, transform=None, window_size=(64, 64)):
            self.xml_folder = xml_folder
            self.image_folder = image_folder
            self.transform = transform
            self.window_size = window_size
            self.xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]

        def __len__(self):
            return len(self.xml_files)

        def __getitem__(self, idx):
            xml_file = self.xml_files[idx]
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find('filename').text
            image_path = os.path.join(self.image_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            positive_samples = []
            negative_samples = []

            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == 'D40':
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    label = 1

                    # Extract positive sample
                    positive_sample = image[ymin:ymax, xmin:xmax]
                    if positive_sample.size > 0:
                        positive_sample = cv2.resize(positive_sample, self.window_size)
                        positive_sample = Image.fromarray(positive_sample)  # Convert to PIL image
                        positive_samples.append((positive_sample, label))

                    # Generate negative samples
                    for _ in range(5):  # Generate 5 negative samples per object
                        x = random.randint(0, image.shape[1] - self.window_size[0])
                        y = random.randint(0, image.shape[0] - self.window_size[1])
                        if x < xmin or x > xmax or y < ymin or y > ymax:
                            negative_sample = image[y:y + self.window_size[1], x:x + self.window_size[0]]
                            if negative_sample.size > 0:
                                negative_sample = Image.fromarray(negative_sample)  # Convert to PIL image
                                negative_samples.append((negative_sample, 0))  # 2 indicates background

            samples = positive_samples + negative_samples
            if len(samples) == 0:
                # Handle case where there are no objects in the image
                x = random.randint(0, image.shape[1] - self.window_size[0])
                y = random.randint(0, image.shape[0] - self.window_size[1])
                negative_sample = image[y:y + self.window_size[1], x:x + self.window_size[0]]
                if negative_sample.size > 0:
                    negative_sample = Image.fromarray(negative_sample)  # Convert to PIL image
                    negative_samples.append((negative_sample, 0))
                samples = negative_samples

            random.shuffle(samples)

            images, labels = zip(*samples)
            images = [self.transform(img) for img in images]

            return torch.stack(images), torch.tensor(labels)
        

    def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10, patience=3):
        best_val_acc = 0.0
        best_model_weights = None
        c_patience = patience

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_dataloader)}")

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            weighted_correct = 0
            weighted_total = 0

            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Apply 10x weight for correct label 1 predictions
                    for i in range(labels.size(0)):
                        if labels[i] == 1:
                            weighted_total += 100
                            if predicted[i] == labels[i]:
                                weighted_correct += 100
                        else:
                            weighted_total += 1
                            if predicted[i] == labels[i]:
                                weighted_correct += 1

            val_loss /= len(val_dataloader)
            accuracy = 100 * correct / total
            weighted_accuracy = 100 * weighted_correct / weighted_total
            print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}%, Weighted Accuracy: {weighted_accuracy}%")

            # Early stopping
            if weighted_accuracy < best_val_acc:
                c_patience -= 1
                if c_patience == 0:
                    model.load_state_dict(best_model_weights)
                    break
            else:
                best_val_acc = weighted_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                c_patience = patience



    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Model training
    xml_folder = 'dataset/Czechtrain/processed_annotations'
    image_folder = 'dataset/Czechtrain/processed_images'
    # xml_folder = 'dataset/Czechtrain/annotations/xmls'
    # image_folder = 'dataset/Czechtrain/images'
    # Load all XML files
    xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]

    # Split into training and testing sets
    train_files, test_files = train_test_split(xml_files, test_size=0.2)

    # Create datasets
    train_dataset = ObjectDetectionDataset(xml_folder, image_folder, transform=transform)
    test_dataset = ObjectDetectionDataset(xml_folder, image_folder, transform=transform)

    # Update the dataset's xml_files attribute
    train_dataset.xml_files = train_files
    test_dataset.xml_files = test_files

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: (torch.cat([i for i, _ in x]), torch.cat([l for _, l in x])))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: (torch.cat([i for i, _ in x]), torch.cat([l for _, l in x])))
    
    model = SimpleCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=epochs, patience=patience)

    return model


def sliding_window(device, image, model, window_size, step_size):
    model.eval()
    windows = []
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            window = cv2.resize(window, (64, 64))  # Resize to match input size of the model
            window = transforms.ToTensor()(window).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = model(window)
                probabilities = F.softmax(probabilities, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                if predicted.item() != 0:  # Not background
                    windows.append((x, y, predicted.item(), confidence.item()))
    return windows

def visualize_detections(image, windows, window_size, offset=(0, 0)):
    image = copy.deepcopy(image)
    offset_x, offset_y = offset
    windows = [(x, y, label, confidence) for (x, y, label, confidence) in windows if confidence > 0.9]
    windows = [(x + offset_x, y + offset_y, label, confidence) for (x, y, label, confidence) in windows]
    for (x, y, label, confidence) in windows:
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), color, 2)
        label_text = f'Pothole: {confidence:.2f}'
        cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image