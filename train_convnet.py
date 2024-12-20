import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import Image
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

class RandomRotation:
    def __call__(self, img):
        angle = (torch.randint(0, 4, (1,)).item() * 90 + 
                5 * (torch.randint(0, 2, (1,)).item() * 2 - 1) + 
                (torch.randint(0, 2, (1,)).item() * 90 + 85) * (torch.randint(0, 2, (1,)).item() * 2 - 1) + 
                (torch.randint(0, 2, (1,)).item() * 90 + 175) * (torch.randint(0, 2, (1,)).item() * 2 - 1) + 
                (torch.randint(0, 2, (1,)).item() * 90 + 265) * (torch.randint(0, 2, (1,)).item() * 2 - 1))
        return transforms.functional.rotate(img, angle)

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
    RandomRotation(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.51054407, 0.4892153,  0.46805718], std=[0.51054407, 0.4892153,  0.46805718])
])

# Load and split the dataset
def create_dataset_splits(json_path, train_size=0.8):
    # Create a mapping of labels to indices
    unique_labels = set()
    labels_data = {}
    with open(json_path, 'r') as f:
        for line in f:
            j_line = json.loads(line)
            if j_line["use"] == 1:
                unique_labels.add(j_line["label"])
                labels_data[j_line["img_path"]] = j_line["label"]

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Create list of (image_path, label_index) pairs
    dataset = []
    for img_name, label in labels_data.items():
        img_path = img_name
        if os.path.exists(img_path) and label:
            dataset.append((img_path, label_to_idx[label]))

    if not dataset:
        raise ValueError("No valid images found in the dataset.")

    train_data, val_data = train_test_split(
        dataset, 
        train_size=train_size, 
        random_state=42, 
        stratify=[x[1] for x in dataset]
    )
    
    return train_data, val_data, label_to_idx  # Return the mapping as well

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label_idx = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_idx)  # Convert label_idx to tensor


import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPieceClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(ChessPieceClassifier, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size of flattened features
        # Assuming input size is 64x64
        self.flatten_size = 128 * 8 * 8
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First Block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second Block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third Block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Returns the predicted class for a single image or batch of images
        """
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs.data, 1)
            return predicted


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    train_losses = []  # List to store training losses
    val_losses = []    # List to store validation losses
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)  # Append training loss
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)  # Append validation loss
        val_acc = 100. * correct / total
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
    # Plotting the loss functions
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def predict(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.51054407, 0.4892153,  0.46805718], std=[0.51054407, 0.4892153,  0.46805718])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    return predicted.item()


# Move the dataset creation inside the if __name__ block
if __name__ == '__main__':
    # Create and train the model
    train_data, val_data, label_mapping = create_dataset_splits('labels.jsonl')
    print("Label mapping:", label_mapping)  # Print only once when running directly
    print("Length of dataset:", len(train_data) + len(val_data))

    train_dataset = ChessDataset(train_data, transform=transform)
    val_dataset = ChessDataset(val_data, transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.51054407, 0.4892153,  0.46805718], 
                           std=[0.51054407, 0.4892153,  0.46805718])
    ]))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ChessPieceClassifier(num_classes=13)
    train_model(model, train_loader, val_loader)
