import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import Image
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RandomRotation:
    def __call__(self, img):
        angle = (torch.randint(0, 4, (1,)).item() * 90 + 
                5 * (torch.randint(0, 2, (1,)).item() * 2 - 1) + 
                (torch.randint(0, 2, (1,)).item() * 90 + 85) * (torch.randint(0, 2, (1,)).item() * 2 - 1) + 
                (torch.randint(0, 2, (1,)).item() * 90 + 175) * (torch.randint(0, 2, (1,)).item() * 2 - 1) + 
                (torch.randint(0, 2, (1,)).item() * 90 + 265) * (torch.randint(0, 2, (1,)).item() * 2 - 1))
        return transforms.functional.rotate(img, angle)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomRotation(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.51054407, 0.4892153,  0.46805718], std=[0.51054407, 0.4892153,  0.46805718])
])

def create_dataset_splits(json_path, train_size=0.8):
    unique_labels = set()
    labels_data = {}
    with open(json_path, 'r') as f:
        for line in f:
            j_line = json.loads(line)
            if j_line["use"] == 1:
                unique_labels.add(j_line["label"])
                labels_data[j_line["img_path"]] = j_line["label"]

    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

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
    
    return train_data, val_data, label_to_idx

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
            
        return image, torch.tensor(label_idx)


class ChessPieceClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(ChessPieceClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # # Freeze early layers
        # for param in list(self.resnet.parameters())[:-4]:
        #     param.requires_grad = False
            
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, val_loader, num_epochs=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
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
        train_losses.append(train_loss)
        train_acc = 100. * correct / total
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        num_classes = 13
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    class_correct[label] += predicted[i].eq(labels[i]).item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_acc = 100. * correct / total
        
        class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if (epoch + 1) % 10 == 0:
            model_save_path = f"chess_piece_classifier_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
            
            plt.figure(figsize=(10, 5))
            bars = plt.bar(range(num_classes), class_accuracies, tick_label=[f'Class {i}' for i in range(num_classes)])
            plt.title(f'Validation Accuracy by Class at Epoch {epoch + 1}')
            plt.xlabel('Class')
            plt.ylabel('Accuracy (%)')
            plt.ylim(0, 100)
            
            for bar, correct, total in zip(bars, class_correct, class_total):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{correct}/{total}', ha='center', va='bottom')
            
            plt.savefig(f'val_accuracy_epoch_{epoch + 1}.png')
            plt.close()

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

if __name__ == '__main__':
    train_data, val_data, label_mapping = create_dataset_splits('labels.jsonl')
    print("Label mapping:", label_mapping)
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