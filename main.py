import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Step 1 - Detailed logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 2 - Dataset path
DATASET_DIR = "foof/food-101/images"

# Check dataset exists
if not os.path.exists(DATASET_DIR):
    logger.error(f"Dataset directory {DATASET_DIR} not found!")
    exit()

logger.info(f"Found dataset directory: {DATASET_DIR}")
available_classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
logger.info(f"Total available classes: {len(available_classes)}")
logger.info(f"First 20 classes: {available_classes[:20]}")

# Step 3 - Enhanced CNN Model with more layers
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third Conv Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fourth Conv Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # 224/16 = 14
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Conv blocks
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout_fc1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        return x

# Step 4 - Data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 5 - Proper Dataset Class with real train/test split
class Food101Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading {self.image_paths[idx]}: {e}")
            # Return dummy data if image fails
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, self.labels[idx]

# Step 6 - Load all data and create proper splits
def load_food_data(dataset_dir, selected_classes, images_per_class=1000):
    all_image_paths = []
    all_labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    logger.info("Loading image paths...")
    for class_name in selected_classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory {class_dir} not found!")
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit number of images per class
        if len(image_files) > images_per_class:
            image_files = random.sample(image_files, images_per_class)
        
        logger.info(f"Class '{class_name}': {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            all_image_paths.append(img_path)
            all_labels.append(class_to_idx[class_name])
    
    logger.info(f"Total images loaded: {len(all_image_paths)}")
    
    # Proper train/validation/test split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_image_paths, all_labels, test_size=0.15, random_state=42, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
    )
    
    logger.info(f"Train set: {len(X_train)} images")
    logger.info(f"Validation set: {len(X_val)} images")
    logger.info(f"Test set: {len(X_test)} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx

# Step 7 - Training function with detailed progress
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=25):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        logger.info("-" * 50)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        logger.info(f"Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_food_model.pth')
            logger.info(f"New best validation accuracy: {best_val_acc:.2f}% - Model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Step 8 - Detailed evaluation function
def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    all_predictions = []
    all_labels = []
    
    logger.info("\nEvaluating on test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    overall_accuracy = 100 * correct / total
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Test Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    logger.info(f"{'='*60}")
    
    # Detailed per-class results
    idx_to_class = {v: k for k, v in class_names.items()}
    logger.info("Per-class accuracy:")
    for class_idx in range(len(class_names)):
        if class_total[class_idx] > 0:
            class_acc = 100 * class_correct[class_idx] / class_total[class_idx]
            class_name = idx_to_class[class_idx]
            logger.info(f"{class_name:15}: {class_acc:6.2f}% ({class_correct[class_idx]:3}/{class_total[class_idx]:3})")
    
    return overall_accuracy

# Step 9 - Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Select classes to train on (you can change these)
    selected_classes = [
        'pizza', 'hamburger', 'apple_pie', 'sushi', 'ice_cream',
        'french_fries', 'chocolate_cake', 'tacos', 'hot_dog', 'donuts'
    ]
    
    logger.info(f"Selected classes: {selected_classes}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load data with proper splits
        X_train, X_val, X_test, y_train, y_val, y_test, class_to_idx = load_food_data(
            DATASET_DIR, selected_classes, images_per_class=800  # Use more images
        )
        
        # Create datasets
        train_dataset = Food101Dataset(X_train, y_train, train_transform)
        val_dataset = Food101Dataset(X_val, y_val, test_transform)
        test_dataset = Food101Dataset(X_test, y_test, test_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Initialize model
        model = EnhancedCNN(num_classes=len(selected_classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train the model
        logger.info("Starting training...")
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30
        )
        
        # Load best model and evaluate
        model.load_state_dict(torch.load('best_food_model.pth'))
        final_accuracy = evaluate_model(model, test_loader, class_to_idx)
        
        logger.info(f"\nTraining completed! Final test accuracy: {final_accuracy:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()