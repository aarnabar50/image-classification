
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import datetime
from PIL import Image

class model_manager:
    def __init__(self):
        """Initialize the ModelTrainer with data loaders."""
        self.init_transforms()
        self.model = None

    def init_transforms(self):
        """Initialize the data loaders for training and validation datasets."""
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])



    def load_datasets(self, train_dir, val_dir):
        """Initialize the data sets for training and validation datasets."""

        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=self.train_transform)
        val_dataset   = datasets.ImageFolder(root=str(val_dir), transform=self.val_transform)

        batch_size = 32
        num_workers = 1

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # changed to True for training
            num_workers=num_workers,
            pin_memory=False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,   # changed to False for validation
            num_workers=num_workers,
            pin_memory=False
        )

        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)

        print("Number of classes:", self.num_classes)
        print("Classes:", self.class_names)        

    def init_model(self, state_dict_path=None):
        """Initialize the model, loss function, and optimizer."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        
        if( state_dict_path):
            checkpoint = torch.load(state_dict_path, map_location=self.device)
            self.class_names = checkpoint.get('class_names', [])
            self.num_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
            self.model = models.resnet18(num_classes=self.num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("State dict loaded from:", state_dict_path)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("Initialized new model.")
            

        self.init_optimizer()


    def init_optimizer(self):
        """Initialize optimizers."""
        learning_rate = 1e-4
        weight_decay = 1e-4
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)  # replace final layer
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay) 



    def train(self, training_percentage=100):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(self.train_loader, desc="Train", leave=False)
        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

            # Check if we've reached the desired training percentage
            progress_percentage = (loop.n / loop.total) * 100
            if progress_percentage >= training_percentage:
                break

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc


    def evaluate(self, eval_percentage):
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Val", leave=False)
            for images, labels in loop:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item())

                # Check if we've reached the desired evaluation percentage
                progress_percentage = (loop.n / loop.total) * 100
                if progress_percentage >= eval_percentage:
                    break


        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_acc, checkpoint_path):
        """Save the model checkpoint."""
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": self.class_names,
        }, checkpoint_path)
        print(f"--> New best model checkpoint saved with val_acc={val_acc:.4f}")


    def infer(self, image_path): 
        """Run inference on a single image."""
        self.model.eval().to(self.device)

        image = Image.open(image_path).convert("RGB")
        tensor = self.val_transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
        pred_class = outputs.argmax(dim=1).item()
        pred_class_name = self.class_names[pred_class] if self.class_names else str(pred_class)
        print("Predicted class index:", pred_class)
        print("Predicted class name:", pred_class_name)
        return pred_class_name