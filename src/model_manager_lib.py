
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import datetime
from PIL import Image
import os
from torchvision.utils import save_image


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


    def load_training_dataset(self, train_dir):
        """Initialize the data sets for training and validation datasets."""

        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=self.train_transform)

        batch_size = 32
        num_workers = 1

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # changed to True for training
            num_workers=num_workers,
            pin_memory=False
        )

        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)

    def load_evaludation_dataset(self, val_dir):
        """Initialize the data sets for training and validation datasets."""

        val_dataset   = datasets.ImageFolder(root=str(val_dir), transform=self.val_transform)

        batch_size = 32
        num_workers = 1


        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,   # changed to False for validation
            num_workers=num_workers,
            pin_memory=False
        )

        self.class_names = val_dataset.classes
        self.num_classes = len(self.class_names)



    def load_datasets(self, train_dir, val_dir):
        """Initialize the data sets for training and validation datasets."""
        self.load_training_dataset(train_dir)
        self.load_evaludation_dataset(val_dir)


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

        epoch_total_images = total  # Total images processed in this epoch
        epoch_time = loop.format_dict['elapsed']  # Time in seconds for this epoch
        epoch_latency = epoch_time / epoch_total_images if epoch_total_images > 0 else 0
        epoch_cpu_time = epoch_time * torch.get_num_threads()  # Approximate CPU time

        return epoch_loss, epoch_acc, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time

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

        epoch_total_images = total  # Total images processed in this epoch
        epoch_time = loop.format_dict['elapsed']  # Time in seconds for this epoch
        epoch_latency = epoch_time / epoch_total_images if epoch_total_images > 0 else 0
        epoch_cpu_time = epoch_time * torch.get_num_threads()  # Approximate CPU time
        

        return epoch_loss, epoch_acc, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time

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
    
    def generate_adversarial_images(self, parturbed_data_directory):
        """Generate adversarial images using the Fast Gradient Sign Method (FGSM)."""
        self.model.eval()
        os.makedirs(parturbed_data_directory, exist_ok=True)

        epsilon = 0.03
        total_images = 0

        # Loop through all batches in validation loader
        loop = tqdm(self.val_loader, desc="Generating adversarial images", leave=False)
        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Set requires_grad attribute of tensor. Important for Attack
            images.requires_grad = True

            # Forward pass the data through the model
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = images.grad.data

            # FGSM Attack
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_images = images + epsilon * data_grad.sign()
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

            # Save the perturbed images to the directory
            for i, perturbed_img in enumerate(perturbed_images):
                label = labels[i].item()
                class_name = self.class_names[label] if self.class_names else str(label)
                
                # Create class subdirectory
                class_dir = os.path.join(parturbed_data_directory, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Save image with unique name using batch_idx and i
                img_path = os.path.join(class_dir, f"perturbed_batch{batch_idx}_img{i}.png")
                save_image(perturbed_img, img_path)
                total_images += 1

        print(f"Saved {total_images} perturbed images to {parturbed_data_directory}")

