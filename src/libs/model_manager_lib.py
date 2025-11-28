
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


    def load_training_dataset(self, train_dir, in_batch_size, in_num_workers):
        """Initialize the data sets for training and validation datasets."""

        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=self.train_transform)

        batch_size = in_batch_size
        num_workers = in_num_workers

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # changed to True for training
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True
        )

        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)

    def load_evaluation_dataset(self, val_dir, in_batch_size, in_num_workers):
        """Initialize the data sets for training and validation datasets."""

        val_dataset   = datasets.ImageFolder(root=str(val_dir), transform=self.val_transform)

        batch_size = in_batch_size
        num_workers = in_num_workers


        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,   # changed to False for validation
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True
        )

        self.class_names = val_dataset.classes
        self.num_classes = len(self.class_names)



    def load_datasets(self, train_dir, val_dir, in_batch_size, in_num_workers):
        """Initialize the data sets for training and validation datasets."""
        self.load_training_dataset(train_dir, in_batch_size, in_num_workers)
        self.load_evaluation_dataset(val_dir, in_batch_size, in_num_workers)

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
            # Model already has trained fc layer, just move to device and setup optimizer
            self.model = self.model.to(self.device)
            self.init_optimizer(replace_fc=False)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("Initialized new model.")
            # New model needs fc layer replaced for custom num_classes
            self.init_optimizer(replace_fc=True)


    def init_optimizer(self, replace_fc=True):
        """Initialize optimizers."""
        learning_rate = 1e-4
        weight_decay = 1e-4
        
        # Only replace fc layer if training from scratch (not loading checkpoint)
        if replace_fc:
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

    def train_yopo(self, training_percentage=100, epsilon=0.031, num_steps=5, step_size=0.007):
        """
        Train the model using YOPO (You Only Propagate Once) adversarial training.
        
        YOPO is an efficient adversarial training method that reduces computational cost
        by propagating gradients only once and reusing cached gradients for attack generation.
        
        Args:
            training_percentage: Percentage of training data to use (1-100)
            epsilon: Maximum perturbation magnitude (L-infinity norm)
            num_steps: Number of PGD steps for attack generation
            step_size: Step size for each PGD iteration
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc="YOPO Train", leave=False)
        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Step 1: Forward pass on clean images
            self.optimizer.zero_grad()
            outputs_clean = self.model(images)
            loss_clean = self.criterion(outputs_clean, labels)
            
            # Step 2: Backward pass to compute gradients (YOPO: propagate once)
            loss_clean.backward()
            
            # Step 3: Cache the gradient of the first layer (input-level gradients)
            # For YOPO, we need to register hooks to capture intermediate gradients
            # Simplified version: use input gradients directly
            images_adv = images.detach().clone()
            images_adv.requires_grad = True
            
            # Generate adversarial examples using PGD with cached information
            for _ in range(num_steps):
                # Forward pass for adversarial generation
                outputs_adv = self.model(images_adv)
                loss_adv = self.criterion(outputs_adv, labels)
                
                # Compute gradients w.r.t. adversarial images
                grad = torch.autograd.grad(loss_adv, images_adv, create_graph=False)[0]
                
                # PGD step
                images_adv = images_adv.detach() + step_size * grad.sign()
                images_adv = torch.max(torch.min(images_adv, images + epsilon), images - epsilon)
                images_adv = torch.clamp(images_adv, 0, 1)
                images_adv.requires_grad = True
            
            # Step 4: Update model parameters using the already computed gradients from clean loss
            # YOPO uses the gradients from Step 2 (clean images) to update parameters
            self.optimizer.step()
            
            # Step 5: Optional - Train on adversarial examples with a separate forward/backward
            # This is a hybrid approach for better robustness
            self.optimizer.zero_grad()
            outputs_adv_final = self.model(images_adv.detach())
            loss_adv_final = self.criterion(outputs_adv_final, labels)
            loss_adv_final.backward()
            self.optimizer.step()
            
            # Track metrics (use adversarial predictions)
            running_loss += loss_adv_final.item() * images.size(0)
            _, preds = outputs_adv_final.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss_adv_final.item())
            
            # Check if we've reached the desired training percentage
            progress_percentage = (loop.n / loop.total) * 100
            if progress_percentage >= training_percentage:
                break
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        epoch_total_images = total
        epoch_time = loop.format_dict['elapsed']
        epoch_latency = epoch_time / epoch_total_images if epoch_total_images > 0 else 0
        epoch_cpu_time = epoch_time * torch.get_num_threads()
        
        return epoch_loss, epoch_acc, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time

    def evaluate(self, eval_percentage):
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_confidence = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Val", leave=False)
            for images, labels in loop:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Calculate confidence (softmax probabilities)
                probs = torch.softmax(outputs, dim=1)
                confidences, preds = probs.max(1)
                
                running_loss += loss.item() * images.size(0)
                correct += (preds == labels).sum().item()
                total_confidence += confidences.sum().item()
                total += labels.size(0)

                loop.set_postfix(loss=loss.item())

                # Check if we've reached the desired evaluation percentage
                progress_percentage = (loop.n / loop.total) * 100
                if progress_percentage >= eval_percentage:
                    break


        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_avg_confidence = total_confidence / total

        epoch_total_images = total  # Total images processed in this epoch
        epoch_time = loop.format_dict['elapsed']  # Time in seconds for this epoch
        epoch_latency = epoch_time / epoch_total_images if epoch_total_images > 0 else 0
        epoch_cpu_time = epoch_time * torch.get_num_threads()  # Approximate CPU time
        

        return epoch_loss, epoch_acc, epoch_avg_confidence, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time

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
        probs = torch.softmax(outputs, dim=1)
        confidence = probs[0, pred_class].item()

        return pred_class_name, confidence
    

    def generate_adversarial_images(self, perturbed_data_directory, epsilon=0.03, 
                                    num_steps=1, step_size=None, normalized=False):
        """
        Generate adversarial images using FGSM or PGD.
        
        Args:
            perturbed_data_directory: Output directory for adversarial images
            epsilon: Maximum perturbation magnitude (L-infinity norm)
            num_steps: Number of PGD iterations (1 = FGSM)
            step_size: Step size for PGD (default: epsilon for FGSM, epsilon/4 for PGD)
            normalized: Whether images are normalized (adjusts clamp bounds)
        """
        self.model.eval()
        os.makedirs(perturbed_data_directory, exist_ok=True)
        
        # Compute valid pixel range if images are normalized
        if normalized and hasattr(self, 'val_transform'):
            # Extract mean/std from Normalize transform
            normalize = [t for t in self.val_transform.transforms if isinstance(t, transforms.Normalize)]
            if normalize:
                mean = torch.tensor(normalize[0].mean).view(3, 1, 1).to(self.device)
                std = torch.tensor(normalize[0].std).view(3, 1, 1).to(self.device)
                pixel_min = (0 - mean) / std
                pixel_max = (1 - mean) / std
            else:
                pixel_min, pixel_max = 0, 1
        else:
            pixel_min, pixel_max = 0, 1
        
        step_size = step_size or (epsilon if num_steps == 1 else epsilon / 4)
        total_images = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        loop = tqdm(self.val_loader, desc="Generating adversarial images", leave=False)
        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # PGD attack (FGSM if num_steps=1)
            perturbed = images.clone().detach()
            perturbed.requires_grad = True
            
            for step in range(num_steps):
                outputs = self.model(perturbed)
                loss = self.criterion(outputs, labels)
                self.model.zero_grad()
                loss.backward()
                
                # Take gradient step
                grad = perturbed.grad.data
                perturbed = perturbed.detach() + step_size * grad.sign()
                
                # Project back to epsilon ball around original image
                perturbed = torch.max(torch.min(perturbed, images + epsilon), images - epsilon)
                perturbed = torch.clamp(perturbed, pixel_min, pixel_max)
                perturbed.requires_grad = True
            
            # Save perturbed images
            for i, perturbed_img in enumerate(perturbed.detach()):
                label = labels[i].item()
                class_name = self.class_names[label] if self.class_names else str(label)
                
                class_dir = f"{perturbed_data_directory}/{class_name}"
                os.makedirs(class_dir, exist_ok=True)
                
                # Unique filename with timestamp to avoid collisions
                img_path = f"{class_dir}/adv_{timestamp}_batch{batch_idx:04d}_img{i:03d}.png"
                save_image(perturbed_img, img_path)
                total_images += 1
        
        print(f"Saved {total_images} adversarial images to {perturbed_data_directory}")
        print(f"Attack: {'FGSM' if num_steps == 1 else f'PGD-{num_steps}'}, epsilon={epsilon}")

