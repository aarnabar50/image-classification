import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import os
from torchvision.utils import save_image


class ImageAutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder for image denoising and adversarial defense.
    Encodes 224x224x3 images to a latent representation and reconstructs them.
    """
    def __init__(self, latent_dim=128):
        super(ImageAutoEncoder, self).__init__()
        
        # Encoder: 224x224x3 -> latent_dim
        self.encoder = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 112x112x64 -> 56x56x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 56x56x128 -> 28x28x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 28x28x256 -> 14x14x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 14x14x512 -> 7x7x512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        
        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (512, 7, 7))
        
        # Decoder: latent_dim -> 224x224x3
        self.decoder = nn.Sequential(
            # 7x7x512 -> 14x14x512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 14x14x512 -> 28x28x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 28x28x256 -> 56x56x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 56x56x128 -> 112x112x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 112x112x64 -> 224x224x3
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range (adjust based on your normalization)
        )
    
    def forward(self, x):
        """Forward pass through encoder and decoder."""
        # Encode
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc_encode(x)
        
        # Decode
        x = self.fc_decode(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder(x)
        
        return reconstructed, latent


class autoencoder_manager:
    """
    Manager class for training and using autoencoders for adversarial defense.
    """
    def __init__(self, device=None):
        """Initialize the autoencoder manager."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.autoencoder_optimizer = None
        self.reconstruction_criterion = nn.MSELoss()
        
        # Default validation transform
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def init_autoencoder(self, latent_dim=128, checkpoint_path=None):
        """
        Initialize the autoencoder model.
        
        Args:
            latent_dim: Dimension of the latent representation
            checkpoint_path: Path to a saved autoencoder checkpoint (optional)
        """
        self.autoencoder = ImageAutoEncoder(latent_dim=latent_dim)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            print(f"Autoencoder loaded from: {checkpoint_path}")
        else:
            print(f"Initialized new autoencoder with latent_dim={latent_dim}")
        
        self.autoencoder = self.autoencoder.to(self.device)
        self.autoencoder_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5
        )

    def train_autoencoder_with_adversarial(self, adversarial_data_dir, clean_data_dir, 
                                          batch_size=32, num_workers=2, training_percentage=100):
        """
        Train the autoencoder to denoise adversarial images.
        
        The autoencoder learns to reconstruct clean images from adversarial inputs,
        effectively learning to remove adversarial perturbations.
        
        Args:
            adversarial_data_dir: Directory containing adversarial images (ImageFolder structure)
            clean_data_dir: Directory containing corresponding clean images
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            training_percentage: Percentage of data to use for training
        
        Returns:
            Tuple of (epoch_loss, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time)
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not initialized. Call init_autoencoder() first.")
        
        # Load adversarial dataset
        adversarial_dataset = datasets.ImageFolder(
            root=str(adversarial_data_dir), 
            transform=self.val_transform
        )
        
        adversarial_loader = DataLoader(
            adversarial_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        
        # Load clean dataset
        clean_dataset = datasets.ImageFolder(
            root=str(clean_data_dir),
            transform=self.val_transform
        )
        
        clean_loader = DataLoader(
            clean_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        
        self.autoencoder.train()
        running_loss = 0.0
        total = 0
        
        # Create iterator for clean images
        print("Training: Creating iterator for clean images...")
        clean_iter = iter(clean_loader)
        print("Interator created.")
        
        loop = tqdm(adversarial_loader, desc="Training AutoEncoder", leave=False)
        for adv_images, _ in loop:
            # Get corresponding clean images
            try:
                clean_images, _ = next(clean_iter)
            except StopIteration:
                print("Restarting clean image iterator...")
                clean_iter = iter(clean_loader)
                clean_images, _ = next(clean_iter)
            
            # Ensure batch sizes match
            batch_size_actual = min(adv_images.size(0), clean_images.size(0))
            adv_images = adv_images[:batch_size_actual].to(self.device)
            clean_images = clean_images[:batch_size_actual].to(self.device)
            
            # Forward pass: reconstruct clean images from adversarial inputs
            self.autoencoder_optimizer.zero_grad()
            reconstructed, _ = self.autoencoder(adv_images)
            
            # Loss: MSE between reconstructed and clean images
            loss = self.reconstruction_criterion(reconstructed, clean_images)
            
            # Backward pass
            loss.backward()
            self.autoencoder_optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * batch_size_actual
            total += batch_size_actual
            
            loop.set_postfix(loss=loss.item())
            
            # Check training percentage
            progress_percentage = (loop.n / loop.total) * 100
            if progress_percentage >= training_percentage:
                break
        
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_total_images = total
        epoch_time = loop.format_dict['elapsed']
        epoch_latency = epoch_time / epoch_total_images if epoch_total_images > 0 else 0
        epoch_cpu_time = epoch_time * torch.get_num_threads()
        
        return epoch_loss, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time

    def evaluate_autoencoder(self, adversarial_data_dir, clean_data_dir,
                            batch_size=32, num_workers=2, eval_percentage=100):
        """
        Evaluate the autoencoder on adversarial images.
        
        Measures reconstruction quality (MSE loss) on adversarial inputs.
        
        Args:
            adversarial_data_dir: Directory containing adversarial images
            clean_data_dir: Directory containing corresponding clean images
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            eval_percentage: Percentage of data to evaluate
        
        Returns:
            Tuple of (epoch_loss, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time)
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not initialized. Call init_autoencoder() first.")
        
        adversarial_dataset = datasets.ImageFolder(
            root=str(adversarial_data_dir), 
            transform=self.val_transform
        )
        
        adversarial_loader = DataLoader(
            adversarial_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        clean_dataset = datasets.ImageFolder(
            root=str(clean_data_dir),
            transform=self.val_transform
        )
        
        clean_loader = DataLoader(
            clean_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        self.autoencoder.eval()
        running_loss = 0.0
        total = 0
        
        print("Evaluation: Creating iterator for clean images...")
        clean_iter = iter(clean_loader)
        print("Iterator created.")
        
        with torch.no_grad():
            loop = tqdm(adversarial_loader, desc="Evaluating AutoEncoder", leave=False)
            for adv_images, _ in loop:
                try:
                    clean_images, _ = next(clean_iter)
                except StopIteration:
                    print("Restarting clean image iterator...")
                    clean_iter = iter(clean_loader)
                    clean_images, _ = next(clean_iter)
                
                batch_size_actual = min(adv_images.size(0), clean_images.size(0))
                adv_images = adv_images[:batch_size_actual].to(self.device)
                clean_images = clean_images[:batch_size_actual].to(self.device)
                
                reconstructed, _ = self.autoencoder(adv_images)
                loss = self.reconstruction_criterion(reconstructed, clean_images)
                
                running_loss += loss.item() * batch_size_actual
                total += batch_size_actual
                
                loop.set_postfix(loss=loss.item())
                
                progress_percentage = (loop.n / loop.total) * 100
                if progress_percentage >= eval_percentage:
                    break
        
        epoch_loss = running_loss / total if total > 0 else 0
        epoch_total_images = total
        epoch_time = loop.format_dict['elapsed']
        epoch_latency = epoch_time / epoch_total_images if epoch_total_images > 0 else 0
        epoch_cpu_time = epoch_time * torch.get_num_threads()
        
        return epoch_loss, epoch_total_images, epoch_time, epoch_latency, epoch_cpu_time

    def save_autoencoder_checkpoint(self, epoch, loss, checkpoint_path):
        """Save autoencoder checkpoint."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not initialized.")
        
        torch.save({
            "epoch": epoch + 1,
            "autoencoder_state_dict": self.autoencoder.state_dict(),
            "optimizer_state_dict": self.autoencoder_optimizer.state_dict(),
            "loss": loss,
        }, checkpoint_path)
        print(f"--> Autoencoder checkpoint saved with loss={loss:.6f}")

    def denoise_with_autoencoder(self, input_dir, output_dir):
        """
        Denoise all images from a directory using the trained autoencoder.
        Preserves the subdirectory structure (ImageFolder format with class folders).
        
        Args:
            input_dir: Directory containing adversarial images (ImageFolder structure)
            output_dir: Directory to save denoised images (will create same structure)
        
        Returns:
            Total number of images processed
        """
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not initialized. Call init_autoencoder() first.")
        
        self.autoencoder.eval()
        
        input_path = os.path.abspath(input_dir)
        output_path = os.path.abspath(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        total_processed = 0
        
        # Walk through all subdirectories (class folders)
        for class_name in os.listdir(input_path):
            class_input_dir = os.path.join(input_path, class_name)
            
            # Skip if not a directory
            if not os.path.isdir(class_input_dir):
                continue
            
            # Create corresponding output class directory
            class_output_dir = os.path.join(output_path, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            print(f"Processing class: {class_name}")
            
            # Process all images in this class folder
            image_files = [f for f in os.listdir(class_input_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for image_file in tqdm(image_files, desc=f"Denoising {class_name}", leave=False):
                image_input_path = os.path.join(class_input_dir, image_file)
                image_output_path = os.path.join(class_output_dir, image_file)
                
                try:
                    # Load and preprocess image
                    image = Image.open(image_input_path).convert("RGB")
                    tensor = self.val_transform(image).unsqueeze(0).to(self.device)
                    
                    # Denoise
                    with torch.no_grad():
                        denoised, _ = self.autoencoder(tensor)
                    
                    # Save denoised image
                    save_image(denoised[0], image_output_path)
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {image_input_path}: {e}")
                    continue
        
        print(f"\nTotal images denoised: {total_processed}")
        print(f"Output directory: {output_path}")
        
        return total_processed

    def set_transform(self, transform):
        """Set custom transform for image preprocessing."""
        self.val_transform = transform
