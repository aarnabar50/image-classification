import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from model_manager_utility import model_manager
def main():
    """Main function to provide user choices for model operations."""
    while True:
        print("\n" + "="*50)
        print("Adversarial Training - Model Operations")
        print("="*50)
        print("1. Train the basic model")
        print("2. Run model in evaluation mode")
        print("0. Quit")
        print("="*50)
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            print("Training the basic model...")            
            run_model_training()
        elif choice == '2':
            print("Running model in inference mode...")
            run_inference_mode()
        else:
            print("Invalid choice. Please try again.")


def run_model_training():
    """Function to run training epochs for adversarial training."""
    best_val_acc = 0.0
    number_of_epochs = 10  # Define the number of epochs
    print_every_epoch = 1  # Define how often to print progress

    print("Enter the number of epochs for training:")
    number_of_epochs = int(input().strip())

    print("Enter the desired level of training (percentage):")
    training_percentage = int(input().strip())


    data_dir = "/Users/aarnabar/image-classification/data/cropped_lisa_1"   # root folder that contains train/ and val/
    train_dir = f"{data_dir}/train_1"
    val_dir   = f"{data_dir}/val_1"

    
    base_model = model_manager()
    base_model.load_datasets(train_dir, val_dir)
    base_model.init_model()    
    base_model.init_optimizer()                      


    for epoch in range(number_of_epochs):
        print(f"\nEpoch {epoch + 1}/{number_of_epochs}")

        train_loss, train_acc = base_model.train(training_percentage)

        val_loss, val_acc = base_model.evaluate()

        if (epoch + 1) % print_every_epoch == 0:
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f}"
            )

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            base_model.save_checkpoint(epoch, val_acc, model_type="BaseModel")


    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)

def run_inference_mode():
    """Function to run the model in inference mode."""

    infer_model = model_manager()
    
    # Load the trained model
    model_path = input("Enter the path to the trained model: ").strip()

    infer_model.init_model(model_path)
    
    while True:
        image_path = input("\nEnter the image path (or 'q' to quit): ").strip()
        
        if image_path.lower() == 'q':
            print("Exiting inference mode...")
            break
        
        try:
            # Run inference on the image
            prediction = infer_model.infer(image_path)
            print(f"Prediction: {prediction}")
        except Exception as e:
            print(f"Error processing image: {e}")



if __name__ == "__main__":
    main()