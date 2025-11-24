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
            print("Running model in evaluation mode...")
            # Add your evaluation logic here
            # evaluate_model()
        else:
            print("Invalid choice. Please try again.")


def run_model_training():
    """Function to run training epochs for adversarial training."""
    best_val_acc = 0.0
    number_of_epochs = 10  # Define the number of epochs
    print_every_epoch = 1  # Define how often to print progress

    print("Enter the number of epochs for training:")
    number_of_epochs = int(input().strip())

    data_dir = "/Users/aarnabar/image-classification/data/cropped_lisa_1"   # root folder that contains train/ and val/
    train_dir = f"{data_dir}/train_1"
    val_dir   = f"{data_dir}/val_1"

    model_mgr = model_manager(train_dir, val_dir)



    for epoch in range(number_of_epochs):
        print(f"\nEpoch {epoch + 1}/{number_of_epochs}")

        train_loss, train_acc = model_mgr.train_one_epoch()

        val_loss, val_acc = model_mgr.evaluate()

        if (epoch + 1) % print_every_epoch == 0:
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f}"
            )


    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)

if __name__ == "__main__":
    main()