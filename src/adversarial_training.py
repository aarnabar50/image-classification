
import datetime
import os

from libs.model_manager_lib import model_manager
from libs.config_manager_lib import config_manager

from intial_training import run_model_training, run_inference_mode
from fgsm_attack import run_fgsm_attack, run_perturbed_evaluation

def main():
    """Main function to provide user choices for model operations."""

    # print the current OS path
    print("Current OS Path:", os.getcwd())

    while True:
        print("\n" + "="*50)
        print("Adversarial Training - Model Operations")
        print("="*50)
        print("1. Train the basic model")
        print("2. Run model in inference mode")
        print("3. Run adversarial training")
        print("4. Run evaluation on adversarial images")
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
        elif choice == '3':
            print("Running FGSM adversarial attack...")
            run_fgsm_attack()
        elif choice == '4':
            print("Running evaluation on adversarial images...")
            run_perturbed_evaluation()
        else:
            print("Invalid choice. Please try again.")




if __name__ == "__main__":
    main()