
import datetime
import os

from libs.model_manager_lib import model_manager
from libs.config_manager_lib import config_manager

from model_training import run_model_training, run_inference_mode
from fgsm_attack import generate_adversarial_images, run_perturbed_evaluation
from auto_encoder_training import denoise_with_autoencoder, run_auto_encoder_training

def main():
    """Main function to provide user choices for model operations."""

    # print the current OS path
    print("Current OS Path:", os.getcwd())

    while True:
        print("\n" + "="*50)
        print("Adversarial Training - Model Operations")
        print("="*50)
        print("1. Train the resetnet18 model with baseline images")
        print("2. Run the model in inference mode")
        print("3. Generate adversarial images using FGSM attacks")
        print("4. Generate adversarial images using PGD attacks")
        print("5. Run evaluation on FGSM adversarial images")
        print("6. Run evaluation on PGD adversarial images")
        print("7. Train the model using YOPO adversarial training")
        print("8. Train the denoiser auto encoder model")
        print("9. Denoise adversarial images using the auto encoder")
        print("0. Quit")
        
        print("="*50)
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            print("Running baseline training...")            
            run_model_training("./src/configurations/01_training_baseline.json", "Baseline")
        elif choice == '2':
            print("Running model in inference mode...")
            run_inference_mode("./src/configurations/02_inference.json")
        elif choice == '3':
            print("Running FGSM adversarial attack...")
            generate_adversarial_images("./src/configurations/03_attack_fgsm.json")
        elif choice == '4':
            print("Running PGD adversarial attack...")
            generate_adversarial_images("./src/configurations/04_attack_pgd.json")
        elif choice == '5':
            print("Running evaluation on FGSM adversarial images...")
            run_perturbed_evaluation("./src/configurations/05_evaluation_fgsm.json")
        elif choice == '6':
            print("Running evaluation on PGD adversarial images...")
            run_perturbed_evaluation("./src/configurations/06_evaluation_pgd.json")
        elif choice == '7':
            print("Running YOPO YOPO training...")
            run_model_training("./src/configurations/07_training_yopo.json", "YOPO")
        elif choice == '8':
            print("Running denoiser auto encoder training...")
            run_auto_encoder_training("./src/configurations/08_training_auto_encoder.json")
        elif choice == '9':
            print("Denoising adversarial images using the auto encoder...")
            denoise_with_autoencoder("./src/configurations/09_denoise_with_auto_encoder.json")
        else:
            print("Invalid choice. Please try again.")




if __name__ == "__main__":
    main()