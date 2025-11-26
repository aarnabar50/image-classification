import datetime
import os

from libs.model_manager_lib import model_manager
from libs.config_manager_lib import config_manager

def run_fgsm_attack():
    """Function to run the model to generate images with FGSM attack."""

    attack_model = model_manager()
    config = config_manager("./src/configurations/config_attack.json")
    val_dir   = config.get_config_value("validation_data_directory")
    parturbed_data_directory = config.get_config_value("parturbed_data_directory")

    attack_model.load_evaludation_dataset(val_dir)
    attack_model.init_model()

    attack_model.generate_adversarial_images(parturbed_data_directory)

    print(f"\nAdversarial images generated and saved to {parturbed_data_directory}")


def run_perturbed_evaluation():
    """Function to run evaluation on perturbed data."""

    print("\n" + "="*50)
    print("Running the evalutation on perturbed data...")
    config = config_manager("./src/configurations/config_parturbed_evaluation.json")
    run_name = config.get_config_value("run_name")
    val_dir   = config.get_config_value("validation_data_directory")
    number_of_epochs = config.get_config_value("epochs", 10)
    eval_percentage = config.get_config_value("eval_percentage", 10)
    batch_size = config.get_config_value("batch_size", 32)
    num_workers = config.get_config_value("num_workers", 4)
    print_every_epoch = config.get_config_value("print_every_epoch", 1)
    checkpoint_path = config.get_config_value("checkpoint_path")

    print("\n" + "-"*50)
    print("Configuration Settings:")
    print("-"*50)
    print(f"Run Name                : {run_name}")
    print(f"Validation Directory    : {val_dir}")
    print(f"Number of Epochs        : {number_of_epochs}")
    print(f"Evaluation Percentage   : {eval_percentage}%")
    print(f"Batch Size              : {batch_size}")
    print(f"Number of Workers       : {num_workers}")
    print(f"Print Every Epoch       : {print_every_epoch}")
    print(f"Checkpoint Path         : {checkpoint_path}")
    print("-"*50)
    

    best_val_acc = 0.0
    epoch = 1
    train_loss = 0
    train_acc = 0
    train_total_images = 0
    train_time = 0
    train_latency = 0
    train_cpu_time = 0
    val_loss = 0
    val_acc = 0
    val_total_images = 0
    val_time = 0
    val_latency = 0
    val_cpu_time = 0

    base_model = model_manager()
    base_model.load_evaludation_dataset(val_dir, batch_size, num_workers)
    base_model.init_model(checkpoint_path)    # Load resnet 18 model along with weights from checkpoint
    

    if(config.load_run_configuration())!=None:
        epoch = int(config.get_last_config_value("epoch", 1)) + 1
        best_val_acc = config.get_last_config_value("best_val_acc", 0.0)

        print(f"Resuming training from epoch {epoch} with best val_acc {best_val_acc:.4f}")
                  

    while epoch <= number_of_epochs:
        print(f"\nEpoch {epoch}/{number_of_epochs}")

        val_loss, val_acc, val_total_images, val_time, val_latency, val_cpu_time = base_model.evaluate(eval_percentage)

        if (epoch) % print_every_epoch == 0:
            print(
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_acc:.4f} | "
                f"Val images: {val_total_images} | "
                f"Val time: {val_time:.2f}s | "
                f"Val latency: {val_latency:.4f}s | "
                f"Val CPU time: {val_cpu_time:.4f}s"
            )


        # Save configuration for the current run
        run_directory = config.get_config_value("run_directory")
        run_name = config.get_config_value("run_name")
        current_run_dir = f"{run_directory}/{run_name}"
        os.makedirs(current_run_dir, exist_ok=True) 


        if val_acc > best_val_acc:
            best_val_acc = val_acc

        config.save_run_configuration(
            epoch, 
            train_loss, 
            train_acc, 
            train_total_images, 
            train_time, 
            train_latency, 
            train_cpu_time,
            val_loss, 
            val_acc, 
            val_total_images, 
            val_time, 
            val_latency, 
            val_cpu_time,
            best_val_acc, 
            checkpoint_path
        )

        # Increment epoch counter
        epoch += 1


    print("\Evaluation finished.")
    print("Best validation accuracy:", best_val_acc)