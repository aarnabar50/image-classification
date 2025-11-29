import datetime
import os

from libs.auto_encoder_lib import autoencoder_manager
from libs.config_manager_lib import config_manager


def run_auto_encoder_training(config_path="./src/configurations/config_training_auto_encoder.json"):
    """Function to run training epochs for denoiser auto encoder training."""

    print("\n" + "="*50)
    print("Running the autoencoder training...")

    # Initialize variables
    best_loss = float('inf')  # Start with infinity so first validation will be saved
    checkpoint_path = ""
    epoch = 1
    train_loss = 0
    train_acc = 0
    train_total_images = 0
    train_time = 0
    train_latency = 0
    train_cpu_time = 0
    val_loss = 0
    val_acc = 0
    val_avg_confidence = 0
    val_total_images = 0
    val_time = 0
    val_latency = 0
    val_cpu_time = 0

    # Load configuration
    config = config_manager(config_path)
    run_name = config.get_config_value("run_name")
    clean_data_dir = config.get_config_value("clean_data_dir")
    adversarial_data_dir   = config.get_config_value("adversarial_data_dir")
    number_of_epochs = config.get_config_value("epochs", 10)
    training_percentage = config.get_config_value("training_percentage", 10)
    eval_percentage = config.get_config_value("eval_percentage", 10)
    batch_size = config.get_config_value("batch_size", 32)
    num_workers = config.get_config_value("num_workers", 4)
    print_every_epoch = config.get_config_value("print_every_epoch", 1)
    latent_dim = config.get_config_value("latent_dim", 128)

    print("\n" + "-"*50)
    print("Configuration Settings:")
    print("-"*50)
    print(f"Run Name                    : {run_name}")
    print(f"Clean Data Directory        : {clean_data_dir}")
    print(f"Adversarial Data Directory  : {adversarial_data_dir}")
    print(f"Number of Epochs            : {number_of_epochs}")
    print(f"Training Percentage         : {training_percentage}%")
    print(f"Evaluation Percentage       : {eval_percentage}%")
    print(f"Batch Size                  : {batch_size}")
    print(f"Number of Workers           : {num_workers}")
    print(f"Print Every Epoch           : {print_every_epoch}")
    print(f"Latent Dimension            : {latent_dim}")
    print("-"*50)
    

    autoencoder_model = autoencoder_manager()

    if(config.load_run_configuration())!=None:
        epoch = int(config.get_last_config_value("epoch", 1)) + 1
        best_loss = config.get_last_config_value("best_val_acc", float('inf')) # For autoencoders, lower loss is better. The key name is kept same for consistency.
        checkpoint_path = config.get_last_config_value("checkpoint_path", "")

        autoencoder_model.init_autoencoder(latent_dim, checkpoint_path)    # Load resnet 18 model along with weights from checkpoint

        print(f"Resuming training from epoch {epoch} with best val_loss {best_loss:.4f} and weights from {checkpoint_path}")
    else:
        autoencoder_model.init_autoencoder(latent_dim)    # Load resnet 18 model with random weights
        print("Starting fresh training with random initialized weights.")


                  

    while epoch <= number_of_epochs:
        print(f"\nEpoch {epoch}/{number_of_epochs}")

        
        train_loss, train_total_images, train_time, train_latency, train_cpu_time = autoencoder_model.train_autoencoder_with_adversarial(
            adversarial_data_dir = adversarial_data_dir, 
            clean_data_dir = clean_data_dir,             
            batch_size = batch_size, 
            num_workers = num_workers,
            training_percentage = training_percentage
        )
        
        val_loss, val_total_images, val_time, val_latency, val_cpu_time = autoencoder_model.evaluate_autoencoder(
            adversarial_data_dir = adversarial_data_dir, 
            clean_data_dir = clean_data_dir, 
            eval_percentage = eval_percentage, 
            batch_size = batch_size, 
            num_workers = num_workers
        )

        if (epoch) % print_every_epoch == 0:
            print(
            f"Train loss: {train_loss:.4f} | "
            f"Train time: {train_time:.2f}s | "
            f"Train latency: {train_latency:.4f}s | "
            f"Train CPU time: {train_cpu_time:.4f}s | "
            f"Val loss: {val_loss:.4f} | "
            f"Val time: {val_time:.2f}s | "
            f"Val latency: {val_latency:.4f}s | "
            f"Val CPU time: {val_cpu_time:.4f}s"
            )


        # Save configuration for the current run
        run_directory = config.get_config_value("run_directory")
        run_name = config.get_config_value("run_name")
        current_run_dir = f"{run_directory}/{run_name}"
        os.makedirs(current_run_dir, exist_ok=True) 


        # Save the best model
        if val_loss < best_loss:  # For autoencoders, lower loss is better
            best_loss = val_loss

            data_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            checkpoint_path = f"{current_run_dir}/state_{data_time_str}.pth"

            autoencoder_model.save_autoencoder_checkpoint(epoch, val_loss, checkpoint_path)

        
        config.save_run_configuration(
            epoch = epoch, 
            train_loss = train_loss, 
            train_total_images = train_total_images,
            train_time = train_time, 
            train_latency = train_latency, 
            train_cpu_time = train_cpu_time,
            val_loss = val_loss, 
            val_total_images = val_total_images,
            val_time = val_time, 
            val_latency = val_latency, 
            val_cpu_time = val_cpu_time,
            best_val_acc = best_loss, 
            checkpoint_path = checkpoint_path
        )

        # Increment epoch counter
        epoch += 1


    print("\nTraining finished.")
    print("Best validation loss:", best_loss)



def denoise_with_autoencoder(config_path="./src/configurations/config_denoise_with_auto_encoder.json"):
    """Function to run the auto encoder model in inference mode."""

    config = config_manager(config_path)
    adversarial_data_dir = config.get_config_value("adversarial_data_dir")
    denoised_data_dir = config.get_config_value("denoised_data_dir")
    checkpoint_path = config.get_config_value("checkpoint_path")
    latent_dim = config.get_config_value("latent_dim", 128)

    print("\n" + "-"*50)
    print("Configuration Settings:")
    print("-"*50)
    print(f"Adversarial Data Directory  : {adversarial_data_dir}")
    print(f"Denoised Data Directory     : {denoised_data_dir}")
    print(f"Checkpoint Path             : {checkpoint_path}")
    print(f"Latent Dimension            : {latent_dim}")
    print("-"*50)
        

    autoencoder_model = autoencoder_manager()
    autoencoder_model.init_autoencoder(latent_dim, checkpoint_path) 

    autoencoder_model.denoise_with_autoencoder(
        input_dir = adversarial_data_dir, 
        output_dir = denoised_data_dir
    )

    print(f"\nDenoised images saved to {denoised_data_dir}")