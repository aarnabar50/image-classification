import datetime
import os

from libs.model_manager_lib import model_manager
from libs.config_manager_lib import config_manager


def run_model_training(config_path="./src/configurations/config_training_baseline.json", Training_Type="Baseline"):
    """Function to run training epochs for adversarial training."""

    print("\n" + "="*50)
    print("Running the model training...")

    # Initialize variables
    best_val_acc = 0.0
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
    train_dir = config.get_config_value("training_data_directory")
    val_dir   = config.get_config_value("validation_data_directory")
    number_of_epochs = config.get_config_value("epochs", 10)
    training_percentage = config.get_config_value("training_percentage", 10)
    eval_percentage = config.get_config_value("eval_percentage", 10)
    batch_size = config.get_config_value("batch_size", 32)
    num_workers = config.get_config_value("num_workers", 4)
    print_every_epoch = config.get_config_value("print_every_epoch", 1)
    print(f"Training Type          : {Training_Type}")
    if Training_Type=="YOPO":
        epsilon= config.get_config_value("epsilon", 0.03)
        num_steps= config.get_config_value("num_steps", 5)
        step_size= config.get_config_value("step_size", 0.007)
        checkpoint_path = config.get_config_value("checkpoint_path")


    print("\n" + "-"*50)
    print("Configuration Settings:")
    print("-"*50)
    print(f"Run Name                : {run_name}")
    print(f"Training Directory      : {train_dir}")
    print(f"Validation Directory    : {val_dir}")
    print(f"Number of Epochs        : {number_of_epochs}")
    print(f"Training Percentage     : {training_percentage}%")
    print(f"Evaluation Percentage   : {eval_percentage}%")
    print(f"Batch Size              : {batch_size}")
    print(f"Number of Workers       : {num_workers}")
    print(f"Print Every Epoch       : {print_every_epoch}")
    if Training_Type=="YOPO":
        print(f"Epsilon                : {epsilon}")
        print(f"Number of Steps        : {num_steps}")
        print(f"Step Size              : {step_size}")
        print(f"Checkpoint Path        : {checkpoint_path}")
    print("-"*50)
    

    base_model = model_manager()
    base_model.load_datasets(train_dir, val_dir, batch_size, num_workers)

    if(config.load_run_configuration())!=None:
        epoch = int(config.get_last_config_value("epoch", 1)) + 1
        best_val_acc = config.get_last_config_value("best_val_acc", 0.0)
        checkpoint_path = config.get_last_config_value("checkpoint_path", "")

        base_model.init_model(checkpoint_path)    # Load resnet 18 model along with weights from checkpoint

        print(f"Resuming training from epoch {epoch} with best val_acc {best_val_acc:.4f} and weights from {checkpoint_path}")
    else:
        if Training_Type=="YOPO":
            base_model.init_model(checkpoint_path)
            print(f"This is an YOPO training. Loaded initial weights from {checkpoint_path}")
        else:
            base_model.init_model()    # Load resnet 18 model with random weights
            print("Starting fresh training with random initialized weights.")


                  

    while epoch <= number_of_epochs:
        print(f"\nEpoch {epoch}/{number_of_epochs}")

        if Training_Type=="YOPO":
            train_loss, train_acc, train_total_images, train_time, train_latency, train_cpu_time = base_model.train_yopo(training_percentage, epsilon, num_steps, step_size)
        else:
            train_loss, train_acc, train_total_images, train_time, train_latency, train_cpu_time = base_model.train(training_percentage)

        val_loss, val_acc, val_avg_confidence, val_total_images, val_time, val_latency, val_cpu_time = base_model.evaluate(eval_percentage)

        if (epoch) % print_every_epoch == 0:
            print(
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_acc:.4f} | "
                f"Train images: {train_total_images} | "
                f"Train time: {train_time:.2f}s | "
                f"Train latency: {train_latency:.4f}s | "
                f"Train CPU time: {train_cpu_time:.4f}s | "
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


        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            data_time_str =  datetime.datetime.now().strftime("%Y%m%d_%H%M")
            checkpoint_path = f"{current_run_dir}/state_{data_time_str}.pth"

            base_model.save_checkpoint(epoch, val_acc, checkpoint_path)

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
            val_avg_confidence,
            val_total_images, 
            val_time, 
            val_latency, 
            val_cpu_time,
            best_val_acc, 
            checkpoint_path
        )

        # Increment epoch counter
        epoch += 1


    print("\nTraining finished.")
    print("Best validation accuracy:", best_val_acc)

def run_inference_mode():
    """Function to run the model in inference mode."""

    config = config_manager("./src/configurations/config_inference.json")
    checkpoint_path = config.get_config_value("checkpoint_path")

    print("\n" + "-"*50)
    print("Configuration Settings:")
    print("-"*50)
    print(f"Checkpoint Path         : {checkpoint_path}")
    print("-"*50)
        

    infer_model = model_manager()
    infer_model.init_model(checkpoint_path)
    
    while True:
        image_path = input("\nEnter the image path (or 'q' to quit): ").strip()
        
        if image_path.lower() == 'q':
            print("Exiting inference mode...")
            break
        
        try:
            # Run inference on the image
            prediction, confidence = infer_model.infer(image_path)
            print(f"Prediction: {prediction} with confidence {confidence:.4f}")
        except Exception as e:
            print(f"Error processing image: {e}")
