
import os
import json
import csv
import datetime
class config_manager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_config_value(self, key, default=None):
        return self.config.get(key, default)

    def get_last_config_value(self, key, default=None):
        return self.last_run_config.get(key, default)


    def load_run_configuration(self):
        """Load run configuration from a JSON file."""
        run_directory = self.config.get("run_directory")
        run_name = self.config.get("run_name")
        current_run_dir = f"{run_directory}/{run_name}"
        run_config_path = f"{current_run_dir}/run_configuration.json"

        if not os.path.isfile(run_config_path):
            return None

        with open(run_config_path, 'r') as f:
            self.last_run_config = json.load(f)
        return self.last_run_config


    def save_run_configuration(self, epoch, 
                               train_loss, train_acc, train_total_images, train_time, train_latency, train_cpu_time,
                               val_loss, val_acc, epoch_avg_confidence, val_total_images, val_time, val_latency, val_cpu_time,
                               best_val_acc, checkpoint_path):
        run_directory = self.config.get("run_directory")
        run_name = self.config.get("run_name")
        current_run_dir = f"{run_directory}/{run_name}"

        os.makedirs(current_run_dir, exist_ok=True) 

        config_data = {
            "run_name": run_name,
            "epoch": epoch,
            "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_total_images": train_total_images,
            "train_time": train_time,
            "train_latency": train_latency,
            "train_cpu_time": train_cpu_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_avg_confidence": epoch_avg_confidence,
            "val_total_images": val_total_images,
            "val_time": val_time,
            "val_latency": val_latency,
            "val_cpu_time": val_cpu_time,
            "best_val_acc": best_val_acc,
            "checkpoint_path": checkpoint_path
        }

        run_config = f"{current_run_dir}/run_configuration.json"
        with open(run_config, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to {run_config}")

        csv_file = f"{current_run_dir}/run_stats.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=config_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(config_data)

        