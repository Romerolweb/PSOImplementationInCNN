# --- START OF FILE pso_cnn_cifar_unified_hpo.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import os
import logging
import time
import json
import itertools  # For Grid Search
import argparse  # For command-line arguments
import copy  # For deep copying configs if needed

# --- Constants and Configuration ---
RANDOM_SEED = 42
DATA_DIR = "./data_cifar_final"
LOG_FILE = "hpo_cnn_optimization.log"  # General log file
# Results file name will include the strategy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Reproducibility ---
def set_seed(seed):
    """Sets random seeds for reproducibility across libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

# --- Logging Configuration ---
# Remove existing handlers to avoid duplicates if re-running
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logging.info(f"Using device: {DEVICE}")
if DEVICE.type == "cpu":
    logging.warning("<<<<< RUNNING ON CPU - THIS WILL BE VERY SLOW! >>>>>")


# --- Custom Dataset Wrapper (No changes needed) ---
class RemappedDataset(Dataset):
    def __init__(self, original_dataset, indices_to_keep, class_map):
        self.original_dataset = original_dataset
        self.indices = indices_to_keep
        self.class_map = class_map
        self.remapped_labels = [
            self.class_map[self.original_dataset.targets[i]] for i in self.indices
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _ = self.original_dataset[original_idx]
        new_label = self.remapped_labels[idx]
        return image, new_label


# --- 1. Data Loading and Preprocessing (No changes needed) ---
def load_cifar10_subset(batch_size=64, data_dir=DATA_DIR, num_workers=2):
    logging.info("Loading and preprocessing CIFAR-10 subset (Cats vs Dogs)...")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logging.info(f"Created data directory: {data_dir}")
        trainset_full = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset_full = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        cat_dog_classes = [3, 5]
        logging.info(
            f"Filtering for classes: Cats({cat_dog_classes[0]}) vs Dogs({cat_dog_classes[1]})"
        )
        label_map = {old: new for new, old in enumerate(cat_dog_classes)}
        logging.info(f"Label map: {label_map}")
        train_indices = [
            i for i, lbl in enumerate(trainset_full.targets) if lbl in cat_dog_classes
        ]
        test_indices = [
            i for i, lbl in enumerate(testset_full.targets) if lbl in cat_dog_classes
        ]
        train_subset = RemappedDataset(trainset_full, train_indices, label_map)
        test_subset = RemappedDataset(testset_full, test_indices, label_map)
        trainloader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        testloader = DataLoader(
            test_subset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logging.info(
            f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}. Data loading complete."
        )
        return trainloader, testloader
    except Exception as e:
        logging.error(f"Data loading error: {e}", exc_info=True)
        raise


# --- 2. CNN Model Definition (No changes needed) ---
class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.fc_layer2 = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc_layer2(self.fc_layer1(self.conv_layer2(self.conv_layer1(x))))


# --- Utility Function for Model Evaluation (No changes needed) ---
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


# --- 3a. Core Training and Evaluation Function (NEW) ---
def train_and_evaluate_model(
    hyperparams,
    model_class,
    trainloader,
    testloader,
    device,
    num_epochs,
    metric_to_return="error",
):
    """
    Trains a new model instance with given hyperparameters and returns a performance metric.
    This is the core function called by different optimization strategies.

    Args:
        hyperparams (dict): Dictionary of hyperparameters (e.g., 'learning_rate', 'dropout_rate').
        model_class (nn.Module): The class of the model to instantiate.
        trainloader, testloader: DataLoaders.
        device: CPU or CUDA device.
        num_epochs (int): Number of epochs to train for this evaluation.
        metric_to_return (str): 'error' or 'accuracy'.

    Returns:
        float: The calculated metric (error or accuracy), or infinity/negative infinity on failure.
    """
    lr = hyperparams.get("learning_rate", 0.001)
    wd = hyperparams.get("weight_decay", 0.0001)
    dr = hyperparams.get("dropout_rate", 0.5)
    # Add extraction for other potential hyperparams like batch_size, num_filters etc.

    log_prefix = f"Eval (LR={lr:.2e}, WD={wd:.2e}, DR={dr:.2f})"  # Concise log prefix

    try:
        logging.debug(f"{log_prefix}: Starting evaluation...")
        model_eval = model_class(num_classes=2, dropout_rate=dr).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_eval.parameters(), lr=lr, weight_decay=wd)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Optional

        start_time = time.time()
        for epoch in range(num_epochs):
            model_eval.train()
            # Simplified training loop for evaluation purposes
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model_eval(images)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
            # scheduler.step() # Optional

        accuracy = evaluate_model(model_eval, testloader, device)
        error = 100.0 - accuracy
        eval_time = time.time() - start_time
        logging.debug(
            f"{log_prefix}: Done. Accuracy={accuracy:.2f}%, Error={error:.2f}%, Time={eval_time:.1f}s"
        )

        if metric_to_return == "accuracy":
            return accuracy
        else:  # Default to error
            return error

    except Exception as e:
        logging.error(f"{log_prefix}: EXCEPTION during evaluation: {e}", exc_info=False)
        # Return worst possible value based on metric
        return -float("inf") if metric_to_return == "accuracy" else float("inf")


# --- 3b. PSO Implementation (Modified to use core evaluation function) ---
class Particle:
    """Represents a particle in PSO."""

    def __init__(self, hyperparameter_config):
        # (Initialization remains the same as previous version)
        self.hyperparam_names = list(hyperparameter_config.keys())
        self.num_dimensions = len(self.hyperparam_names)
        self.bounds_min = np.array([v[0] for v in hyperparameter_config.values()])
        self.bounds_max = np.array([v[1] for v in hyperparameter_config.values()])
        self.param_types = [
            int if isinstance(lb, int) and isinstance(ub, int) else float
            for lb, ub in zip(self.bounds_min, self.bounds_max)
        ]
        self.position = np.zeros(self.num_dimensions)
        for i in range(self.num_dimensions):
            pos_val = random.uniform(self.bounds_min[i], self.bounds_max[i])
            self.position[i] = (
                int(round(pos_val)) if self.param_types[i] == int else pos_val
            )
        self.velocity = np.array(
            [
                random.uniform(
                    -(self.bounds_max[i] - self.bounds_min[i]) * 0.1,
                    (self.bounds_max[i] - self.bounds_min[i]) * 0.1,
                )
                for i in range(self.num_dimensions)
            ]
        )
        self.pbest_position = self.position.copy()
        self.pbest_value = float("inf")  # PSO minimizes error

    def _get_hyperparams_dict(self):
        params = {}
        for i, name in enumerate(self.hyperparam_names):
            params[name] = (
                int(round(self.position[i]))
                if self.param_types[i] == int
                else float(self.position[i])
            )
        return params

    def update_velocity(self, gbest_position, iw, c1, c2):  # Use specific param names
        r1 = np.random.rand(self.num_dimensions)
        r2 = np.random.rand(self.num_dimensions)
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = iw * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds_min, self.bounds_max)
        for i in range(self.num_dimensions):
            if self.param_types[i] == int:
                self.position[i] = int(round(self.position[i]))

    # This function is NO LONGER needed inside Particle, PSO will call the external one
    # def evaluate_fitness(...):
    #     ...


class PSO:
    """Manages the PSO optimization process."""

    def __init__(self, num_particles, hyperparameter_config, pso_params):
        self.num_particles = num_particles
        self.hyperparameter_config = hyperparameter_config
        self.pso_params = pso_params
        self.hyperparam_names = list(hyperparameter_config.keys())
        self.num_dimensions = len(self.hyperparam_names)
        logging.info("Initializing PSO...")
        logging.info(f"  Number of particles: {self.num_particles}")
        logging.info(
            f"  Optimizing {self.num_dimensions} hyperparameters: {self.hyperparam_names}"
        )
        self.particles = [Particle(hyperparameter_config) for _ in range(num_particles)]
        self.gbest_position = None
        self.gbest_value = float("inf")  # PSO minimizes error
        self.gbest_hyperparams_dict = {}
        logging.info("PSO Initialization complete.")
        self.iteration_log = []

    def optimize(
        self,
        model_class,
        trainloader,
        testloader,
        num_iterations,
        num_epochs_per_eval,
        device,
    ):
        """Runs the main PSO loop using the external evaluation function."""
        logging.info(
            f"Starting PSO: {num_iterations} iterations, {num_epochs_per_eval} epochs/eval."
        )
        total_start_time = time.time()

        iw = self.pso_params["inertia_weight"]
        c1 = self.pso_params["cognitive_coeff"]
        c2 = self.pso_params["social_coeff"]

        for iteration in range(num_iterations):
            iter_start_time = time.time()
            logging.info(f"--- PSO Iteration {iteration + 1}/{num_iterations} ---")
            particle_errors = []

            for i, particle in enumerate(self.particles):
                logging.debug(f"  Evaluating Particle {i+1}/{self.num_particles}...")
                hyperparams_dict = particle._get_hyperparams_dict()

                # *** CALL THE EXTERNAL EVALUATION FUNCTION ***
                error = train_and_evaluate_model(
                    hyperparams=hyperparams_dict,
                    model_class=model_class,  # Need to know which model to train
                    trainloader=trainloader,
                    testloader=testloader,
                    device=device,
                    num_epochs=num_epochs_per_eval,
                    metric_to_return="error",  # PSO minimizes error
                )
                particle_errors.append(error)

                # Update particle's best based on the returned error
                if error < particle.pbest_value:
                    particle.pbest_value = error
                    particle.pbest_position = particle.position.copy()

                # Update global best (using particle's pbest value)
                if particle.pbest_value < self.gbest_value:
                    old_gbest_value = self.gbest_value
                    self.gbest_value = particle.pbest_value
                    self.gbest_position = particle.pbest_position.copy()
                    self.gbest_hyperparams_dict = (
                        particle._get_hyperparams_dict()
                    )  # Get dict from best particle
                    logging.info(
                        f"  *** PSO New Global Best Found (Particle {i+1})! ***"
                    )
                    logging.info(
                        f"    Prev Best Error: {old_gbest_value:.4f} -> New Best Error: {self.gbest_value:.4f}"
                    )
                    logging.info(f"    Best Hyperparameters:")
                    for name, value in self.gbest_hyperparams_dict.items():
                        log_val = (
                            int(value)
                            if isinstance(value, (int, np.integer))
                            else f"{value:.6f}"
                        )
                        logging.info(f"      - {name}: {log_val}")

            # Initialize gBest after first iteration if needed
            if iteration == 0 and self.gbest_position is None:
                valid_pBests = [
                    p.pbest_value
                    for p in self.particles
                    if p.pbest_value != float("inf")
                ]
                if valid_pBests:
                    best_initial_particle = min(
                        self.particles, key=lambda p: p.pbest_value
                    )
                    self.gbest_value = best_initial_particle.pbest_value
                    self.gbest_position = best_initial_particle.pbest_position.copy()
                    self.gbest_hyperparams_dict = (
                        best_initial_particle._get_hyperparams_dict()
                    )
                    logging.warning(
                        f"Initialized PSO Global Best from best initial particle pBest (Error: {self.gbest_value:.4f})"
                    )
                else:
                    logging.error(
                        "All particles failed initial PSO evaluation. Stopping."
                    )
                    return None, float("inf")  # Stop if no particle succeeded

            # Update Velocities and Positions
            if self.gbest_position is not None:
                for particle in self.particles:
                    particle.update_velocity(self.gbest_position, iw, c1, c2)
                    particle.update_position()
            else:
                logging.warning("Skipping PSO particle update: gbest_position is None.")

            iter_time = time.time() - iter_start_time
            valid_errors = [e for e in particle_errors if e != float("inf")]
            avg_iter_error = np.mean(valid_errors) if valid_errors else float("nan")
            logging.info(
                f"--- PSO Iteration {iteration + 1} Complete. Avg Error: {avg_iter_error:.4f}. Best Error So Far: {self.gbest_value:.4f}. Time: {iter_time:.1f}s ---"
            )

            # Log iteration summary (using gbest_hyperparams_dict)
            iter_summary = {
                "iteration": iteration + 1,
                "best_error_so_far": self.gbest_value,
                "avg_iteration_error": avg_iter_error,
                "best_hyperparams_so_far": self.gbest_hyperparams_dict,
                "iteration_time_s": iter_time,
            }
            self.iteration_log.append(iter_summary)

        # --- End of PSO Loop ---
        total_time = time.time() - total_start_time
        logging.info(f"--- PSO Optimization Finished ({num_iterations} Iterations) ---")
        logging.info(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

        if self.gbest_position is not None:
            logging.info(f"Final Best Error (PSO): {self.gbest_value:.4f}")
            logging.info(
                f"Final Best Hyperparameters (PSO): {self.gbest_hyperparams_dict}"
            )
            self.save_results(self.gbest_hyperparams_dict, self.gbest_value)
            return self.gbest_hyperparams_dict, self.gbest_value
        else:
            logging.error("PSO finished without finding a valid global best position.")
            return None, float("inf")

    def save_results(self, best_hyperparams, best_error):
        """Saves PSO optimization results to JSON."""
        results_filename = "pso_search_results.json"  # PSO specific results file
        results = {
            "search_strategy": "pso",
            "best_error": best_error,
            "best_accuracy": 100.0 - best_error if best_error != float("inf") else 0.0,
            "best_hyperparameters": best_hyperparams,
            "pso_config": self.pso_params,
            "hyperparameter_bounds": self.hyperparameter_config,
            "iteration_log": self.iteration_log,
        }
        try:

            def convert_numpy(obj):  # Helper for JSON serialization
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj

            results_serializable = convert_numpy(results)
            with open(results_filename, "w") as f:
                json.dump(results_serializable, f, indent=4, sort_keys=True)
            logging.info(f"PSO results saved to {results_filename}")
        except Exception as e:
            logging.error(
                f"Failed to save PSO results to {results_filename}: {e}", exc_info=True
            )


# --- 3c. Grid/Random Search Implementation (Modified to use core evaluation function) ---
class HyperparameterOptimizer:
    """Performs Grid Search or Random Search using the core evaluation function."""

    def __init__(
        self,
        model_class,
        param_space,
        trainloader,
        testloader,
        device,
        num_epochs_per_eval,
        metric="accuracy",
        direction="maximize",
    ):
        self.model_class = model_class
        self.param_space = param_space
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.num_epochs_per_eval = num_epochs_per_eval
        self.metric = metric.lower()
        self.direction = direction.lower()
        if self.direction not in ["maximize", "minimize"]:
            raise ValueError("Dir must be max/min")
        if self.metric not in ["accuracy", "error"]:
            raise ValueError("Metric must be acc/err")
        self.best_params_ = None
        self.best_score_ = (
            -float("inf") if self.direction == "maximize" else float("inf")
        )
        self.results_log_ = []
        logging.info("HyperparameterOptimizer (Grid/Random) initialized.")
        logging.info(f"  Optimizing metric: {self.metric} ({self.direction})")
        logging.info(f"  Epochs per evaluation: {self.num_epochs_per_eval}")
        logging.info(f"  Parameter Space: {self.param_space}")

    def _generate_grid_combinations(self):
        keys = list(self.param_space.keys())
        value_lists = list(self.param_space.values())
        combinations = list(itertools.product(*value_lists))
        param_dicts = [dict(zip(keys, combo)) for combo in combinations]
        logging.info(f"Generated {len(param_dicts)} combinations for Grid Search.")
        return param_dicts

    def _generate_random_combination(self):
        params = {}
        for name, space_def in self.param_space.items():
            try:  # Add try-except for robustness
                if (
                    isinstance(space_def, (list, tuple))
                    and len(space_def) == 2
                    and isinstance(space_def[0], (int, float))
                ):
                    min_val, max_val = space_def
                    is_int = isinstance(min_val, int) and isinstance(max_val, int)
                    is_log = (
                        ("learning_rate" in name or "weight_decay" in name)
                        and min_val > 0
                        and max_val > 0
                    )

                    if is_log and not is_int:
                        log_min, log_max = np.log(min_val), np.log(max_val)
                        params[name] = float(np.exp(random.uniform(log_min, log_max)))
                    elif is_int:
                        params[name] = random.randint(min_val, max_val)
                    else:
                        params[name] = random.uniform(min_val, max_val)
                elif isinstance(space_def, list):
                    params[name] = random.choice(space_def)
                else:
                    logging.warning(
                        f"Unsupported space def for '{name}': {space_def}. Using default if available."
                    )
            except Exception as e:
                logging.error(
                    f"Error generating random param for '{name}' with space {space_def}: {e}"
                )
                params[name] = None  # Indicate failure
        return params

    # This method now simply calls the external evaluation function
    def _evaluate_combination(self, hyperparams):
        """Calls the standalone evaluation function."""
        return train_and_evaluate_model(
            hyperparams=hyperparams,
            model_class=self.model_class,
            trainloader=self.trainloader,
            testloader=self.testloader,
            device=self.device,
            num_epochs=self.num_epochs_per_eval,
            metric_to_return=self.metric,
        )

    def search(self, strategy="grid", n_trials=20):
        """Executes the Grid or Random search loop."""
        logging.info(f"Starting {strategy.capitalize()} Search...")
        search_start_time = time.time()
        if strategy == "grid":
            combinations_to_test = self._generate_grid_combinations()
        elif strategy == "random":
            combinations_to_test = [
                self._generate_random_combination() for _ in range(n_trials)
            ]
            logging.info(f"Testing {n_trials} random combinations.")
        else:
            raise ValueError("Strategy must be 'grid' or 'random'")
        total_combinations = len(combinations_to_test)

        for i, params in enumerate(combinations_to_test):
            logging.info(
                f"--- {strategy.capitalize()} Trial {i+1}/{total_combinations} ---"
            )
            logging.info(f"  Params: {params}")
            if None in params.values():  # Skip if param generation failed
                logging.warning("Skipping trial due to parameter generation error.")
                self.results_log_.append({"params": params, self.metric: float("nan")})
                continue

            score = self._evaluate_combination(params)
            logging.info(f"  Trial Result: {self.metric}={score:.4f}")
            is_better = (self.direction == "maximize" and score > self.best_score_) or (
                self.direction == "minimize" and score < self.best_score_
            )
            if is_better and score not in [
                float("inf"),
                -float("inf"),
            ]:  # Ensure score is valid
                self.best_score_ = score
                self.best_params_ = params
                logging.info(
                    f"  *** New Best Score Found: {self.best_score_:.4f} ({self.metric}) ***"
                )
            self.results_log_.append({"params": params, self.metric: score})

        search_time = time.time() - search_start_time
        logging.info(f"--- {strategy.capitalize()} Search Complete ---")
        logging.info(f"Total Time: {search_time:.1f}s ({search_time/60:.1f} minutes)")
        if self.best_params_ is not None:
            logging.info(f"Best {self.metric} found: {self.best_score_:.4f}")
            logging.info(f"Best Hyperparameters: {self.best_params_}")
            self.save_search_results(
                strategy, n_trials if strategy == "random" else total_combinations
            )
            return self.best_params_, self.best_score_
        else:
            return None, self.best_score_

    def save_search_results(self, strategy, n_trials):
        """Saves Grid/Random search results to JSON."""
        results_filename = f"{strategy}_search_results.json"
        output = {
            "search_strategy": strategy,
            "optimized_metric": self.metric,
            "optimization_direction": self.direction,
            "best_score": self.best_score_,
            "best_hyperparameters": self.best_params_,
            "parameter_space": self.param_space,
            "num_epochs_per_eval": self.num_epochs_per_eval,
            "results_log": self.results_log_,
        }
        if strategy == "random":
            output["n_trials"] = n_trials
        try:

            def convert_types(obj):  # Helper for JSON serialization
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                return obj

            serializable_output = convert_types(output)
            with open(results_filename, "w") as f:
                json.dump(serializable_output, f, indent=4, sort_keys=True)
            logging.info(f"Search results saved to {results_filename}")
        except Exception as e:
            logging.error(f"Failed to save search results: {e}", exc_info=True)


# --- 5. Main Execution with Argument Parsing ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for CatDogCNN using PSO, Grid Search, or Random Search."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="pso",
        choices=["pso", "grid", "random"],
        help="Optimization strategy to use ('pso', 'grid', 'random'). Default: pso",
    )
    parser.add_argument(
        "--epochs_per_eval",
        type=int,
        default=7,  # Increased default
        help="Number of epochs to train CNN during each evaluation step. Default: 7",
    )
    # PSO specific args
    parser.add_argument(
        "--pso_particles",
        type=int,
        default=20,
        help="Number of particles for PSO. Default: 20",
    )
    parser.add_argument(
        "--pso_iterations",
        type=int,
        default=15,
        help="Number of iterations for PSO. Default: 15",
    )
    # Random Search specific args
    parser.add_argument(
        "--random_trials",
        type=int,
        default=30,
        help="Number of trials for Random Search. Default: 30",
    )
    # General args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loading. Default: 64",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for DataLoader. Default: 2",
    )

    args = parser.parse_args()

    try:
        # --- Shared Configuration ---
        MODEL_CLASS = CatDogCNN  # The model we are optimizing

        # --- Define Hyperparameter Space (adjust as needed) ---
        # Common space definition suitable for all methods
        hyperparameter_config_space = {
            # Name: (Lower Bound, Upper Bound) for Random/PSO or [List] for Grid
            "learning_rate": (1e-5, 1e-2),
            "weight_decay": (1e-6, 1e-3),
            "dropout_rate": (0.1, 0.6),
            # Add 'batch_size': (16, 128) # Example if optimizing batch size (needs handling in train_and_evaluate)
        }

        # Grid search needs specific values - create if strategy is grid
        grid_search_param_space = {
            "learning_rate": [0.005, 0.001, 0.0005, 1e-4],  # Example grid values
            "weight_decay": [1e-6, 1e-5, 1e-4],
            "dropout_rate": [0.25, 0.4, 0.55],
        }

        # --- Data Loading ---
        trainloader, testloader = load_cifar10_subset(
            batch_size=args.batch_size, num_workers=args.num_workers
        )

        # --- Select and Run Optimizer ---
        best_hyperparams = None
        best_score = None  # Use score appropriate for direction (accuracy or error)

        if args.strategy == "pso":
            pso_params = {  # Gather PSO specific params
                "num_particles": args.pso_particles,
                "num_iterations": args.pso_iterations,
                "inertia_weight": 0.7,  # Could make these args too
                "cognitive_coeff": 1.5,
                "social_coeff": 1.5,
            }
            pso_optimizer = PSO(
                num_particles=pso_params["num_particles"],
                hyperparameter_config=hyperparameter_config_space,  # Use range definition
                pso_params=pso_params,
            )
            # PSO minimizes error by default in this implementation
            best_hyperparams, best_error = pso_optimizer.optimize(
                model_class=MODEL_CLASS,
                trainloader=trainloader,
                testloader=testloader,
                num_iterations=pso_params["num_iterations"],
                num_epochs_per_eval=args.epochs_per_eval,
                device=DEVICE,
            )
            if best_error is not None:
                best_score = (
                    100.0 - best_error
                )  # Convert error to accuracy for final report

        elif args.strategy == "grid" or args.strategy == "random":
            hpo = HyperparameterOptimizer(
                model_class=MODEL_CLASS,
                # Use specific grid values if grid search, otherwise use ranges for random
                param_space=(
                    grid_search_param_space
                    if args.strategy == "grid"
                    else hyperparameter_config_space
                ),
                trainloader=trainloader,
                testloader=testloader,
                device=DEVICE,
                num_epochs_per_eval=args.epochs_per_eval,
                metric="accuracy",  # Let's optimize accuracy directly
                direction="maximize",
            )
            best_hyperparams, best_score = hpo.search(
                strategy=args.strategy,
                n_trials=args.random_trials,  # Only used if strategy is 'random'
            )

        # --- Final Summary ---
        logging.info("=" * 60)
        logging.info(
            f"Hyperparameter Optimization using [{args.strategy.upper()}] Strategy Complete!"
        )
        if best_hyperparams is not None:
            logging.info(f"Best Hyperparameters Found:")
            for name, value in best_hyperparams.items():
                log_val = (
                    int(value)
                    if isinstance(value, (int, np.integer))
                    else f"{value:.6f}"
                )
                logging.info(f"  - {name}: {log_val}")
            # Report accuracy consistently
            final_accuracy = (
                best_score
                if args.strategy != "pso"
                else (100.0 - best_score if best_score != float("inf") else 0.0)
            )
            logging.info(
                f"Best Test Accuracy Achieved during Optimization: {final_accuracy:.2f}%"
            )
            results_file = f"{args.strategy}_search_results.json"
            logging.info(f"Details saved in: {LOG_FILE} and {results_file}")
            logging.info(
                "Consider training a final model with these parameters for more epochs."
            )
        else:
            logging.warning(
                f"{args.strategy.upper()} Search did not find a valid best combination."
            )
        logging.info("=" * 60)

    except Exception as e:
        logging.error(
            f"An critical error occurred in the main execution block: {e}",
            exc_info=True,
        )

# --- END OF FILE pso_cnn_cifar_unified_hpo.py ---
