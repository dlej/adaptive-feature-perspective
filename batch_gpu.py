# Initially generated by ChatGPT-4 on 2023-08-24

from datetime import datetime
import subprocess
import os
from time import sleep
import argparse
import json
from hashlib import sha256


def load_experiments_from_json(file_path, script, seeds=None):
    # allow multiple config files to be specified
    file_paths = file_path.split(",")
    if len(file_paths) > 1:
        experiments = []
        for file_path in file_paths:
            experiments.extend(
                load_experiments_from_json(file_path, script, seeds=seeds)
            )
        return experiments
    else:
        file_path = file_paths[0]

    with open(file_path, "r") as file:
        config = json.load(file)
    base = config["base"]
    base["config_file"] = os.path.basename(file_path)
    base["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base["script"] = os.path.basename(script)

    experiments = []
    for entry in config["configs"]:
        experiment = base.copy()
        experiment.update(entry)

        if seeds is not None:
            start, end = map(int, seeds.split(":"))
            for seed in range(start, end):
                experiment = experiment.copy()
                experiment["seed"] = seed
                experiments.append(experiment)
        else:
            experiments.append(experiment)

    return experiments


def format_timedelta(td):
    """Format a timedelta object as a string."""
    seconds = int(td.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def run_experiment(script, gpu_id, experiment, dry_run=False):
    """
    Run an experiment on a specified GPU.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("=" * 80)
    print(f"Running experiment {experiment} on GPU {gpu_id}")
    experiment_str = json.dumps(experiment).encode("utf-8")
    id = sha256(experiment_str).hexdigest()[:16]
    print(f"Experiment ID: {id}")

    results_dir = os.path.join(
        "results",
        f"{experiment['script']}.d",
        f"{experiment['config_file']}_{experiment['timestamp']}.d",
    )

    del experiment["script"]
    del experiment["config_file"]
    del experiment["timestamp"]

    # Construct your command. This might differ based on how you run your experiments.
    cmd = ["python", script]
    for key, value in experiment.items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--id", id])

    # Create a directory for the experiment's results
    exp_dir = os.path.join(results_dir, id)
    os.makedirs(exp_dir, exist_ok=True)

    cmd.extend(["--results_dir", exp_dir])

    with open(os.path.join(exp_dir, "cmd.txt"), "w") as file:
        file.write(f"CUDA_VISIBLE_DEVICES={gpu_id} " + " ".join(cmd))
    with open(os.path.join(exp_dir, "experiment.json"), "w") as file:
        json.dump(experiment, file, indent=4)
    with open(os.path.join(exp_dir, "out.log"), "w") as file:
        if dry_run:
            process = subprocess.Popen(["echo", ""], env=env, stdout=file, stderr=file)
        else:
            process = subprocess.Popen(cmd, env=env, stdout=file, stderr=file)

    return process


if __name__ == "__main__":
    batch_start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description="Run a batch of experiments from a JSON config file."
    )
    parser.add_argument(
        "script",
        type=str,
        help="Path to the experiment script to run",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON file containing experiment configurations",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for running experiments",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Colon-separated range of seeds to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate running the batch without actually running any experiments",
    )
    args = parser.parse_args()

    experiments = load_experiments_from_json(args.config_file, args.script, args.seeds)
    total_experiments = len(experiments)
    completed_experiments = 0
    available_gpus = list(range(args.gpus))
    running_experiments = []

    print(f"Running {len(experiments)} experiments on {args.gpus} GPUs")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    while experiments or running_experiments:
        # Assign available GPUs to experiments
        while experiments and available_gpus:
            gpu = available_gpus.pop()
            experiment = experiments.pop(0)
            process = run_experiment(args.script, gpu, experiment, args.dry_run)
            start_time = datetime.now()
            running_experiments.append((gpu, process, start_time))

        # Check for completed experiments and free up GPUs
        for gpu, process, start_time in running_experiments.copy():
            retcode = process.poll()
            if retcode is not None:
                completed_experiments += 1
                print("-" * 80)
                print(
                    f"Experiment on GPU {gpu} has finished in {format_timedelta(datetime.now() - start_time)}. {completed_experiments}/{total_experiments} experiments completed."
                )
                running_experiments.remove((gpu, process, start_time))
                available_gpus.append(gpu)

        # Wait for 1 second before checking again
        if experiments or running_experiments:
            sleep(1)

    print("=" * 80)
    print(f"Batch completed in {format_timedelta(datetime.now() - batch_start_time)}")
