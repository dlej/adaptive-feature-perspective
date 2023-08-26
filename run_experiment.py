import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import operator
import os
import pickle
import sys

# fix an nvlink error
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
from clu import metrics
from flax import linen as nn, struct
from flax.training import train_state
import optax

# from neural_tangents import linearize

import tensorflow as tf
import tensorflow_datasets as tfds


########################################
# network models
########################################


class CNN(nn.Module):
    """A simple CNN model."""

    width: int = 16

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.width, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=2 * self.width, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=4 * self.width, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=8 * self.width, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=8 * self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(features=8 * self.width)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class MLP(nn.Module):
    """A simple MLP model."""

    depth: int = 3
    width: int = 128

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        for _ in range(self.depth - 1):
            x = nn.Dense(features=self.width)(x)
            x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class MLPResNet(nn.Module):
    """A simple MLP model."""

    depth: int
    width: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        z = None
        for _ in range(self.depth - 1):
            if z is None:
                z = nn.Dense(features=self.width)(x)
            else:
                z = nn.Dense(features=self.width)(x + z)
            x = nn.relu(z)
        x = nn.Dense(features=10)(x)
        return x


########################################
# training
########################################


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(params, apply_fn, optimizer, learning_rate, momentum):
    """Creates an initial `TrainState`."""

    match optimizer:
        case "sgd":
            tx = optax.sgd(learning_rate, momentum)
        case "adam":
            tx = optax.adam(learning_rate, b1=momentum)

    return TrainState.create(
        apply_fn=apply_fn, params=params, tx=tx, metrics=Metrics.empty()
    )


def create_train_state_from_module(
    module, input_shape, rng, optimizer, learning_rate, momentum
):
    params = module.init(rng, jnp.ones(input_shape))["params"]
    return create_train_state(params, module.apply, optimizer, learning_rate, momentum)


def create_linearized_train_state_from_module(
    module, input_shape, rng, optimizer, learning_rate, momentum
):
    params_0 = module.init(rng, jnp.ones(input_shape))["params"]
    apply_fn_lin, params = linearize(module.apply, {"params": params_0})
    return create_train_state(params, apply_fn_lin, optimizer, learning_rate, momentum)


def create_factorized_train_state_from_module(
    module, input_shape, rng, optimizer, learning_rate, momentum
):
    params_0 = module.init(rng, jnp.ones(input_shape))["params"]
    apply_fn_fac, params = factorize(module.apply, {"params": params_0})
    return create_train_state(params, apply_fn_fac, optimizer, learning_rate, momentum)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch["label"], loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


########################################
# linearization
########################################


# modified from neural-tangents
def linearize(apply_fn, params):
    """Linearizes a function around a set of parameters."""

    params_0 = deepcopy(params)

    def f_lin(p, *args, **kwargs):
        dp = p["params"]["delta"]
        f_params_x, proj = jax.jvp(
            lambda param: apply_fn(param, *args, **kwargs), (params_0,), (dp,)
        )
        return tree_map(operator.add, f_params_x, proj)

    new_params = {"init": params, "delta": tree_map(jnp.zeros_like, params)}

    return f_lin, new_params


def unfactorize_tree(tree):
    return tree_map(
        unfactorize_leaf,
        tree,
        is_leaf=lambda x: "Dense_factorized" in x or "Conv_factorized" in x,
    )


def unfactorize_leaf(factorized_leaf):
    if "Dense_factorized" in factorized_leaf:
        factors = factorized_leaf["Dense_factorized"]
        return {
            "kernel": factors["M1"] @ (factors["kernel"] @ factors["M2"]),
            "bias": factors["M2"].T @ factors["bias"],
        }

    elif "Conv_factorized" in factorized_leaf:
        factors = factorized_leaf["Conv_factorized"]
        kernel = factors["kernel"]
        W = kernel.reshape(
            (kernel.shape[0] * kernel.shape[1], kernel.shape[2], kernel.shape[3])
        )
        MWMM = jnp.einsum(
            "ij,jkl,km,ln->imn", factors["M1"], W, factors["M2"], factors["M3"]
        )
        return {
            "kernel": MWMM.reshape(kernel.shape),
            "bias": factors["M3"].T @ factors["bias"],
        }

    else:
        return factorized_leaf


def factorize(apply_fn, params):
    """Linearizes a function around a set of parameters."""

    params_0 = deepcopy(params)

    def f_factorized(p, *args, **kwargs):
        dp = unfactorize_tree(p["params"]["delta"])

        f_params_x, proj = jax.jvp(
            lambda param: apply_fn(param, *args, **kwargs), (params_0,), (dp,)
        )
        return tree_map(operator.add, f_params_x, proj)

    delta = tree_map(jnp.zeros_like, params)
    for k, v in params["params"].items():
        if k.startswith("Dense"):
            kernel = v["kernel"]
            delta["params"][k] = {
                "Dense_factorized": {
                    "M1": jnp.eye(kernel.shape[0]),
                    "M2": jnp.eye(kernel.shape[1]),
                    "kernel": jnp.zeros(kernel.shape),
                    "bias": jnp.zeros(kernel.shape[1]),
                }
            }
        elif k.startswith("Conv"):
            kernel = v["kernel"]
            delta["params"][k] = {
                "Conv_factorized": {
                    "M1": jnp.eye(kernel.shape[0] * kernel.shape[1]),
                    "M2": jnp.eye(kernel.shape[2]),
                    "M3": jnp.eye(kernel.shape[3]),
                    "kernel": jnp.zeros(kernel.shape),
                    "bias": jnp.zeros(kernel.shape[3]),
                }
            }

    new_params = {"init": params, "delta": delta}

    return f_factorized, new_params


########################################
# utils
########################################


def get_datasets(name, num_epochs, batch_size, n_train=None):
    """Load MNIST train and test datasets into memory."""

    train_ds = tfds.load(name, split="train")
    test_ds = tfds.load(name, split="test")

    total_train = train_ds.cardinality().numpy().item()
    if n_train is None:
        n_train = total_train

    train_ds = train_ds.shuffle(buffer_size=total_train).take(n_train)

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255.0,
            "label": sample["label"],
        }
    )
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255.0,
            "label": sample["label"],
        }
    )

    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.shuffle(1024)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, test_ds


def save_model_params(params, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(params, f)


def log(msg):
    print(f"{datetime.now()}: {msg}")
    sys.stdout.flush()


def flushprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


########################################
# arguments
########################################


@dataclass
class ExperimentParams:
    id: str
    results_dir: str
    seed: str
    dataset: str
    n_train: int
    model: str
    depth: int
    width: int
    adaptivity: str
    optimizer: str
    max_epochs: int
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run an MNIST experiment.")
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Experiment ID",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Results directory",
    )
    parser.add_argument(
        "--seed", type=int, default=0, required=False, help="Random seed"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10"],
        default="mnist",
        required=False,
        help="Dataset (mnist or cifar10)",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=60000,
        required=False,
        help="Number of training examples",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "mlp", "mlpresnet"],
        default="mlp",
        required=False,
        help="Model architecture",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        required=False,
        help="Depth of the model",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        required=False,
        help="Width of the model",
    )
    parser.add_argument(
        "--adaptivity",
        type=str,
        choices=["full", "linear", "factorized"],
        default="full",
        required=False,
        help="Adaptivity of the model",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        required=False,
        help="Optimizer",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        required=False,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        required=False,
        help="Learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        required=False,
        help="Momentum",
    )

    args = parser.parse_args()
    return ExperimentParams(
        id=args.id,
        results_dir=args.results_dir,
        seed=args.seed,
        dataset=args.dataset,
        n_train=args.n_train,
        model=args.model,
        depth=args.depth,
        width=args.width,
        adaptivity=args.adaptivity,
        optimizer=args.optimizer,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )


########################################
# main
########################################

if __name__ == "__main__":
    exp = parse_arguments()

    flushprint("=" * 40)
    log(f"Running {exp.dataset} experiment")
    flushprint("Experiment parameters:")
    flushprint(exp)
    flushprint("=" * 40)

    flushprint("Creating model")
    match exp.model:
        case "cnn":
            model = CNN(width=exp.width)
        case "mlp":
            model = MLP(depth=exp.depth, width=exp.width)
        case "mlpresnet":
            model = MLPResNet(depth=exp.depth, width=exp.width)

    match exp.adaptivity:
        case "full":
            create_train_state_fn = create_train_state_from_module
        case "linear":
            create_train_state_fn = create_linearized_train_state_from_module
        case "factorized":
            create_train_state_fn = create_factorized_train_state_from_module

    match exp.dataset:
        case "mnist":
            input_shape = [1, 28, 28, 1]
        case "cifar10":
            input_shape = [1, 32, 32, 3]

    log("Creating train state")
    init_rng = jax.random.PRNGKey(exp.seed)
    state = create_train_state_fn(
        model, input_shape, init_rng, exp.optimizer, exp.learning_rate, exp.momentum
    )
    del init_rng  # Must not be used anymore.

    log("Loading dataset")
    tf.random.set_seed(exp.seed)
    train_ds, test_ds = get_datasets(
        exp.dataset, exp.max_epochs, exp.batch_size, exp.n_train
    )

    # since train_ds is replicated max_epochs times in get_datasets(), we divide by max_epochs
    num_steps_per_epoch = train_ds.cardinality().numpy() // exp.max_epochs
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    log("Beginning training")
    best_test_accuracy = 0.0
    best_params = None
    best_epoch = None

    # save initial model
    save_model_params(
        state.params,
        os.path.join(exp.results_dir, f"initial_params.pkl"),
    )

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run optimization steps over training batches and compute batch metrics
        state = train_step(
            state, batch
        )  # get updated train state (which contains the updated parameters)
        state = compute_metrics(state=state, batch=batch)  # aggregate batch metrics

        if (step + 1) % num_steps_per_epoch == 0:  # one training epoch has passed
            epoch = (step + 1) // num_steps_per_epoch
            log(f"Epoch {epoch}")

            for metric, value in state.metrics.compute().items():  # compute metrics
                metrics_history[f"train_{metric}"].append(value.item())
            # reset train_metrics for next training epoch
            state = state.replace(metrics=state.metrics.empty())

            # Compute metrics on the test set after each training epoch
            test_state = state
            for test_batch in test_ds.as_numpy_iterator():
                test_state = compute_metrics(state=test_state, batch=test_batch)

            for metric, value in test_state.metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value.item())

            flushprint(
                f"Train: "
                f"loss: {metrics_history['train_loss'][-1]}, "
                f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
            )
            flushprint(
                f"Test: "
                f"loss: {metrics_history['test_loss'][-1]}, "
                f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
            )

            # Save best model parameters
            if metrics_history["test_accuracy"][-1] > best_test_accuracy:
                best_test_accuracy = metrics_history["test_accuracy"][-1]
                best_params = state.params
                best_epoch = epoch

            # Stop training if test accuracy has not improved for 20 epochs
            if epoch > best_epoch + 20:
                break

    log("Training complete")
    flushprint(f"Best model epoch: {best_epoch}")
    flushprint(
        "Train: "
        f"loss: {metrics_history['train_loss'][best_epoch - 1]}, "
        f"accuracy: {metrics_history['train_accuracy'][best_epoch - 1] * 100}"
    )
    flushprint(
        "Test: "
        f"loss: {metrics_history['test_loss'][best_epoch - 1]}, "
        f"accuracy: {metrics_history['test_accuracy'][best_epoch - 1] * 100}"
    )

    # Save best parameters to disk
    save_model_params(
        best_params,
        os.path.join(exp.results_dir, f"best_params_epoch_{best_epoch}.pkl"),
    )

    # Save metrics
    with open(os.path.join(exp.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=4)
