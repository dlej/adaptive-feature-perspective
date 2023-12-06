# Experiments for "An Adaptive Tangent Feature Perspective of Neural Networks"

This project contains the source code necessary to run the experiments for the paper

Daniel LeJeune and Sina Alemohammad. An Adaptive Tangent Feature Perspective of Neural Networks. 1st Annual Conference on Parsimony and Learning, 2024.

Figure 1 and 3 experiments are run with `torch_experiment.py`

Figure 4 experiments are run using
```bash
$ python batch_gpu.py run_experiment.py configs/mnist_full.json,configs/mnist_linear.json,configs/mnist_factorized.json,configs/cifar10_full.json,configs/cifar10_linear.json,configs/cifar10_factorized.json --gpus 8 --seeds 0:10
```

Afterwards, collect results by running `Collect results.ipynb` and generate figures using `Figures.ipynb`.