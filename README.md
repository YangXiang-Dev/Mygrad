# MyGrad

A from-scratch neural network with automatic differentiation, built by studying [Karpathy's micrograd](https://github.com/karpathy/micrograd).

## What's in here

- **engine.py** — `Value` class (autograd engine), `Neuron`, `Layer`, `MLP`
- **viz_server.py** — Real-time training visualization with local HTTP server
- **viz.html** — Interactive web UI (used by viz_server)
- **index.html** — Standalone SPA version (loads pre-exported data)
- **export_data.py** — Exports training data to JSON for the SPA
- **training_data.json** — 300 epochs of training data (decision boundaries on a 49×70 grid)

## Quick start

Train and visualize in real-time:

```bash
python viz_server.py
```

Opens `http://localhost:8085` with an interactive slider to scrub through training steps.

Or view the pre-trained SPA:

```bash
python -m http.server 8085
# open http://localhost:8085/index.html
```

## The model

- **Task**: Binary classification on `sklearn.datasets.make_moons` (100 samples)
- **Architecture**: MLP with layers [2, 10, 10, 1] and ReLU activations
- **Loss**: MSE
- **Optimizer**: SGD, lr=0.05, batch_size=16
- **Result**: 100% accuracy after 300 epochs

## Features

- Scroll / arrow keys / slider to step through training epochs
- Decision boundary updates every single step
- Loss and accuracy charts with step highlight
