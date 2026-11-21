# Handwritten Digit Recognition with a Neural Network

An academic Artificial Intelligence project developed in Java to distinguish handwritten **2s and 3s** using a Multilayer Perceptron (MLP). The neural network, including forward propagation and backpropagation, was implemented without a machine learning framework.

**Best reported classification accuracy: 97.625%** during project experimentation (see the project report). This result is not a benchmark reproduced automatically by the repository.

## Project overview

The task is binary classification of 20 × 20-pixel handwritten digit images. The implementation covers model construction, training, preprocessing, parameter saving/loading, and inference.

### Neural network architecture

- **Input:** 400 features (20 × 20 grayscale pixels)
- **Hidden layer:** 32 neurons, LeakyReLU activation
- **Output layer:** 1 neuron, Sigmoid activation
- **Architecture:** `400 -> 32 -> 1`
- **Training:** backpropagation with gradient descent and mini-batches
- **Loss used for monitoring:** Mean Squared Error (MSE)
- **Weight initialization:** He-style uniform initialization

## Preprocessing and experimentation

The project explored different neural network architectures, learning rates, batch sizes, activation functions, normalization strategies, and dataset compositions. Experiments also used MNIST-derived images, noise, and data augmentation to improve generalization.

The final implementation includes:

- **2 × 2 average pooling** to reduce each 20 × 20 image to 10 × 10, followed by expansion back to 20 × 20.
- **Z-score normalization for training**, with statistics computed from the training split.
- **Min-Max normalization for the standalone inference program** (`P4.java`).
- An **80/20 train/test split** and mini-batch training (`batch size = 32`, `learning rate = 0.05`, up to `30 epochs`).
- Early stopping based on the MSE measured on the held-out test split.

**Implementation note:** Training and standalone inference use different normalization strategies. Also, the held-out test split is used for early stopping, so it is not an entirely untouched final evaluation set. These are limitations of the original academic implementation.

## Repository structure

```text
src/
├── P4.java                    # Loads saved parameters and classifies input images
├── W0.txt, W1.txt             # Trained weight matrices
├── B0.txt, B1.txt             # Trained bias vectors
├── apps/
│   └── TrainP4.java          # Model training and evaluation
├── math/
│   └── Matrix.java           # Matrix utility used by the project
└── neural/
    ├── MLP.java              # Feedforward neural network and backpropagation
    └── activation/           # Activation functions and interface
```

## Requirements

- Java JDK (a modern JDK supporting the source code)
- No external machine learning libraries

## Compile

From the repository root:

```bash
javac -d out $(find src -name '*.java')
```

## Run inference

The repository includes saved weights and biases for the trained classifier. From the repository root:

```bash
java -cp out P4 < input.csv
```

`input.csv` must contain one image per line, with **400 comma-separated numeric pixel values** representing a 20 × 20 image. The program prints `2` or `3` for each valid input row. The inference code expects the weight and bias files under `src/`.

The repository does **not** include a sample input image; provide an input file in the documented format.

## Train the model

The training application expects two files in the repository root:

- `dataset_last.csv`: one image per line, with 400 comma-separated values.
- `labels_last.csv`: one label (`2` or `3`) per line, matching the image order.

When these files are available, run:

```bash
java -cp out apps.TrainP4
```

The original dataset is **not included in this repository** because of its size. The training program writes learned parameters to the four `.txt` files in `src/`, overwriting the included parameters.

## Results

The accompanying academic report records a **best accuracy of 97.625%** after iterative adjustments to network architecture, preprocessing, and dataset composition. Individual training runs may yield different results.

## Academic context

Developed as a university Artificial Intelligence coursework project at the **University of Algarve**, in collaboration with classmates.

The `Matrix.java` utility includes an attribution to course-provided material; the neural network and training/inference implementation are presented here as part of the group coursework. This repository is for educational and portfolio purposes.
