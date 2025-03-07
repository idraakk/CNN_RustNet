# RustNet CNN – A Convolutional Neural Network Library in Rust

RustNet CNN is an initial design for a convolutional neural network (CNN) library written in Rust. The library is aimed at RF signal detection/classification tasks, but its design is modular enough to be extended to other applications. This README outlines the structure of the project, describes the purpose of each file, and provides a high-level design overview.

---

## Project Overview

The CNN is designed with the following key components:

- **Convolutional Layer:** Extracts local spatial features from 2D input (e.g., RF signal spectrograms or images).
- **Activation Functions:** Uses ReLU for intermediate activations and Sigmoid for the final layer to produce outputs in the range [0, 1] (for binary classification).
- **Pooling Layer:** Downsamples the feature maps to reduce dimensionality and retain salient features.
- **Dense (Fully-Connected) Layer:** Processes the flattened feature maps to produce final classification results.
- **Training Modules:** Implements forward propagation, backpropagation (with gradient clipping), loss calculation, and training loops.
- **Serialization:** Uses Serde to save and load models in JSON format.
- **Python Integration:** Exposes the library to Python via PyO3.

The design is modular to allow easy experimentation with each layer type and to support future extensions.

---

## Project Structure

```
rust_net_cnn/
├── Cargo.toml               # Project configuration and dependency declarations.
└── src/
    ├── activations.rs       # Contains activation functions (ReLU and Sigmoid) and their derivatives.
    ├── conv_layer.rs        # Implements the convolutional layer (filter initialization, forward pass).
    ├── pool_layer.rs        # Implements the max pooling layer for downsampling feature maps.
    ├── dense_layer.rs       # Implements the dense (fully-connected) layer.
    ├── loss.rs              # Defines the Mean Squared Error (MSE) loss function.
    ├── serde_arrays.rs      # Provides custom serialization/deserialization for ndarray's Array2.
    ├── cnn_train.rs         # Integrates the CNN layers into a complete model with forward and backward passes.
    └── lib.rs               # Library entry point with PyO3 bindings to expose the CNN to Python.
```

---

## File Descriptions & Code Skeletons

### Cargo.toml
This file specifies project metadata, dependencies (ndarray, rand, serde, PyO3, etc.), and tells Cargo to build a cdylib for Python interop.

```toml
[package]
name = "rust_net_cnn"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = { version = "0.15", features = ["serde"] }
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
csv = "1.1"
pyo3 = { version = "0.18", features = ["extension-module"] }

[lib]
crate-type = ["cdylib"]
```

### src/activations.rs
Defines ReLU and Sigmoid functions plus their derivatives.

```rust
pub fn relu(input: &Array2<f64>) -> Array2<f64> { /* Implementation */ }
pub fn relu_derivative(input: &Array2<f64>) -> Array2<f64> { /* Implementation */ }
pub fn sigmoid(input: &Array2<f64>) -> Array2<f64> { /* Implementation */ }
pub fn sigmoid_derivative(input: &Array2<f64>) -> Array2<f64> { /* Implementation */ }
```

### src/conv_layer.rs
Implements a convolutional layer that applies a filter over the input image.

```rust
pub struct ConvLayer { /* Fields */ }

impl ConvLayer {
    pub fn new(filter_size: usize, stride: usize) -> Self { /* Initialization */ }
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> { /* Forward Pass */ }
    pub fn backward(&mut self, d_out: &Array2<f64>, learning_rate: f64) { /* Backpropagation */ }
}
```

### src/pool_layer.rs
Implements max pooling to downsample feature maps.

```rust
pub struct PoolLayer { /* Fields */ }

impl PoolLayer {
    pub fn new(pool_size: usize, stride: usize) -> Self { /* Initialization */ }
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> { /* Pooling Operation */ }
    pub fn backward(&self, d_out: &Array2<f64>, input_shape: (usize, usize)) -> Array2<f64> { /* Backpropagation */ }
}
```

### src/dense_layer.rs
Implements a fully-connected layer.

```rust
pub struct DenseLayer { /* Fields */ }

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self { /* Initialization */ }
    pub fn forward(&mut self, input: &Array2<f64>) -> Array2<f64> { /* Forward Pass */ }
    pub fn backward(&mut self, d_out: &Array2<f64>, learning_rate: f64) -> Array2<f64> { /* Backpropagation */ }
}
```

### src/loss.rs
Defines the MSE loss function.

```rust
pub fn mean_squared_error(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 { /* Calculation */ }
```

### src/cnn_train.rs
Integrates the CNN components into a full model with forward and backward passes.

```rust
pub struct CNNNetwork { /* Fields */ }

impl CNNNetwork {
    pub fn new(/* Params */) -> Self { /* Initialization */ }
    pub fn forward(&mut self, input: Array2<f64>) -> Array2<f64> { /* Forward Pass */ }
    pub fn backward(&mut self, input: &Array2<f64>, targets: &Array2<f64>, learning_rate: f64) { /* Backpropagation */ }
    pub fn train(&mut self, input: &Array2<f64>, target: &Array2<f64>, learning_rate: f64, epochs: usize) { /* Training Loop */ }
    pub fn save(&self, path: &str) { /* Save Model */ }
    pub fn load(path: &str) -> Self { /* Load Model */ }
}
```

### src/lib.rs
Exposes the CNN as a Python module using PyO3.

```rust
#[pyclass]
pub struct PyCNN { /* Fields */ }

#[pymethods]
impl PyCNN {
    #[new]
    pub fn new(/* Params */) -> Self { /* Initialization */ }
    pub fn train(&mut self, input: Vec<Vec<f64>>, target: Vec<Vec<f64>>, learning_rate: f64, epochs: usize) { /* Train */ }
    pub fn predict(&mut self, input: Vec<Vec<f64>>) -> Vec<Vec<f64>> { /* Predict */ }
    pub fn save(&self, path: &str) { /* Save Model */ }
    #[staticmethod]
    pub fn load(path: &str) -> Self { /* Load Model */ }
}

#[pymodule]
fn rust_net_cnn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCNN>()?;
    Ok(())
}
```

---

## Design Description

### Overall Architecture

The CNN is designed as an extension of the initial ANN project in Rust with the following modifications:
- **Convolutional Layer:**  
  Processes 2D input data using a filter (kernel) to extract local features.
- **Pooling Layer:**  
  Downsamples the convolved feature maps to reduce dimensionality and emphasize dominant features.
- **Dense Layer:**  
  Maps the flattened pooled output to a final output. Sigmoid activation is applied in the final layer for binary classification.
- **Python Exposure:**  
  The CNN is exposed as a Python module using PyO3.

This README provides a structured and clear overview of the project.

---

