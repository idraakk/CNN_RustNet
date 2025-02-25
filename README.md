# 🧠 CNN Library in Rust

This project is a **Convolutional Neural Network (CNN) library** implemented from scratch in **Rust**. It provides a simple yet scalable architecture for building and training CNNs for image classification tasks.

---

## 📂 Project Structure

```
my_cnn_project/
├── Cargo.toml       # Rust project configuration and dependencies
├── src/
│   ├── main.rs      # Entry point: Initializes and trains the CNN model
│   ├── lib.rs       # Exports all internal modules
│   ├── tensor.rs    # Tensor struct for handling multi-dimensional data
│   ├── layer.rs     # Defines the Layer trait for all layers
│   ├── loss.rs      # Implements Softmax Cross-Entropy loss
│   ├── model.rs     # Composes the CNN model using different layers
│   ├── utils.rs     # Helper functions (e.g., flattening tensors)
│   ├── layers/      # Directory containing different layer implementations
│   │   ├── mod.rs       # Re-exports all layer modules
│   │   ├── conv2d.rs    # Convolutional (Conv2D) layer
│   │   ├── linear.rs    # Fully connected (Linear) layer
│   │   ├── relu.rs      # ReLU activation layer
│   │   ├── maxpool.rs   # Max pooling layer
```

---

## 📜 Explanation of Each File

### **1️⃣ Cargo.toml**
This file defines the project and dependencies.

```toml
[package]
name = "my_cnn_project"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8"
```

---

### **2️⃣ main.rs (Entry Point)**
The **entry point** of the project where the model is initialized and trained.

```rust
use my_cnn_project::model::CNN;
use my_cnn_project::loss::SoftmaxCrossEntropy;
use my_cnn_project::tensor::Tensor;

fn main() {
    let mut model = CNN::new();
    let loss_fn = SoftmaxCrossEntropy {};

    let input = Tensor::new(vec![1, 1, 28, 28]);  // Dummy input
    let label = Tensor::new(vec![1, 10]);         // One-hot encoded label

    for epoch in 0..10 {
        let logits = model.forward(&input);
        let (predictions, loss) = loss_fn.forward(&logits, &label);
        println!("Epoch {}: Loss = {}", epoch, loss);

        let grad_loss = loss_fn.backward(&logits, &label);
        model.backward(&input, &grad_loss, 0.01);
    }
}
```

---

### **3️⃣ lib.rs (Library Module)**
Exports all the modules in the library.

```rust
pub mod tensor;
pub mod layer;
pub mod layers;
pub mod loss;
pub mod model;
pub mod utils;
```

---

### **4️⃣ tensor.rs (Tensor Data Structure)**
Defines a basic **Tensor struct** for handling multi-dimensional arrays.

```rust
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }
    
    pub fn new_random(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let size = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data = (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Tensor { data, shape }
    }
}
```

---

### **5️⃣ layer.rs (Layer Trait)**
Defines the **Layer** trait that all layers must implement.

```rust
use crate::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor;
    fn update_parameters(&mut self, learning_rate: f32);
}
```

---

### **6️⃣ layers/mod.rs**
Module file that re-exports all layer implementations.

```rust
pub mod conv2d;
pub mod linear;
pub mod relu;
pub mod maxpool;
```

---

### **7️⃣ layers/conv2d.rs (Convolutional Layer)**
Defines a **Conv2D** layer.

```rust
use crate::tensor::Tensor;
use crate::layer::Layer;

pub struct Conv2D {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let weights = Tensor::new_random(vec![out_channels, in_channels, kernel_size, kernel_size]);
        let bias = Tensor::new(vec![out_channels]);
        Conv2D { weights, bias }
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        Tensor::new(input.shape.clone()) // Placeholder
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        Tensor::new(input.shape.clone()) // Placeholder
    }

    fn update_parameters(&mut self, _learning_rate: f32) {}
}
```

---

## 🚀 Future Work
- Implement proper convolution and pooling functions.
- Add support for backpropagation and optimizer methods.
- Expand with additional activation functions (Sigmoid, Tanh, etc.).
- Integrate with real datasets for training.

---

