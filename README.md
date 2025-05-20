# Transformer Model Implementation (PyTorch)

This repository contains a from-scratch implementation of the Transformer model, as described in the seminal paper "Attention is All You Need" by Vaswani et al. The model is built using PyTorch and leverages PyTorch Lightning for streamlined training (though the core model is independent and can be used separately).

This implementation aims to provide a clear, modular, and understandable version of the Transformer architecture, suitable for educational purposes and as a foundation for further research and development.

## Features

*   **Complete Architecture:** Implements both the Encoder and Decoder stacks of the Transformer.
*   **Core Components:** Includes implementations for:
    *   Multi-Head Self-Attention
    *   Cross-Attention
    *   Position-wise Feed-Forward Networks
    *   Positional Encoding
    *   Input Embeddings
    *   Layer Normalization
    *   Skip Connections
*   **Customizable Model:** Easily configure model parameters such as:
    *   Number of Encoder/Decoder Layers
    *   Number of Attention Heads
    *   Embedding Dimension
    *   Feed-Forward Network Dimension
    *   Dropout Rate
*   **Modular Design:** Each component (e.g., `EncoderBlock`, `MultiHeadAttention`, `FeedForwardBlock`) is implemented as a `torch.nn.Module`, making them easy to understand, modify, and integrate into other PyTorch projects.
*   **Dependency Management:** Uses a `requirements.txt` file for easy setup of the Python environment.
*   **Clear Entry Point:** Provides a `build_model` function in `transformer.py` to conveniently construct a complete Transformer model.

## Structure and Components

The model is organized into several key modules, each implemented in its own Python file:

*   **`transformer.py`**: Defines the overall `Transformer` class, bringing together the encoder and decoder. It also includes the `build_model` function for easy model instantiation.
*   **`encoder.py`**: Contains the `Encoder` and `EncoderBlock` classes. The Encoder is responsible for processing the input sequence.
*   **`decoder.py`**: Contains the `Decoder` and `DecoderBlock` classes. The Decoder generates the output sequence based on the encoder's output and its own previous outputs.
*   **`multiheadattention.py`**: Implements the `MultiheadAttention` mechanism, a core component of both encoder and decoder blocks.
*   **`feedforward.py`**: Implements the `FeedForwardBlock`, another key component within each encoder and decoder block.
*   **`embedding.py`**: Contains `InputEmbedding` for creating word embeddings and `PositionalEncoding` for injecting sequence order information.
*   **`layer_normalization.py`**: Implements `LayerNormalization`.
*   **`skip_connection.py`**: Implements `SkipConnection` (residual connections).

This modular structure makes the codebase easier to navigate and understand. Each component can be studied and modified independently.

## Model Parameters

The `build_model` function in `transformer.py` allows you to easily create a Transformer model with specific dimensions and configurations. Here are the key parameters:

*   `src_vocab_size (int)`: The size of the source vocabulary (e.g., number of unique words in the input language).
*   `tgt_vocab_size (int)`: The size of the target vocabulary (e.g., number of unique words in the output language).
*   `src_seq_len (int)`: The maximum sequence length for source inputs. Used for positional encoding.
*   `tgt_sequence_len (int)`: The maximum sequence length for target inputs. Used for positional encoding.
*   `embedding_dimension (int, optional)`: The dimensionality of the input and output embeddings. Defaults to `512`.
*   `n_layer (int, optional)`: The number of `EncoderBlock`s in the Encoder and `DecoderBlock`s in the Decoder. Defaults to `6`.
*   `num_heads (int, optional)`: The number of attention heads in the `MultiheadAttention` layers. Defaults to `8`. The `embedding_dimension` must be divisible by `num_heads`.
*   `dropout (float, optional)`: The dropout rate used in various parts of the model (e.g., embeddings, attention, feed-forward blocks). Defaults to `0.1`.
*   `feed_forward_dim (int, optional)`: The dimensionality of the inner layer in the `FeedForwardBlock`. Defaults to `2048`.

These parameters provide flexibility in tailoring the model architecture to specific tasks and datasets.

## Getting Started

Follow these steps to get the model up and running:

### 1. Prerequisites

*   Python 3.x
*   PyTorch

### 2. Installation

Clone the repository:
\`\`\`bash
git clone <repository_url> # Replace <repository_url> with the actual URL
cd <repository_directory>
\`\`\`

Install the required dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`
This will install PyTorch, PyTorch Lightning, and other necessary packages.

### 3. Basic Usage

You can build a Transformer model using the `build_model` function from `transformer.py`. Here's a basic example:

\`\`\`python
from transformer import build_model
import torch

# Define model parameters
src_vocab_size = 5000
tgt_vocab_size = 5000
src_seq_len = 100
tgt_seq_len = 120
embedding_dim = 512
num_layers = 6
num_heads = 8

# Build the model
model = build_model(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_seq_len=src_seq_len,
    tgt_sequence_len=tgt_seq_len,
    embedding_dimension=embedding_dim,
    n_layer=num_layers,
    num_heads=num_heads
)

print("Transformer model built successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Example dummy input (replace with actual tokenized data for training/inference)
# Batch size = 1, sequence length = 10
dummy_src = torch.randint(0, src_vocab_size, (1, 10))
dummy_tgt = torch.randint(0, tgt_vocab_size, (1, 12))

# Create dummy masks (no padding, look-ahead for target)
dummy_src_mask = None # Or appropriate mask
dummy_tgt_mask = None # Or appropriate mask, typically a causal mask for self-attention

# Forward pass (example)
# Note: For a real use case, you'd embed and add positional encoding first,
# or use the model's encode/decode methods which handle this.
# The build_model function initializes these components.

# Using the encode and decode methods:
encoded_src = model.encode(dummy_src, dummy_src_mask)
decoded_output = model.decode(encoded_src, dummy_src_mask, dummy_tgt, dummy_tgt_mask)
logits = model.linear_step(decoded_output) # Project to vocabulary

print(f"Logits shape: {logits.shape}") # Expected: (batch_size, tgt_seq_len, tgt_vocab_size)
\`\`\`

This snippet demonstrates how to instantiate the model with your desired parameters. For actual training or inference, you would feed tokenized and properly masked sequences to the model.

### 4. Running the Example Script

The repository includes an `example.py` script that demonstrates how to:
1.  Instantiate the Transformer model using `build_model`.
2.  Perform a dummy forward pass with sample tensor inputs.
3.  Print the shapes of the intermediate and final outputs.

To run the example:
\`\`\`bash
python example.py
\`\`\`
This will output information about the model construction and the tensor shapes at each step of the forward pass, giving you a basic understanding of the data flow.

## Testing

Currently, this project does not include a dedicated suite of automated tests. While the individual components have been developed with care, comprehensive unit and integration tests are crucial for ensuring robustness and facilitating future development.

Contributions in this area are highly welcome! If you'd like to help by adding tests, please feel free to fork the repository and submit a pull request.

## Contributing

Contributions are welcome! If you have improvements, bug fixes, or new features you'd like to add, please follow these steps:

1.  **Fork the repository** on GitHub.
2.  **Create a new branch** for your feature or fix:
    \`\`\`bash
    git checkout -b feature/your-feature-name
    \`\`\`
    or
    \`\`\`bash
    git checkout -b fix/your-bug-fix
    \`\`\`
3.  **Make your changes** and commit them with clear, descriptive messages.
4.  **Push your branch** to your forked repository:
    \`\`\`bash
    git push origin feature/your-feature-name
    \`\`\`
5.  **Open a Pull Request (PR)** against the main branch of this repository.
    *   Provide a clear title and description for your PR.
    *   Explain the changes you've made and why.
    *   If applicable, reference any related issues.

We appreciate your help in making this project better!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
