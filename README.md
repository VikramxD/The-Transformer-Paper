# Transformer Model Implementation

This repository contains a from-scratch implementation of the Transformer model, as described in the paper "Attention is All You Need" by Vaswani et al. The Transformer model is a type of attention-based model that has been widely used in various Natural Language Processing (NLP) tasks.

## Features

- Customizable model parameters, including the number of layers, the number of attention heads, and the dimension of the model.
- Includes both the encoder and decoder components of the Transformer model.
- Uses PyTorch for efficient computation and gradient calculations.
- Includes Positional Encoding, Multi-Head Attention, and Feed Forward Network modules.

## Code Example

```python
from transformer import Transformer
import torch.nn as nn

# Initialize the Transformer model
model = Transformer(n_layers=6, n_heads=8, d_model=512)

# Forward pass
output = model.forward(input_tensor)
```

## Installation

To use this module, you need to have Python and PyTorch installed. You can install PyTorch by following the instructions on the [official website](https://pytorch.org/).

## Dataset

This implementation uses the [Multi30K](https://github.com/multi30k/dataset) dataset for training and evaluation. The Multi30K dataset is a collection of 30,000 images and their corresponding English and German descriptions.

## Training

To train the model, run the `train.py` script. You can customize the training parameters by modifying the `config.py` file.

## Evaluation

To evaluate the model, run the `evaluate.py` script. This script will load the trained model and evaluate it on the test set.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
