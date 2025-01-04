# tiny-transformer

Implementation of a next character predicting language model using transformer architecture.

## Requirements

```zsh
pip install torch einops
```

## Usage

```zsh
python tiny_transformer.py --epochs 10 --batch_size 16 --seq_length 32 --learning_rate 1e-4 --embedding_size 64 --num_heads 4 --num_layers 2 --hidden_size 128 --dataset tinystories.txt --inference_interval 100 --inference_length 200 --optimizer adam
```

## Dataset

Download tinystories dataset from [here](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) to play around.

## References

- @karpthy's nanogpt
- @hkproj's pytorch_transformer
