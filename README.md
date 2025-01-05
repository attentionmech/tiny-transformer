# tiny-transformer

Implementation of a next character predicting language model using multi-head attention transformer. The goal of this is to allow observation on what is the network's capability while training (inspired from nanoGPT train.py).

![demo](assets/demo.gif)


## Requirements

```zsh
pip install torch einops numpy
```

## Usage

```zsh
python tiny_transformer.py --epochs 10 --batch_size 16 --seq_length 32 --learning_rate 1e-4 --embedding_size 64 --num_heads 4 --num_layers 2 --hidden_size 128 --dataset assets/tinystories.txt --inference_interval 100 --inference_length 200 --optimizer adam
```

## Dataset

A sample dataset is present in assets/ path. Download tinystories full dataset from [here](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) to play around.

## References

- [nanogpt](https://github.com/karpathy/nanoGPT)
- [pytorch_transformer](https://github.com/hkproj/pytorch-transformer)
- [simple_transformer](https://github.com/xjdr-alt/simple_transformer/blob/main/simple_transformer.py)

## Credits

- Claude
- ChatGPT
- CoPilot
