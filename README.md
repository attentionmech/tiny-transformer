# tiny-transformer

Implementation of a next character predicting language model using multi-head attention transformer. The goal of this is to allow observation on what is the network's capability while training (inspired from nanoGPT train.py).

![demo](assets/demo.gif)


## Requirements

```zsh
pip install torch einops numpy
```

## Usage

```zsh
 python tiny_transformer.py --epochs 100 --dataset assets/tinystories.txt --num_rows 100
```

## Dataset

A sample dataset is present in assets/ path. Download tinystories full dataset from [here](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) to play around.

## References

- [nanogpt](https://github.com/karpathy/nanoGPT)
- [pytorch_transformer](https://github.com/hkproj/pytorch-transformer)
- [simple_transformer](https://github.com/xjdr-alt/simple_transformer/blob/main/simple_transformer.py)

## Credits

- Vscode Copilot
- Claude
- ChatGPT
