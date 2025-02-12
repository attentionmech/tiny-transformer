import os
import argparse
import torch

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def train_test_split(data, test_ratio=0.1):
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


def read_data(file_path, num_rows=None):
    with open(file_path, "r") as file:
        data = file.read()
    if num_rows is not None:
        lines = data.splitlines()[:num_rows]
        data = "\n".join(lines)
    return data


def preprocess_data(data):
    chars = sorted(set(data))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    data_indices = [char_to_idx[char] for char in data]
    return chars, char_to_idx, idx_to_char, data_indices


class CharDataset(Dataset):
    def __init__(self, data, seq_length=16):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx : idx + self.seq_length]
        target_seq = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)


class TTReLU(torch.nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))


class TTLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        x = x @ self.weight.T
        if self.bias is not None:
            x += self.bias
        return x


class TTLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TTEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        return self.weight[x]


class TTDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return mask * x / (1 - self.p)


class TTMultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = TTLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = TTLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = TTLinear(embed_dim, embed_dim, bias=False)
        self.out_proj = TTLinear(embed_dim, embed_dim, bias=False)
        self.dropout = TTDropout(dropout)

    def forward(self, query, key, value):
        batch_size, seq_len, embed_dim = query.shape
        q = (
            self.q_proj(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = (
            (attn_weights @ v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.out_proj(self.dropout(attn_output))


class TTFeedForward(torch.nn.Module):
    def __init__(self, embed_dim, ff_hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = TTLinear(embed_dim, ff_hidden_dim)
        self.relu = TTReLU()
        self.dropout1 = TTDropout(dropout)
        self.fc2 = TTLinear(ff_hidden_dim, embed_dim)
        self.dropout2 = TTDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class CharTransformerModel(torch.nn.Module):
    def __init__(
        self, embed_dim, num_heads, ff_hidden_dim, vocab_size, seq_len, dropout=0.2
    ):
        super().__init__()
        self.token_embedding = TTEmbedding(vocab_size, embed_dim)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.attention = TTMultiheadAttention(embed_dim, num_heads, dropout)
        self.ff = TTFeedForward(embed_dim, ff_hidden_dim, dropout)
        self.layer_norm1 = TTLayerNorm(embed_dim)
        self.layer_norm2 = TTLayerNorm(embed_dim)
        self.output_layer = TTLinear(embed_dim, vocab_size)
        self.dropout = TTDropout(dropout)

    def forward(self, x):
        x = self.token_embedding(x) + self.pos_embedding
        x = self.dropout(x)

        attn_output = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        ff_output = self.ff(x)
        x = self.layer_norm2(x + ff_output)

        return self.output_layer(x)


def train(
    model,
    epochs,
    device,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    vocab_size,
    idx_to_char,
    char_to_idx,
    seq_len,
):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    seed_text = "Once upon a time "
                    input_ids = [char_to_idx[c] for c in seed_text]
                    if len(input_ids) < seq_len:
                        input_ids = [0] * (seq_len - len(input_ids)) + input_ids

                    temperature = 0.6
                    for _ in range(100):
                        x_infer = (
                            torch.tensor(input_ids[-seq_len:], dtype=torch.long)
                            .unsqueeze(0)
                            .to(device)
                        )
                        logits = model(x_infer)
                        probs = (logits[0, -1] / temperature).softmax(dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                        input_ids.append(next_token)

                    generated_text = "".join(idx_to_char[i] for i in input_ids)
                    print(
                        f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}, Generated Text: {generated_text.strip()}\n"
                    )

                    model.train()

        print(f"Epoch {epoch+1}, Avg Train Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, criterion, val_loader, device, vocab_size)


def evaluate(model, criterion, val_loader, device, vocab_size):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
        print(f"Validation Loss: {total_loss / len(val_loader):.4f}\n")


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    num_epochs,
    vocab_size,
    idx_to_char,
    char_to_idx,
    seq_len,
    device="mps",
):
    train(
        model,
        num_epochs,
        device,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
        vocab_size,
        idx_to_char,
        char_to_idx,
        seq_len,
    )


def main():

    parser = argparse.ArgumentParser(description="tiny-transformer")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs (default 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default 64)",
    )
    parser.add_argument(
        "--seq_length", type=int, default=64, help="Sequence length (default 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate (default 1e-3)"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=128, help="Embedding size (default 300)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads (default 6)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of transformer layers (default 6)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden size of transformer layers (default 64)",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help="Number of rows from dataset to use (default 100)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        help="Optimizer type (default adam)",
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to use for training",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "mps"],
        help="Device to run the model on (default: cpu)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling during inference",
    )

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(
        args.device
        if torch.backends.mps.is_available() or args.device == "cpu"
        else "cpu"
    )

    print("\n\n------------------")
    print(f"Configuration:\n")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Embedding Size: {args.embedding_size}")
    print(f"Number of Heads: {args.num_heads}")
    print(f"Number of Layers(Blocks): {args.num_layers}")
    print(f"MLP Layer Size: {args.hidden_size}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {device}")

    if args.dataset and os.path.exists(args.dataset):
        data = read_data(args.dataset, num_rows=args.num_rows)
    else:
        tiny_stories_download_link = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        print("Download the tinystories dataset from:", tiny_stories_download_link)
        print(f"Dataset {args.dataset} not found. Exiting...")
        return

    chars, char_to_idx, idx_to_char, data_indices = preprocess_data(data)
    vocab_size = len(chars)

    print(f"Vocabulary Size: {vocab_size}")
    print("\n\n")

    train_data, test_data = train_test_split(data_indices, test_ratio=0.1)

    train_dataset = CharDataset(train_data, seq_length=args.seq_length)
    test_dataset = CharDataset(test_data, seq_length=args.seq_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = CharTransformerModel(
        embed_dim=args.embedding_size,
        num_heads=args.num_heads,
        ff_hidden_dim=args.hidden_size,
        vocab_size=vocab_size,
        seq_len=args.seq_length,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    train_model(
        model,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        args.epochs,
        vocab_size,
        idx_to_char,
        char_to_idx,
        args.seq_length,
        device=device,
    )


if __name__ == "__main__":
    main()
