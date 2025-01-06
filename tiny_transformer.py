import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from einops import rearrange


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


def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones((seq_len, seq_len)))
    mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, seq_len, seq_len]
    return mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_dim = emb_size // num_heads

        assert (
            emb_size % num_heads == 0
        ), f"embedding size {emb_size} must be divisible by num_heads {num_heads}"

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

        self.fc_out = nn.Linear(emb_size, emb_size)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask=None):

        values = rearrange(
            values,
            "batch_size value_len (num_heads head_dim) -> num_heads batch_size value_len head_dim",
            num_heads=self.num_heads,
        )
        keys = rearrange(
            keys,
            "batch_size key_len (num_heads head_dim) -> num_heads batch_size key_len head_dim",
            num_heads=self.num_heads,
        )
        query = rearrange(
            query,
            "batch_size query_len (num_heads head_dim) -> num_heads batch_size query_len head_dim",
            num_heads=self.num_heads,
        )

        score = torch.matmul(
            query,
            rearrange(
                keys,
                "batch_size num_heads key_len head_dim -> batch_size num_heads head_dim key_len",
            ),
        )

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(score / (self.head_dim ** (1 / 2)), dim=-1)

        attention = self.attn_dropout(attention)

        out = torch.matmul(attention, values)

        out = rearrange(
            out,
            "num_heads batch_size query_len head_dim -> batch_size query_len (num_heads head_dim)",
        )

        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_size, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(emb_size, num_heads, dropout)
        self.feed_forward = FeedForward(emb_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.layer_norm2 = nn.LayerNorm(emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_out = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attention_out))

        ff_out = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_out))

        return x


class CharTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size=32,
        num_heads=2,
        num_layers=1,
        hidden_size=64,
        dropout=0.1,
        device="cpu",
    ):
        super(CharTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size).to(device)
        self.positional_encoding = nn.Embedding(1000, emb_size).to(device)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(emb_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = (
            torch.arange(0, seq_len, device=x.device)
            .unsqueeze(1)
            .expand(seq_len, batch_size)
            .T
        )
        embedded = self.embedding(x) + self.positional_encoding(positions)

        mask = create_causal_mask(seq_len).to(x.device)
        transformer_output = embedded

        for block in self.transformer_blocks:
            transformer_output = block(transformer_output, mask)

        output = self.fc_out(transformer_output)
        return output

    def generate(
        self, start_text, char_to_idx, idx_to_char, max_length=100, temperature=0.3
    ):
        input_seq = [char_to_idx.get(char, char_to_idx[" "]) for char in start_text]
        input_seq = (
            torch.tensor(input_seq).unsqueeze(1).to(next(self.parameters()).device)
        )
        generated_text = start_text
        for _ in range(max_length):
            output = self(input_seq)
            last_char_logits = output[-1, 0, :]

            last_char_logits = last_char_logits / temperature

            probs = torch.softmax(last_char_logits, dim=-1)

            predicted_idx = torch.multinomial(probs, 1).item()
            predicted_char = idx_to_char[predicted_idx]
            generated_text += predicted_char
            input_seq = torch.cat(
                [input_seq, torch.tensor([[predicted_idx]]).to(input_seq.device)], dim=0
            )
        return generated_text


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    num_epochs,
    vocab_size,
    args,
    char_to_idx,
    idx_to_char,
    inference_text=None,
    device="cpu",
):
    for epoch in range(num_epochs):
        total_steps = len(train_dataloader)

        total_train_loss = 0

        for step, (input_seq, target_seq) in enumerate(train_dataloader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if step % args.inference_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{total_steps}], Train Loss: {loss.item():.4f}"
                )
                generated_text = model.generate(
                    inference_text or "",
                    char_to_idx,
                    idx_to_char,
                    max_length=args.inference_length,
                    temperature=args.temperature,
                )
                print(f"\n{generated_text}\n")

        if len(train_dataloader):
            avg_train_loss = total_train_loss / len(train_dataloader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Avg. Train Loss: {avg_train_loss:.4f}\n"
            )

        total_steps = len(test_dataloader)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for step, (input_seq, target_seq) in enumerate(test_dataloader):
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                output = model(input_seq)
                test_loss = criterion(
                    output.reshape(-1, vocab_size), target_seq.reshape(-1)
                )
                total_test_loss += test_loss.item()

                if step % args.inference_interval == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{total_steps}], Test Loss: {loss.item():.4f}"
                    )
                    generated_text = model.generate(
                        inference_text or "",
                        char_to_idx,
                        idx_to_char,
                        max_length=args.inference_length,
                        temperature=args.temperature,
                    )
                    print(f"\n{generated_text}\n")

        if len(test_dataloader):
            avg_test_loss = total_test_loss / len(test_dataloader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Avg. Train Loss: {avg_train_loss:.4f}  Avg. Test Loss: {avg_test_loss:.4f}\n"
            )

        torch.save(model.state_dict(), "model.pth")


def main():

    parser = argparse.ArgumentParser(
        description="tiny-transformer: Character-level Transformer Language Model Training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default 100)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default 64)",
    )
    parser.add_argument(
        "--seq_length", type=int, default=16, help="Sequence length (default 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate (default 1e-3)"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=300, help="Embedding size (default 300)"
    )
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of attention heads (default 6)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of transformer layers (default 6)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of transformer layers (default 64)",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help="Number of rows from dataset to use (default 100)",
    )
    parser.add_argument(
        "--inference_interval",
        type=int,
        default=500,
        help="Number of iterations between inference",
    )
    parser.add_argument(
        "--inference_length",
        type=int,
        default=100,
        help="Number of characters to generate during inference",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        help="Optimizer type (default adam)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate (default 0.2)"
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
        "--inference_text",
        type=str,
        default="Once upon a time",
        help="Text to start inference from (default: Once upon a time)",
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

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay only valid for AdamW right now (defaut: 0.01)",
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
    print(f"Inference Interval: {args.inference_interval}")
    print(f"Inference Length: {args.inference_length}")
    print(f"Inference Text: {args.inference_text}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Device: {device}")
    print("------------------\n\n")

    if args.dataset and os.path.exists(args.dataset):
        print("Reading dataset...")
        data = read_data(args.dataset, num_rows=args.num_rows)
        print("Dataset loaded.")
    else:
        tiny_stories_download_link = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        print("Download the tinystories dataset from:", tiny_stories_download_link)
        print(f"Dataset {args.dataset} not found. Exiting...")
        return

    chars, char_to_idx, idx_to_char, data_indices = preprocess_data(data)
    vocab_size = len(chars)

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
        vocab_size,
        emb_size=args.embedding_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        device=device,
    ).to(device)

    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
        args,
        char_to_idx,
        idx_to_char,
        inference_text=args.inference_text,
        device=device,
    )


if __name__ == "__main__":
    main()
