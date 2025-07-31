import math
import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, n_heads=4, n_layers=2, output_dim=1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        """
        x = self.embedding(input_ids)                # [batch, seq_len, embed_dim]
        x = self.pos_encoder(x)                      # Add positional encoding
        x = x.permute(1, 0, 2)                       # [seq_len, batch, embed_dim]

        if attention_mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        else:
            x = self.transformer_encoder(x)

        x = x.mean(dim=0)  # [batch_size, embed_dim]
        return self.output_head(x)  # [batch_size, output_dim]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        returns: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :] # type: ignore
        return x


def train_model(model, train_loader, num_epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)         # [batch_size, seq_len]
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            targets = batch['targets'].to(device)             # [batch_size, output_dim]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Example usage (outside this file):
# from torch.utils.data import DataLoader
# train_loader = DataLoader(your_dataset, batch_size=32, shuffle=True)
# model = TransformerRegressor(vocab_size=..., output_dim=...)
# train_model(model, train_loader, num_epochs=10, lr=1e-3, device='cuda')

meta_features = [
    "num_features", "num_classes", "num_samples", "num_numerical_features",
    "num_categorical_features", "num_text_features", "num_time_series_features",
    "has_missing_values", "has_outliers", "has_text_data", "has_time_series_data"
]
num_algorithms = 6  # Number of algorithms in the meta-learning setup
model = TransformerRegressor(vocab_size=len(meta_features), output_dim=num_algorithms)
print(model)
print(len(meta_features))  # Should match the vocab_size
print(model.embedding.weight.shape)