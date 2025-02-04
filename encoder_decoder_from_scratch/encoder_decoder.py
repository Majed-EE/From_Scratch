import torch
import torch.nn as nn

## step 1 make embedddings

class InputEmbeddings(nn.Module):
    def __init__(self, embd_dim, vocab_size):
        super().__init__()
        self.embd_dim = embd_dim # embedding size is d_model in the paper
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embd_dim)
        self.embedding.weight.data.uniform_(-1, 1)


    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.embd_dim) # bcause it is done in the paper









## step 2 add position embeddings