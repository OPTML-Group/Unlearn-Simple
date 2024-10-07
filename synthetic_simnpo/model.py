import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
import math
import pdb

def get_activation(activation="relu"):
    if activation == "relu":
        return F.relu
    elif activation == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    else:
        raise NotImplementedError

class DecoderTransformerBackbone(nn.Module):
    def __init__(self, config, activation="relu",
                 normalize_attn=True, mlp=True, layernorm=True, positional_embedding=True):
        super(DecoderTransformerBackbone, self).__init__()
        self.n_positions = config.n_positions
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.activation = get_activation(activation)
        self.normalize_attn = normalize_attn
        self.layernorm = layernorm
        self.mlp = mlp
        self.positional_embedding = positional_embedding

        # Positional embeddings
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.wpe.weight.data.normal_(mean=0.0, std=config.initializer_range)

        # Layers
        self._queries = nn.ModuleList()
        self._keys = nn.ModuleList()
        self._values = nn.ModuleList()
        self._mlps = nn.ModuleList()
        self._lns_1 = nn.ModuleList()
        self._lns_2 = nn.ModuleList()
        for i in range(self.n_layer):
            self._queries.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._keys.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._values.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            self._lns_1.append(nn.LayerNorm(self.n_embd))
            self._mlps.append(
                nn.Sequential(
                    nn.Linear(self.n_embd, self.n_embd),
                    nn.ReLU(),
                    nn.Linear(self.n_embd, self.n_embd),
                )
            )
            self._lns_2.append(nn.LayerNorm(self.n_embd))

        # Pre-compute decoder attention mask
        with torch.no_grad():
            self.mask = torch.zeros(1, self.n_positions, self.n_positions)
            for i in range(self.n_positions):
                if self.normalize_attn:
                    self.mask[0, i, :(i+1)].fill_(1./(i+1))
                else:
                    self.mask[0, i, :(i+1)].fill_(1.)



    def forward(self, inputs_embeds=None, position_ids=None, return_hidden_states=False):
        assert inputs_embeds is not None
        hidden_states = []
        N = inputs_embeds.shape[1]
        H = inputs_embeds
        if self.positional_embedding:
            if position_ids is None:
                position_ids = torch.arange(N, dtype=torch.long, device=H.device)
                position_ids = position_ids.unsqueeze(0).expand_as(inputs_embeds[:, :, 0])
            position_embeds = self.wpe(position_ids)
            H = H + position_embeds
        hidden_states.append(H)
        for (q, k, v, ln1, mlp, ln2) in zip(
                self._queries, self._keys, self._values, self._lns_1, self._mlps, self._lns_2,
        ):
            query = q(H)
            key = k(H)
            value = v(H)

            # Compute attention scores
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.n_embd)

            # Create causal mask
            causal_mask = torch.tril(torch.ones(N, N, device=H.device)).unsqueeze(0)
            causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))

            # Apply mask before softmax
            attn_scores = attn_scores + causal_mask

            # Compute attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Compute attention output
            attn_output = torch.matmul(attn_weights, value)
            H = H + attn_output

            if self.layernorm:
                H = ln1(H)
            if self.mlp:
                H = H + mlp(H)
                if self.layernorm:
                    H = ln2(H)
            hidden_states.append(H)
        if return_hidden_states:
            return H, hidden_states
        return H


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.state_size = config['state_size']
        self.n_positions = config['n_positions']
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.dropout = config['dropout']
        self.activation = config['activation']

        # Embedding layer
        self.embedding = nn.Embedding(self.state_size, self.n_embd)

        # Transformer backbone
        gpt2_config = GPT2Config(
            n_positions=self.n_positions,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )

        self.transformer = DecoderTransformerBackbone(
            gpt2_config,
            activation=self.activation,
            positional_embedding=True,
            normalize_attn=False,
        )

        # Output projection
        self.output_layer = nn.Linear(self.n_embd, self.state_size)

    def forward(self, input_ids,log_target = False):
        # input_ids: [batch_size, seq_length]
        # Get input embeddings
        inputs_embeds = self.embedding(input_ids)  # [batch_size, seq_length, n_embd]
        # Get position_ids
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
       
        # Pass through transformer
        hidden_states = self.transformer(inputs_embeds=inputs_embeds, position_ids=position_ids)
        # Project to output
        logits = self.output_layer(hidden_states)  # [batch_size, seq_length, state_size]

        if log_target:
            return F.log_softmax(logits, dim=-1)
        
        return logits
