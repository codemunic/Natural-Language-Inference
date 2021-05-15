import torch
import torch.nn as nn
from pdb import set_trace

__all__ = ['bilstm']

class BiLSTM(nn.Module):
    def __init__(self, options):
        super(BiLSTM, self).__init__()
        self.embed_dim = 300
        self.hidden_size = options['d_hidden']
        self.directions = 2
        self.num_layers = 2
        self.concat = 4
        self.device = options['device']
        self.embedding = nn.Embedding.from_pretrained(torch.load('.vector_cache/{}_vectors.pt'.format(options['dataset'])))
        self.projection = nn.Linear(self.embed_dim, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers,
                            bidirectional = True, batch_first = True, dropout = options['dp_ratio'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = options['dp_ratio'])

        self.lin1 = nn.Linear(self.hidden_size * self.directions * self.concat, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin3 = nn.Linear(self.hidden_size, options['out_dim'])

        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.out = nn.Sequential(
			self.lin1,
			self.relu,
			self.dropout,
			self.lin2,
			self.relu,
			self.dropout,
			self.lin3
		)

    def forward(self, batch):
        #premise [seq_len, batch_size]
        #hypothesis [seq_len, batch_size]
        

        premise_embed = self.embedding(batch.premise)
        hypothesis_embed = self.embedding(batch.hypothesis)

        #premise_embed = [seq_len, batch_size, embed_dim]
        #hypothesis_embed = [seq_len, batch_size, embed_dim]

        premise_proj = self.relu(self.projection(premise_embed))
        hypothesis_proj = self.relu(self.projection(hypothesis_embed))

        #premise_proj = [seq_len, batch_size, hidden_size]
        #hypothesis_proj = [seq_len, batch_size, hidden_size]

        h0 = c0 = torch.tensor([]).new_zeros((self.num_layers * self.directions, batch.batch_size, self.hidden_size)).to(self.device)

        #h0 = c0 = [4, batch_size, hidden_size]

        _, (premise_ht, _) = self.lstm(premise_proj, (h0, c0))
        _, (hypothesis_ht, _) = self.lstm(hypothesis_proj, (h0, c0))

        #premise_ht = [4, batch_size, hidden_size]
        #hypothesis_ht = [4, batch_size, hidden_size]

        premise = premise_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)
        hypothesis = hypothesis_ht[-2:].transpose(0, 1).contiguous().view(batch.batch_size, -1)

        #premise = [batch_size, 2*hidden_size]
        #hypothesis = [batch_size, 2*hidden_size]

        combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), 1)
        
        #print('Combined size: ', combined.size())

        #combined = [batch_size, 2*4*hidden_size]

        return self.out(combined)
        # [batch_size, out_dim]

def bilstm(options):
	return BiLSTM(options)
