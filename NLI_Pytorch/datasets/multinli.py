import os
import sys

import torch

from torchtext.legacy.data import Field, Iterator, TabularDataset
from torchtext.legacy import datasets
from utils import makedirs

from pdb import set_trace
import nltk
nltk.download('punkt')
tokenizer = nltk.RegexpTokenizer(r"\w+").tokenize

__all__ = ['multinli']

class MultiNLI():
    def __init__(self,options):
        
        #self.TEXT = Field(lower=True, tokenize='spacy', batch_first=True) #spacy is too slow
        
        self.TEXT = Field(lower=True, tokenize=tokenizer, batch_first=True)
        self.LABEL = Field(sequential=False, unk_token=None, is_target=True)
        print('data split started')
        self.train, self.dev, self.test = datasets.MultiNLI.splits(self.TEXT, self.LABEL)
        #self.train = generator class<class, datasets.snli.SNLI>
        
        print('data split done...')
        print('Example: ', vars(self.train.examples[0]))
        print(f"Number of training examples: {len(self.train)}")
        print(f"Number of validation examples: {len(self.dev)}")
        print(f"Number of testing examples: {len(self.test)}")

        self.TEXT.build_vocab(self.train, self.dev)
        self.LABEL.build_vocab(self.train)

        vector_cache_loc = '.vector_cache/multinli_vectors.pt' #pytorch use pickle

        if os.path.isfile(vector_cache_loc):
            print('Vectors already present...Loading vectors!!!', end = '\t')
            self.TEXT.vocab.vectors = torch.load(vector_cache_loc)
            print('done...')
        else:
            print('Vectors not present...Downloading vectors!!!', end='\t')
            self.TEXT.vocab.load_vectors('glove.840B.300d')
            makedirs(os.path.dirname(vector_cache_loc))
            torch.save(self.TEXT.vocab.vectors, vector_cache_loc)
            print('done...')
            
        print('Total vocabulary Size is', len(self.TEXT.vocab))
        print('Embedding vectors size is', len(self.TEXT.vocab.vectors))

        self.train_iter, self.dev_iter, self.test_iter = Iterator.splits((self.train, self.dev, self.test),
																		batch_size=options['batch_size'],
                                                                        device=options['device'],
																		sort_key = lambda x: len(x.premise),
                                                                        sort_within_batch = False,
                                                                        shuffle = True)

    def vocab_size(self):
        return len(self.TEXT.vocab)

    def out_dim(self):
        return len(self.LABEL.vocab)

    def labels(self):
        return self.LABEL.vocab.stoi
        
    def top_frequency_token(self):
        return self.TEXT.vocab.freqs.most_common(9)
        
    
    def get_top_k_text_itos(self, k):
        return self.TEXT.vocab.itos[:k]
        
    def get_label_itos(self):
        return self.LABEL.vocab.itos
        
    def get_top_k_vectors(self, k):
        return self.TEXT.vocab.vectors[:k]
        
    def get_label_count(self):
        return self.LABEL.vocab.freqs.most_common()

def multinli(options):
	return MultiNLI(options)
