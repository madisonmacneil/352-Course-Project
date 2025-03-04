#seq2seq model NL --> SQL 

import torch 
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F 
import numpy as np 
import spacy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

en_nlp = spacy.load("en_core_web_sm")

#Encoder
    # Input embedding layer: converts tokens to dense vector representations

    # LTSM architecture to capture contextual information 

#Context Vector: compressed representation of the entire input sequence 
                #Created by the encoder's final hidden state 
                #bridge between encoder and decoder 
#Decoder 
    # Initial Hidden State: Initialized using the context vector from the encoder 
    # Converts output tokens into dense vector representations 
    # RNN: generates output tokens sequentially (context vector +previously generated tokens as input )
    # Attention Mechanism: allows decoder to focuse on different parts of the input 

#Output Layer: 
    # Softmax Layer: converts the decoder's hidden state into a probability distribution 

#Training: 
    #Cross Entropy Loss
    #Optimization: Adam 