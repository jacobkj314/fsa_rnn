from torch.nn import Module, Embedding, ReLU
import torch.nn.functional as F
from torch import randn, Tensor
import torch

class FSA_Embedding(Embedding):
    def __init__(self, vocab_size, hidden_dim, **kwargs):
        self.vocab_size = vocab_size
        super().__init__(num_embeddings=vocab_size, 
                       embedding_dim = hidden_dim * hidden_dim,
                       **kwargs)
        self.padding = torch.eye(hidden_dim).flatten()[None,:] #set extra layer to identity matrix, to be used for padding 
        self.padding.requires_grad = False
        #self.weight.data = torch.zeros_like(self.weight.data)#set all to zeros

    def forward(self, input: Tensor) -> Tensor:
        self.padding.requires_grad = False
        return F.embedding(
            input % self.weight.shape[0], torch.concat((self.weight, self.padding)), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class FSA_RNN(Module):
    def __init__(self, alphabet_size = 10, num_states=6):
        super().__init__()

        self.alphabet_size = alphabet_size
        self.num_states = num_states
        self.transitions = FSA_Embedding(alphabet_size, num_states)
        self.start_states = torch.nn.Parameter(randn(num_states, requires_grad=True))
        self.final_states = torch.nn.Parameter(randn(num_states, requires_grad=True))
        self.relu = ReLU()

    def _forward_step(self, x : Tensor, hidden_state):
        batch_size = x.shape[0]
        embedding = self.transitions(x).reshape((batch_size, self.num_states, self.num_states))
        out = self.relu(embedding) @ hidden_state
        return out
    
    def forward(self, x : Tensor, label=None):
        batch_size, string_length = x.shape

        first_hidden_state = torch.stack([self.start_states] * batch_size, dim=0)[:,:,None] #add extra dimension for broadcasting

        hidden_states = [self.relu(first_hidden_state)]

        for token in range(string_length):
            hidden_states.append(self._forward_step(x[:,token], hidden_states[-1]))

        score = torch.sum(hidden_states[-1] * torch.stack([self.final_states] * batch_size, dim=0)[:,:,None], axis = (1,2)) #want to sum along hidden_size and dummy axis

        if label is None:
            return score
        
        loss = torch.sum( # sum across instances
                            torch.abs(score - label) * ((2 * label - 1)*(score - label) < 0)
                        ) / batch_size 
        #regularization attempts:
        #   #These try to balance the weights corresponding to edges, so the negativeness of the negative weights is about as negative as the positiveness of the positive weights
        + torch.abs(torch.sum(self.final_states))
        + torch.abs(torch.sum(self.start_states))
        + torch.abs(torch.sum(self.transitions.weight))
        #   #this one tries to minimize the number of transitions
        + torch.sum(self.transitions.weight > 0)
        
        return {"score":score, "loss":loss}
    
    def visualize(self):
        states = [str(i) for i in range(self.num_states)]

        #label states
        for i, final in enumerate(self.final_states > 0):
            if final:
                states[i] = 'F' + states[i]
        for i, start in enumerate(self.start_states > 0):
            if start:
                states[i] = 'I' + states[i]

        joveString = 'NFA\n'

        for symbol, symbol_transitions in enumerate(self.transitions.weight > 0):
            for from_to, transition in enumerate(symbol_transitions):
                fr, to = from_to % self.num_states, from_to // self.num_states
                if transition:
                    joveString += f'{states[fr]} : {symbol} -> {states[to]}\n'

        return joveString
            
        
