import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from FSA_RNN import FSA_RNN
from FSA_DataLoader import FSA_DataLoader
from mtsl_random import generate_mtsl_acceptor

#target_grammar = generate_mtsl_acceptor()
#print(target_grammar(None))

#this grammar is balanced in terms of accepted/rejected strings in sigma*
sigma='01'
def target_grammar(w):
    return w != '' and w[0] == w[-1]

# Instantiate the model
model = FSA_RNN(2, 5) #this architecture can learn this grammar!!

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print_delay = 1000
rolling_loss = 0

# Train the model
for epoch in range(4):
    for batch_idx, (inputs, targets) in enumerate(FSA_DataLoader(sigma, 16, target_grammar, 8)):
        # Set the model to training mode
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, targets)

        # Compute the loss
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        rolling_loss += loss.item()

        # Print the loss every few iterations
        if batch_idx % print_delay == 0:
            print(f'Epoch {epoch}, Iteration {batch_idx}: Average Loss over past {print_delay} iterations: {rolling_loss / print_delay}')
            rolling_loss = 0

'''torch.save(model, open('model.pt', 'wb'))
with open('mtsl.g', 'w') as mtsl:
    mtsl.write(repr(target_grammar(None)))'''