from random import shuffle
from Lambert import *
from torch import Tensor

def FSA_DataLoader(sigma, maxLen, acc, batch_size=8):

    inputs = []; targets = []

    def pad(elements, length):
        return elements + [-1 for _ in range(length - len(elements))]

    sigstar = list(star(sigma, maxLen))
    shuffle(sigstar)

    for i, w in enumerate(sigstar):
        inputs.append(pad(list(map(int, w)), maxLen))
        targets.append(1 if acc(w) else 0)

        if i%batch_size == 0:
            yield Tensor(inputs).long(), Tensor(targets).long()
            inputs = []; targets = []