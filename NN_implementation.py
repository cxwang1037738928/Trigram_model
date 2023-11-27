import torch
from math import exp, log
import torch.nn.functional as F
import matplotlib.pyplot as plt

word = open("names.txt").read().splitlines()
# 3D array to store weights and probabilities
chars = sorted(list(set(''.join(word))))
# creating the lookup table
stoi = {s:i+1 for i, s in enumerate(chars)} # i + 1 so every character is shifted by 1
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

#creating the training set

xs, ys = [], [] 
for w in word:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2, ch3 in zip(chrs, chrs[1:], chrs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append([ix1, ix2])
        ys.append(ix3)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27*2, 27), generator=g, requires_grad=True) # inital weight of 27 neurons, forward pass

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() / 2 # the number of two word combinations in txt file
# print(xs.shape)
# print(ys.shape)
# print(W.shape)


for k in range(50):
    xenc = F.one_hot(xs, num_classes=27).float() # does not change data type, which is why we need .float()
    yenc = F.one_hot(ys, num_classes=27).float()
    
    #softmax after input

    logits = xenc.view(-1, 27*2) @ W # log counts, -1 in view makes it assume dimension 
    counts = logits.exp() # exponentiated
    prob = counts/counts.sum(1, keepdim=True) #probabilities for the next character
    #calculating loss

    loss = -prob[torch.arange(ys.shape[0]), ys].log().mean() + 0.01*(W**2).mean() # regularization, making W cnverge on 0

    #backward pass

    W.grad = None #set zeroes to gradient
    loss.backward()
    #updating the weights
    W.data += -50*W.grad

for i in range(5):
    out = []
    ix = 0
    iy = 1
    while True:
        xenc = F.one_hot(torch.tensor([ix, iy]), num_classes=27).float()
        logits = xenc.view(-1, 27*2) @ W
        counts = logits.exp()
        p = counts/counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))