import torch
from math import exp, log
import matplotlib.pyplot as plt
import matplotlib

log_likelihood = 0.0
word = open("names.txt").read().splitlines()
# 3D array to store weights and probabilities
#N_2 = torch.zeros((27, 27), dtype=torch.int32)
N = torch.zeros((27, 27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(word))))
# creating the lookup table
stoi = {s:i+1 for i, s in enumerate(chars)} # i + 1 so every character is shifted by 1
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}



for w in word:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2, ch3 in zip(chrs, chrs[1:], chrs[2:]): 
    #for ch1, ch2 in zip(chrs, chrs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        N[ix1, ix2, ix3] += 1

        #N_2[ix1, ix2] += 1


#plots the distribution of words

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 6}
# matplotlib.rc('font', **font)
# plt.figure(figsize=(16, 16))
# plt.imshow(N_2, cmap='Blues')
# for i in range(27):
#     for x in range(27): 
#         chrstr = itos[i] + itos[x]
#         plt.text(x, i, chrstr, ha='center', va='bottom', color='grey') #where texts are in boxes
#         plt.text(x, i, N[i, x][1].item(), ha='center', va='top', color='grey') # texts are the number of occurences instead of comb of letters
# plt.axis('off')
# plt.show()



g = torch.Generator().manual_seed(2147483647) # deterministic generator for outcomes
# p = torch.rand(3, generator=g) # generates 3 probabilities
# p = p/p.sum()

# ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item(), samples 1 outcome using probabilities in p, generates a letter



log_likelihood = 0.0
P = (N+1).float() # making a tensor object that has its values be probabilities instead of number of occurences, modeling smoothing
P = P/P.sum(2, keepdim=True) # int is dim to reduce, (0, 1) reduces to 1, 1, 27, 0 reduces to 1, 27, 27, 2 reduces to 27, 27, 1
                             # Need keepdim=True, else its 27, 27, or risk of normalizing wrong dimension
n = 0
for w in word:
    chrs = ["."] + list(w) + ["."]
    for ch1, ch2, ch3 in zip(chrs, chrs[1:], chrs[2:]): 
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        prob = P[ix1, ix2, ix3]
        logprob = torch.log(prob) # calculatiing log likelihood of a single combination
        #print(f'{ch1}{ch2}{ch3} {logprob}')
        log_likelihood += logprob # since log likelihood is culmulative e.g log(a*b*c) = log(a) + log(b) + log(c)
        n += 1
nll = -1*log_likelihood # negative log likelihood  
print(nll/n) #loss function balanced by taking the average



for i in range(10):
    out = []
    iz = 0
    while True:
        #p = N[ix][iy].float() #given two words, this is the probability of the next word
        p = torch.rand(27, generator=g)
        p = p/p.sum()
        #print(p)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # next two words
        iy = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() 
        out.append(itos[ix])
        out.append(itos[iy])
        pz = P[ix][iy]
        pz = pz/pz.sum()
        iz = torch.multinomial(pz, num_samples=1, replacement=True, generator=g).item() # next word after randomly generating two words
        #print(itos[ix])
        out.append(itos[iz])
        break
    print(''.join(out))

#print(P)


