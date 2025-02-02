
import numpy as np
def process_data(file_path):
  data = open(f'{file_path}.txt', 'r').read()
  data = data.replace("\n", "").replace(" ", "")
  return data
data = process_data("anna")
chars = list(set(data)) # vocabulary
data_size, vocab_size = len(data), len(chars)
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size = 256 # size of hidden state vectors
seq_length = 60 # number of steps to unroll the RNN for
learning_rate = 1e-1

Whx = np.random.randn(hidden_size, vocab_size)*np.sqrt(2.0/(vocab_size + hidden_size)) # input to hidden weight
Whh = np.random.randn(hidden_size, hidden_size)*np.sqrt(2.0/(vocab_size + hidden_size)) # hidden_t-1 to hidden_t weight
Why = np.random.randn(vocab_size, hidden_size)*np.sqrt(2.0/(vocab_size + hidden_size)) # hidden to output weight
bh = np.zeros(hidden_size,) # hidden bias
by = np.zeros(vocab_size,) # output bias

def lossFunc(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  total_loss = 0
  loss_arr = []
  temperature = 0.5
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros(vocab_size,) # encode in 1-of-k representation
    xs[t][char_to_idx[inputs[t]]] = 1
    # print("input --> ", xs[t], xs[t].shape)
    # hs[t] = np.tanh(np.dot(Whx, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    hs[t] = np.tanh(np.dot(Whx, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
    ys[t] = np.dot(Why, hs[t]) + by
    ys[t] = ys[t] / temperature
    ys[t] -= np.max(ys[t])
    # print("output --> ", ys[t])
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    # print("probability --> ", ps[t])
    # print(idx_to_char[np.argmax(ps[t])])
    try:
      loss = -np.log(ps[t][np.argmax(targets[t])])
      total_loss += loss
      loss_arr.append(loss)
    except:
      pass
      # print("last output of RNN::")
      # print(idx_to_char[np.argmax(ps[t])])

  # print("xs shape : ", xs[0].shape)
  # print("ps shape : ", ps[0].shape)
  # print("ys shape : ", ys[0].shape)
  # print("hs shape : ", hs[0].shape)


  # backpropagation
  dWhy = np.zeros_like(Why)
  dWhh = np.zeros_like(Whh)
  dWhx = np.zeros_like(Whx)

  dbh = np.zeros_like(bh)
  dby = np.zeros_like(by)

  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs)-1)):
    dy = np.copy(ps[t])
    dy[np.argmax(targets[t])] -= 1
    # print(ps[t].shape)
    # print("Wdhy shape is : ", dWhy.shape)
    dWhy += np.dot(dy[:, None], hs[t][None, :])
    # print(dWhy.shape) 38,100
    # print(dy.shape)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext
    # Gradient of ReLU
    dhraw = (hs[t] > 0) * dh  # element-wise multiplication for ReLU gradient
    dbh += dhraw
    dWhx += np.dot(dhraw[:, None], xs[t][None, :])
    dWhh += np.dot(dhraw[:, None], hs[t-1][None, :])
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWhx, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -1, 1, out=dparam)

  # printing to see the shapes and outputs
  # print("error in dy: ", dy, "shape of dy: ", dy.shape)
  # print("error in dWhy: ", dWhy, "shape of dWhy: ", dWhy.shape)
  # print("error in dby: ", dby, "shape of dby: ", dby.shape)
  # print("error in dh: ", dh, "shape of dh: ", dh.shape)
  # print("error in dhraw: ", dhraw, "shape of dhraw: ", dhraw.shape)
  # print("error in dbh: ", dbh, "shape of dbh: ", dbh.shape)
  # print("error in dWhx: ", dWhx, "shape of dWhx: ", dWhx.shape)
  # print("error in dWhh: ", dWhh, "shape of dWhh: ", dWhh.shape)
  # print("error in dhnext: ", dhnext, "shape of dhnext: ", dhnext.shape)

  # print(total_loss, dWhx, dWhh, dWhy, dbh, dby, hs[len(inputs)-1])

  return total_loss, dWhx, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
inputs = np.array(list(data))
targets = []
print(inputs)
for t in range(len(inputs)):
  try:
    arr = np.zeros(vocab_size,)
    arr[char_to_idx[inputs[t+1]]] = 1
    print(arr, inputs[t+1])
    targets.append(arr)
  except Exception as e:
    pass
h_init = np.random.randn(hidden_size,)*np.sqrt(2.0/(hidden_size+vocab_size))
def sample(h, initial_input, n):
  x = np.zeros(vocab_size,)
  x[char_to_idx[initial_input]] = 1
  temperature = 0.5
  characters = []
  for t in range(n):
    # print("initial input: ", x, initial_input)
    h = np.tanh(np.dot(Whx, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    y = y / temperature
    y -= np.max(y)
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p)
    # print(f"Probabilities: {p}, Selected index: {ix}, Selected char: {idx_to_char[ix]}")
    x = np.zeros(vocab_size,)
    x[ix] = 1
    characters.append(idx_to_char[ix])
  sample_txt = ''.join(char for char in characters)
  return sample_txt
# min_threshold = 0.05
n, p = 0, 0
mWhx, mWhy, mWhh, = np.zeros_like(Whx), np.zeros_like(Why), np.zeros_like(Whh)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while True:
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros(hidden_size,) # reset RNN memory
    p = 0
  inputs = [ch for ch in data[p : p+seq_length]]
  targets = [ch for ch in data[p+1 : p+seq_length+1]]


  if (n > 0) and (n % 100 == 0):
    sample_text = sample(hprev, inputs[0], 50)
    print(sample_text)
    print("--------$$$$--------")

  loss, dWhx, dWhh, dWhy, dbh, dby, hprev = lossFunc(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  if (n > 0) and (n % 100 == 0):
    print("iteration: ", n, " loss: ", loss)

  for param, dparam, mem in zip([Whx, Whh, Why, bh, by],
                                 [dWhx, dWhh, dWhy, dbh, dby],
                                 [mWhx, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

  # print("inputs: ", inputs, "targets: ", targets)
  # print(loss, dWhx, dWhh, dWhy, dbh, dby, hprev)

  if (n > 5000):
    break

  p += seq_length
  n += 1





