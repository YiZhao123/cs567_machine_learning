from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  allpro = np.zeros([N, S])

  for s in range(S):
    allpro[0, s] = pi[s]

  '''for n in range(1, N):
    for s in range(S):
      temppro = 0
      for index in range(S):
        temppro += A[index, s]*allpro[n-1, index]
      allpro[n, s] = temppro'''

  for j in range(S):
    alpha[j, 0] = allpro[0, j]*B[j, O[0]]

  for time in range(1, N):
    for j in range(S):
      pro = 0
      for innerj in range(S):
        pro += alpha[innerj, time-1] * A[innerj, j]
      pro = pro * B[j, O[time]]

      alpha[j, time] = pro

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################

  for j in range(S):
    beta[j, N-1] = 1

  for time in range(N-2, -1, -1):
    for j in range(S):
      pro = 0
      for innerj in range(S):
        pro += beta[innerj, time+1] * A[j, innerj] * B[innerj, O[time+1]]

      beta[j, time] = pro

  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  S = len(alpha)
  lastt = len(alpha[0])-1
  for j in range(S):
    prob += alpha[j, lastt]
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  for j in range(len(pi)):
    prob += beta[j, 0] * pi[j] * B[j, O[0]]

  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  pathdic = {}
  deltazero = []
  deltaone = []

  finalzero = 0
  finalone = 0

  N = len(O)
  temp0 = pi[0]*B[0, O[0]]
  temp1 = pi[1]*B[1, O[0]]
  deltazero.append(temp0)
  deltaone.append(temp1)
  for time in range(1,N):
    tempdict = {}
    lastzero = deltazero[len(deltazero)-1]
    lastone = deltaone[len(deltaone) - 1]

    hehe0 = lastzero * A[0,0] * B[0, O[time]]
    hehe1 = lastone * A[1,0] * B[0, O[time]]
    if(hehe0 > hehe1):
      tempdict[0] = 0
      if(time == N-1):
        finalzero = hehe0
      deltazero.append(hehe0)
    else:
      tempdict[0] = 1
      if (time == N - 1):
        finalzero = hehe1
      deltazero.append(hehe1)

    hehe0 = lastzero * A[0, 1] * B[1, O[time]]
    hehe1 = lastone * A[1, 1] * B[1, O[time]]
    if (hehe0 > hehe1):
      tempdict[1] = 0
      if (time == N - 1):
        finalone = hehe0
      deltaone.append(hehe0)
    else:
      tempdict[1] = 1
      if (time == N - 1):
        finalone = hehe1
      deltaone.append(hehe1)

    pathdic[time] = tempdict

  lastone = 0
  if finalzero > finalone:
    path.append(0)
    lastone = 0
  else:
    lastone = 1
    path.append(1)

  for time in range(5, 0, -1):
    nowdict = pathdic[time]
    backwardone = nowdict[lastone]
    path.append(backwardone)
    lastone = backwardone

  path.reverse()
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()