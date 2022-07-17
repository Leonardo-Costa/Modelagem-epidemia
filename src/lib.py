import random
import math
import networkx as nx
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
import numpy as np

def GetGraph(N, M, seed=None):
  """ 
  Args:
      N (int): número de elementos do grafo
      M (int): metade do grau médio do grafo
      seed (int): seed de geração do grafo

  Returns:
      graph: grafo de uma rede livre de escala
  """

  G = nx.barabasi_albert_graph(n = N, m = M, seed=seed, initial_graph = None)
  return G

def FindLength(l):
  """"
  Args:
       l (list): lista a ser analisada

  Returns:
      int: comprimento da lista
  """
  i = 0
  while True:
    try:
      if l[i] != None:
        pass
    except:
      break
    i += 1
  return i

def GetMatrix(N, edgesLen, edges):
  """
  Args:
      N (int): número de elementos de um grafo
      edgesLen (int): comprimento da lista edges
      edges (list): lista de conexões 

  Returns:
      list: matrix de adjacência do grafo
  """

  mtx = []
  for i in range(N):
    mtx.append([])
    for j in range(N):
      mtx[i].append([])
  if edgesLen > 2:
    for i in range(edgesLen):
      first = edges[i][0]
      scnd = edges[i][1]
      mtx[first][scnd] = 1
      mtx[scnd][first] = 1

    for i in range(len(mtx)):
      for j in range(len(mtx[i])):
        if mtx[i][j] != 1:
          mtx[i][j] = 0
  else:
    mtx = [
              [[0],[1]],
              [[1],[0]]
              ]

  return mtx

def infect(element, probabilidade):
  """
  Args:
      element (int): elemento do grafo
      probabilidade (float): probabilidade de um elemento contrarir uma população inicial do patogeno de um elemento transmissor a cada encontro/iteraçao do loop
  """
  
  temp = findConections(element)
  spread(temp, probabilidade)

def findConections(element):
  """
  Args:
      element (int): elemento do grafo

  Returns:
      list: conexões desse elemento
  """
  conections = []
  for i in range(len(matrix)):
    if matrix[element][i] == 1:
      conections.append(i)
  return conections

def spread(conections, probabilidade):
  """
  Args:
      conections (list): lista de conexões de um elemento do grafo
      probabilidade (float): probabilidade de um elemento contrarir uma população inicial do patogeno de um elemento transmissor a cada encontro/iteraçao do loop
  """
  for i in range(FindLength(conections)):
    rN =  random.randint(0, 1 / probabilidade)
    if lista[conections[i]][-1] == 0 and rN == 0:
      V[conections[i]] = 1
      if C[conections[i]] == 0:
        C[conections[i]] = 1

def SaveData(Tmax, DT, k, minAmmount, probabilidade, N, M, pMin, pMax, seed, graphSeed, Amostragem, Decimais, file):
  """Salva os dados em um arquivo .txt da simulação realizada

  Args:
      Tmax (int): tempo da simulação em segundos
      DT (float): passo de integração
      k (int): crescimento máximo/saturação do patógeno
      minAmmount (int): quantidade minima para que alguem seja considerado transmissor
      probabilidade (float): probabilidade de contagio a cada encontro
      N (int): numero de elementos na rede
      M (int): metade do grau medio da rede
      pMin (float): parâmetro referente ao sistema imunológico
      pMax (float): parâmetro referente ao sistema imunológico
      seed (float): seed para os numero aleatórios da probabilidade de contagio
      graphSeed (float): seed para a geração do grafo
      Amostragem (int): amostragem para salvar os valores no arquivo .txt
      Decimais (int): quantidade de decimais a serem salvos
      file (str): nome do arquivo a ser salvo
  """
  print(file)
  f = open(file, 'a')  
  for i in range(N):
    for j in range(len(lista[i])):
      if j != len(lista[i]):
        f.write(str(round(lista[i][j], Decimais)) + ',')
    f.write(';')
  parameters = str(Tmax) + ',' + str(DT) + ',' + str(k) + ',' + str(minAmmount) + ',' + str(probabilidade) + ',' + str(N) + ',' + str(M) + ',' + str(pMin) + ',' + str(pMax) + ',' + str(seed) + ',' + str(graphSeed) + ',' + str(Amostragem) + ',' + str(Decimais)
  f.write(parameters)
  f.write('-')
  f.close()

def CleanData(file, condition=False):
  """limpa o aqruivo passado como argumento

  Args:
      file (str): nome do arquivo
      condition (bool, optional): Parametro para definir se executa a função ou nao. Defaults to False.
  """
  if condition:
    f = open(file, 'w').close()
  GetSize(file)

def ShowData(file):
  """Exibe gráficos a partir dos dados recuperados do arquivo .txt gerado pela função SaveData

  Args:
      file (str): nome do arquivo
  """
  
  params = []
  content = []
  data = GetData(file)
  for i in range(len(data)):
    content.append(data[i])
    params.append(data[i][-1])

  for k in range(len(content)):
    print('---------- SIMULAÇÃO {} ----------'.format(k + 1))
    G = GetGraph(int(params[k][5]), int(params[k][6]),seed=int(params[k][10]))
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    listaDiscretizada = []
    for i in range(int(params[k][5])):
      for j in range(len(content[k][i])):
        if content[k][i][j] >= 60 and content[k][i][j - 1] < 60:
            listaDiscretizada.append(j)
    listaDiscretizada.sort()

    listaDiscretizada2 = []
    for i in range(len(content[k][0])):
      sum = 0
      for j in range(int(params[k][5])):
        sum += 1 if content[k][j][i] != None and content[k][j][i] > params[k][3] else 0
      listaDiscretizada2.append(sum)
  
    fig = plt.figure("Degree of a random graph", figsize=(10, 10))
    axgrid = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(axgrid[1, 0])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Distribuição de graus")
    ax1.set_ylabel("Grau")
    ax1.set_xlabel("Quantidade de elementos")

    ax2 = fig.add_subplot(axgrid[1, 1])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Histograma de grau")
    ax2.set_xlabel("Grau")
    ax2.set_ylabel("Quantidade de elementos")

    ax3 = fig.add_subplot(axgrid[0, 0:])
    for i in range(int(params[k][5])):
      ax3.plot(content[k][i])
    temp = []
    for i in range(int(params[k][0] / (params[k][1] * params[k][11]))):
      temp.append(params[k][3])
    ax3.plot(temp)

    ax4 = fig.add_subplot(axgrid[1, 2])
    ax4.bar(list(range(len(listaDiscretizada))), listaDiscretizada)
    ax4.set_title("Primeira infecção")
    ax4.set_xlabel("Elementos da rede")
    ax4.set_ylabel("Tempo")
    fig.tight_layout()

    ax5 = fig.add_subplot(axgrid[2, 0:])
    ax5.plot(listaDiscretizada2)
    ax5.set_title("Quantidade de casos")
    ax5.set_xlabel("Tempo")
    ax5.set_ylabel("Infectados transmissores")
    plt.savefig(file[:-4] + 'png')
    #plt.show()

def Simulate(Tmax=1000, DT=0.01, k=100, minAmmount=60, probabilidade=0.001, N=20, M=4, pMin=0.01, pMax=0.08, seed=10, graphSeed=10, Amostragem=10, Decimais=3, file=None):
  """Realiza a simulação com os parâmetros passados e salva os valores usando a função SaveData

   Args:
      Tmax (int): tempo da simulação em segundos
      DT (float): passo de integração
      k (int): crescimento máximo/saturação do patógeno
      minAmmount (int): quantidade minima para que alguem seja considerado transmissor
      probabilidade (float): probabilidade de contagio a cada encontro
      N (int): numero de elementos na rede
      M (int): metade do grau medio da rede
      pMin (float): parâmetro referente ao sistema imunológico
      pMax (float): parâmetro referente ao sistema imunológico
      seed (float): seed para os numero aleatórios da probabilidade de contagio
      graphSeed (float): seed para a geração do grafo
      Amostragem (int): amostragem para salvar os valores no arquivo .txt
      Decimais (int): quantidade de decimais a serem salvos
      file (str): nome do arquivo a ser salvo
  """
  G = GetGraph(N=N, M=M, seed=graphSeed)
  random.seed(seed)
  global lista
  global matrix
  global C
  global V
  DV = [] 
  DC = [] 
  V = []
  C = []
  f = []
  lista = []
  matrix = GetMatrix(N, len(list(G.edges())), list(G.edges()))
  for i in range(N):
    DV.append(0)
    DC.append(0)
    V.append(0)
    C.append(0)
    f.append(random.uniform(pMin, pMax))

  V[1] = 1
  C[1] = 1
  f[1] = pMin

  for i in range(N):
    lista.append([])
  for i in tqdm(range(int(Tmax / DT))): #loop da simulação
    for j in range(N): #loop dos elementos da rede
      DV[j] = V[j] * 0.1*(1 - V[j] / k) * DT
      DC[j] = (f[j] * C[j]) * DT    
      V[j] += DV[j]
      C[j] += DC[j]
      res = V[j] - C[j] if (V[j] - C[j]) >=0 else 0
      if i % Amostragem == 0:
        lista[j].append(res)
      if res >= minAmmount and i % 10 == 0:
        infect(j, probabilidade)  
  SaveData(Tmax, DT, k, minAmmount, probabilidade, N, M, pMin, pMax, seed, graphSeed, Amostragem, Decimais, file)

def GetData(file):
  """Recupera os dados de um arquivo .txt

  Args:
      file (str): nome do arquivo

  Returns:
      list: lista com os dados do arquivo
  """
  f = open(file, 'r')
  content = f.read()
  content = content.split('-')
  if '' in content:
    content.remove('')
  for i in range(len(content)):
    content[i] = content[i].split(';')
    if '' in content[i]:
      content[i].remove('')
    for j in range(len(content[i])):
      content[i][j] = content[i][j].split(',')
      if '' in content[i][j]:
        content[i][j].remove('')
      for k in range(len(content[i][j])):
        content[i][j][k] = float(content[i][j][k].strip('e')) 
  f.close()
  return content
  
def GetSize(file):
  """Retorna o tamanho de um arquivo

  Args:
      file (str): nome do arquivo
  """
  size = os.path.getsize(file)
  size = str(float(size)/1000000)
  print('{}: {} MB'.format(file, size[:6]))
