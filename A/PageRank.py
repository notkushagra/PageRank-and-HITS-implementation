import numpy as np
import scipy as sc
import scipy.linalg
import networkx as nx
import matplotlib.pyplot as plt


# make a matrix
# transform it to tp matrix
# apply power iteration

def generate_matrix(n) :
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2) > (np.abs(c - c.T) + 1)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c

def print_matrix(mat):
    rows=len(mat)
    columns=len(mat[0])
    for i in range(rows):
        for j in range(columns):
            print(mat[i][j],end=' ')
        print()

def generate_tp_matrix(alpha,G):
    rows=len(G)
    N=rows
    columns=len(G[0])
    P = np.zeros((rows,columns))
    sum_of_1s=[]
    
    #sum_of_1s[rows]
    for i in range(rows):
        rowsum=0;
        for j in range(columns):
            if(G[i][j]==1): rowsum+=1
        sum_of_1s.append(rowsum)
        #print(sum_of_1s[i])
        
    #dividing by row sum
    for i in range(rows):
        for j in range(columns):
            if(G[i][j]==1): 
                temp=G[i][j]/sum_of_1s[i]
                #print(temp)
                P[i][j]=temp
    
    #multiply by 1-alpha
    for i in range(rows):
        for j in range(columns):
            P[i][j]*=(1-alpha)
            
    #adding alpha by n to P
    for i in range(rows):
        for j in range(columns):
            P[i][j]+=(alpha/N)
    return P

# test= np.array([[0,0,1,0,0,0,0],
#                 [0,1,1,0,0,0,0],
#                 [1,0,1,1,0,0,0],
#                 [0,0,0,1,1,0,0],
#                 [0,0,0,0,0,0,1],
#                 [0,0,0,0,0,1,1],
#                 [0,0,0,1,1,0,1]], dtype=float)

# P=generate_tp_matrix(0.14,test)

# print(P)

def power_itr100(xp,P):
    count=100
    prev=np.zeros(columns)

    while(count):
        prev=xp
        xp=np.dot(xp,P)
        count-=1
    print(xp)

def power_itr_inf(xp,P):
    count=0
    prev=np.zeros(columns)

    while(True):
        if np.array_equal(prev,xp):
            break;
        prev=xp
        xp=np.dot(xp,P)
        count+=1
    print("The value of xP and the next iteration becomes same after "+ str(count)+" iterations")
    return xp

def input_adj():
    irows=int(input("Enter number of points: "))
    icolumns=irows
    mat=np.zeros((irows,icolumns))
    for i in range(irows):
        for j in range(icolumns):
            mat[i][j]=int(input("Is "+'A'-str(i)+" and "+str(j)+" connected?\nPress 1 for yes and 0 for NO :"))
    return mat

def input_adj_as_matrix():
    R = int(input("Enter the number of rows:"))
    C = R
       
    print("Enter the entries in a single line (separated by space): ")

    entries = list(map(int, input().split()))
      
    matrix = np.array(entries).reshape(R, C)
    return matrix

alpha=float(input("\nEnter the value of alpha: "))
G=input_adj_as_matrix()

P=generate_tp_matrix(alpha,G)

columns=len(G[0])
x=np.zeros(columns)
x[0]=1
xp=np.dot(x,P)
print()
PageRank_vector=power_itr_inf(xp,P)
print("\nThe PageRank vector of the input Graph is as follows: ")
print(PageRank_vector)

