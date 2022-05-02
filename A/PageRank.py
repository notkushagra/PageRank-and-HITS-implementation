import numpy as np
import networkx as nx

# make a matrix
# transform it to tp matrix
# apply power iteration

def print_matrix(mat):
    """
    This function prints the matrix
    """
    rows=len(mat)
    columns=len(mat[0])
    for i in range(rows):
        for j in range(columns):
            print(mat[i][j],end=' ')
        print()

        

def generate_tp_matrix(alpha,G):
    """
    This function generates the total probability matrix.
    """
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
# 0 0 1 0 0 0 0 0 1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 1 1 0 1

# P=generate_tp_matrix(0.14,test)

# print(P)

def power_itr100(xp,P):
    """
    This function multiplies matrix P with matrix xp 100 times.
    """
    count=100
    prev=np.zeros(columns)

    while(count):
        prev=xp
        xp=np.dot(xp,P)
        count-=1
    print(xp)

def power_itr_inf(xp,P):
    """
    This function keeps on multiplying matrix P with matrix xp till the resultant matrix attains a constant value.
    """
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
    """
    This function takes input for adjacency matrix.
    For some matrix M and particular i,j values, M[i][j]=1 or(M[i][j]>1) denotes the presence of an edge between node i and node j.
    M[i][j]=0 denotes that there is no edge between node i and node j.
   
    """
    irows=int(input("Enter number of points: "))
    icolumns=irows
    mat=np.zeros((irows,icolumns))
    for i in range(irows):
        for j in range(icolumns):
            mat[i][j]=int(input("Is "+'A'-str(i)+" and "+str(j)+" connected?\nPress 1 for yes and 0 for NO :"))
    return mat

def input_adj_as_matrix():
    """
    This function takes input for adjacency matrix. The user enters the values in a linear fashion.
   
    """
    R = int(input("Enter the number of nodes: "))
    edges = int(input("Enter the number of edges: "))
    C = R
    matrix = np.zeros((R, C))   
    
    for k in range(edges):
        i=int(input())
        j=int(input())
        matrix[i-1][j-1]=1

    return matrix

# Taking input of damping factor(alpha).

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

# Printing the final page rank vector.
print(PageRank_vector)
print()
