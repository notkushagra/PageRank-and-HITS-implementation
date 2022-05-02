#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np 
import collections

import networkx as nx #pip install networkx
from numpy.linalg import eig

import nltk
from nltk.tokenize import sent_tokenize , word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from collections import defaultdict
from collections import deque

stwords = set(stopwords.words('english'))

web_graph = nx.read_gpickle("web_graph.gpickle")
G=web_graph


# In[2]:


def editDist(word1, word2):
    '''
    Finds the distance between two similar words.
    '''
    m=len(word1)
    n=len(word2)
    
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j    # Minimum possible operations for this case = j

            elif j == 0:
                dp[i][j] = i    # Minimum possible operations for this case = i
 
            elif word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Count Insert
                                   dp[i-1][j],        # Count Remove
                                   dp[i-1][j-1])      # Count Replace and find minimum
 
    return dp[m][n]


# In[3]:


def getClosest(word):
    '''
    Function returns the closest word in the dictionary.
    '''
    dictionary = {}
    for w in ii.keys():
        dist=editDist(word,w)
        dictionary[w]=dist
        if dist == 1:
            return w
    v=(sorted(dictionary.items(), key=lambda item: item[1]))
    return v[0][0] 


# In[4]:


def inverted_index_creation():
    '''
    This function creates an inverted index list for all the data present in the dataset.
    Here the data set is being accessed by accessing the different pagecontent of the nodes in the web graph
    '''

    inverted_index = defaultdict(set)

    #importing stopwords from nltk and making it into a set
    stwords = set(stopwords.words('english'))

    #importing the porterStemmer
    ps = PorterStemmer()

    def remove_special_characters(text):
        import re
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex,'',text)
        return text_returned

    web_graph = nx.read_gpickle("web_graph.gpickle")
    G=web_graph

    for docId in range(len(G)):
        text=G.nodes[docId]['page_content']
        text=remove_special_characters(text)

        for sent in sent_tokenize(text):
            for word in word_tokenize(sent):
                word_lower = word.lower()
                if word_lower not in stwords:
                    #stemms the words
                    word_stem = ps.stem(word_lower)
                    inverted_index[word_stem].add(docId)
    return inverted_index


# In[5]:


def print_hubs():
    '''
    This function prints the hubs vector in a dictionary fashion.
    It prints the hub score of a particular node w.r.t. the query
    '''
    print("Nodes(DocId)    Hub Score")
    for i in range(len(baseset)):
        print(str(baseset[i])+"            :"+str(hubs[0,i]))


# In[6]:


def print_authorities():
    '''
    This function prints the authorities vector in a dictionary fashion.
    It prints the authority score of a particular node w.r.t. the query
    '''
    print("Nodes(DocId)    Authority Score")
    for i in range(len(baseset)):
        print(str(baseset[i])+"            :"+str(authorities[0,i]))


# In[7]:


node_index= 0
G.nodes[node_index]['page_content']


# In[8]:


pos={i:web_graph.nodes[i]['pos'] for i in range(len(web_graph.nodes))}

nx.draw(web_graph,pos)


# In[9]:


ii=inverted_index_creation()


# In[10]:


rootset=[]
baseset=[]

query = input('Enter your query:')
    
query = query.lower()
query_tokens=query.split()
query_words=[]

q = deque()

ps=PorterStemmer()

for word in query_tokens:
    word=word.lower()
    if word not in stwords:
        word=ps.stem(word)
        query_words.append(word)

for word in query_words:
    if word in ii.keys():
        set_a=ii[word]
        set_contains=(set(set_a)) #typecasting to set
        q.append(set_contains)

    else:
        cword=getClosest(word)
        set_a=ii[cword]
        set_contains=(set(set_a))
        q.append(set_contains)
        continue

n=len(query_words)

for i in range(n-1):
    set_a = q.pop()
    set_b = q.pop()
    unioned=set(set_a).union(set(set_b))
    q.appendleft(unioned)

rootset=list(collections.deque(q[0]))


# In[11]:


baseset=rootset.copy()


# In[12]:


for i in range(len(rootset)):
    ind=rootset[i]
    in_edges=list(G.in_edges(ind))
    out_edges=list(G.out_edges(ind))
    len_in_edges=len(in_edges)
    len_out_edges=len(out_edges)
    for j in range(len_in_edges):
        baseset.append(in_edges[j][0])
    for j in range(len_out_edges):
        baseset.append(out_edges[j][1])


# In[13]:


baseset=np.array(baseset)
baseset=np.unique(baseset)


# In[14]:


sub_graph=web_graph.subgraph(baseset)


# In[15]:


n=len(sub_graph.nodes)
A=nx.adjacency_matrix(sub_graph).todense()
#print(A.todense())
AT=A.transpose()
#print(AT.todense())
# a=np.ones(n)
# h=np.ones(n)
# x=np.zeros((n,n))
# print(A)
# print()
# print(AT)
# print()


# In[16]:


nx.hits(sub_graph)


# In[17]:


X = A@A.T
h=eig(X)[1][:,0]
h = h/sum(h)
#w,v=eig(a)
#print(v[:,1]
hubs=h.T



# In[18]:


Y =A.T@A
a=eig(Y)[1][:,0]
a= a/sum(a)
#w,v=eig(a)
#print(v[:,1]
authorities=a.T

print()
print("The rootset for the input query is :")
print(rootset)
print()

print("The baseset for the input query is :")
print(baseset)
print()

print("The hubs vector for the subgraph is :")
print(hubs)
print()

print("The authorities vector for the subgraph is :")
print(authorities)
print()

print_hubs()
print()
print_authorities()
print()





