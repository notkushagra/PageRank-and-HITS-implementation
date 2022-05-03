<h1> HITS Hyperlink Induced Topic Search </h1>

## Libraries imported:
`numpy`
`networkx`
`nltk`
`collections`
<h2> Process </h2>

## Step 1
Query is inputted from the user followed by preprocessing of the query.
Preprocessing steps involve:</br>
1. Stopword removal</br>
2. Stemming</br>
3. Removal of special characters</br>
4. Spelling correction</br>

## Step 2
The corpus is accessed by accessing the pagecontents of different nodes in the `web_graph`
We get the text corpus by accessing the page contents of different nodes in the web graph.
Inverted index is created for all the web pages in the corpus following the pre-processing steps of Stopword removal, Stemming and Removal of special characters.


## Step 3
The Root Set is created which contains the original list of web pages in which the given query is present.
The Base Set is generated which contains the Root Set and all other in-edges and out-edges for all the pages in the Root Set.
The Subgraph and the Adjacency matrix is generated for the base set.

## Step 4
Hub Score and Authority Scores are calculated.

Hubs Scores is calculated as a normalised eigen vector of A.A-Transpose 
Authorities Scores is calculated as a normalised eigen vector of A-Transpose.A
The eigen vector calculation of the matrix is done using the eig function under numpy.linalg.

## Step 5
The Hub Score and the Authority Scores are displayed for all the pages in the base set.


<h3>To Run </h3>
Make sure the folder structure is the same as follows and the gpickle file is in the same location. If you want to you may include your own gpickle in the file.
Run the python file in the command line using the following commnad.

```
python HITS.py
```
