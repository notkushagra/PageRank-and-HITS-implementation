# Page Rank Algorithm Implementation


## Libraries Imported

`numpy`
`networkx`

<h2> Process </h2>

## STEP 1
Size of matrix is inputted from the user and then the **Adjacency matrix** **G** is constructed by inputting **1 or 0** denoting whether there is an edge between 2 nodes or not respectively.

## STEP 2
Teleportation factor is inputted in **alpha** from the user.

## STEP 3
Then we generate the transition probability matrix using the adjacency matrix and the teleportation factor by calling the function **generate_tp_matrix(alpha,G)**

We store the **order of the matrix** in **N** and construct a 2D matrix **'P'** of order **N** with 0 at all indices.

Sum of each row of **G** is stored in an array named **sum_of_1s**.

The values in P are updated by dividing each index with the sum of 1s in that row taken from sum_of_1s array.
Then the value at each index is multiplied by (1-alpha) followed by addition of (alpha/N) to each index to get the desired transition probability matrix.

1 X N matrix **x** is created where the value at first index is 1 and all other indices are 0.

## STEP 4

The **function power_itr_inf(xp,P)** is called where xp is the product of matrices **x** and **P**.
This function keeps on multiplying matrix P with matrix xp till the resultant matrix attains a constant value. 
This constant value matrix is the desired page rank vector.

<h3>To Run </h3>
Make sure the folder structure is the same as follows and the gpickle file is in the same location. If you want to you may include your own gpickle in the file.
Run the python file in the command line using the following commnad.

```
python PageRank.py
```
