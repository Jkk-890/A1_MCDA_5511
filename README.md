# A1_MCDA_5511
To get the code to run you must

1. Run "uv sync"
2. Select the python "3.10.16" kernal
3. Run .venv\Scripts\Activate
4. Run .venv\Scripts\python.exe -m ensurepip --upgrade
5. Run .venv\Scripts\python.exe -m pip install ipykernel

After completing these steps the code should run fine

If you are on Mac you probably need to switch every '\' to a '/' in the above 
commands.

![alt text](a_visualization.png)

# What Are Embeddings? (Part_1):
In this code, we have a dataset that contains the names and short descriptions 
of the interests of students and instructors in MCDA 5511. The goal of word 
and sentence embeddings in our context is to transform these textual 
descriptions into numerical representations while embedding semantic meaning. 
This allows the program to analyze the numerical data and determine which 
people have similar interests.

A vector can be thought of as an arrow in 2-dimensional space, where the tail 
of the arrow is at the origin and the arrow itself has both magnitude and 
direction. However, in our case, we don't limit ourselves to just 2 
dimensions. Our dataset uses 384-dimensional vectors, which are impossible to 
visualize but are easy for a computer to process.

In the context of embeddings, a word is represented as a numerical vector. If 
two vectors have a similar direction or are close together in space, their
corresponding words have related meanings. For example, the words "leaf" and 
"tree" would have vectors that are close together because they both relate to 
nature. On the other hand, "leaf" and "lamp" would be farther apart because 
they are unrelated.

Sentence embeddings follow the same principle, but instead of assigning a 
vector to each individual word, an entire sentence is mapped to a single 
vector. For instance, the sentences "I went for a run" and "I enjoy running" 
would have similar vectors because they express related ideas. However, "I 
went for a run" and "My eyes hurt a lot" would have very different vectors 
since they convey unrelated concepts.
