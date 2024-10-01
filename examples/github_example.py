import numpy
from copkmeans.cop_kmeans import cop_kmeans
input_matrix = numpy.random.rand(100, 500)
must_link = [(0, 10), (0, 20), (0, 30)]
cannot_link = [(1, 10), (2, 10), (3, 10)]
clusters, centers = cop_kmeans(dataset=input_matrix, k=5, ml=must_link,cl=cannot_link)
