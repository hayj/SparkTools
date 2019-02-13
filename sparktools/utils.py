from systemtools.logger import *
from systemtools.duration import *
from systemtools.basics import *
from systemtools.location import *
from systemtools.system import *
from collections import defaultdict

from pyspark.ml.linalg import SparseVector, Vectors, VectorUDT

def sparseVectorAdd(v1, v2):
    """
        This function sum 2 sparse vectors
        https://stackoverflow.com/questions/32981875/how-to-add-two-sparse-vectors-in-spark-using-python
    """
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size
    values = defaultdict(float) # Dictionary with default value 0.0
    # Add values from v1
    for i in range(v1.indices.size):
        values[v1.indices[i]] += v1.values[i]
    # Add values from v2
    for i in range(v2.indices.size):
        values[v2.indices[i]] += v2.values[i]
    return Vectors.sparse(v1.size, dict(values))



def rddStreamCollect(rdd, chunksSize=1000000, logger=None, verbose=True):
    """
        This function stream a RDD collect.
        About collect in the documentation:
         > This method should only be used if the resulting array is
         > expected to be small, as all the data is loaded into the driver’s memory.
        So here we use this trick:
        https://stackoverflow.com/questions/37368635/what-is-the-best-practice-to-collect-a-large-data-set-from-spark-rdd
    """
    indexed_rows = rdd.zipWithIndex().cache()
    count = indexed_rows.count()
    start = 0
    end = start + chunksSize
    pb = ProgressBar(count, message="RDD stream collect", logger=logger, verbose=verbose)
    while start < count:
        chunk = indexed_rows.filter(lambda r: r[1] >= start and r[1] < end).collect()
        for row in chunk:
            yield row[0]
            pb.tic()
        start = end
        end = start + chunksSize