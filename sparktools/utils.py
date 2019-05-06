from systemtools.logger import *
from systemtools.duration import *
from systemtools.basics import *
from systemtools.location import *
from systemtools.system import *
from collections import defaultdict
from pyspark.ml.linalg import SparseVector, Vectors, VectorUDT
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import HashingTF, IDF, CountVectorizer
from pyspark.sql import Row
from collections import OrderedDict
import pyspark

def dict2SparseVector(theDict):
    return SparseVector(theDict["size"], theDict["indices"], theDict["values"])

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

def dfStreamCollect(df, groupBy=None, chunksSize=100000, logger=None, verbose=True, defaultRandomGroupCol="randomGroupCol"):
    if groupBy is None:
        from pyspark.sql.functions import rand
        from pyspark.sql.functions import udf
        from pyspark.sql.types import LongType
        log("Starting to generate a random column to split the df", logger=logger, verbose=verbose)
        groupBy = defaultRandomGroupCol
        def normalizeRandomCol(randomFloat):
            return int(chunksSize * randomFloat)
        df = df.withColumn(groupBy, rand())
        normalizeRandomColUDF = udf(normalizeRandomCol, LongType())
        df = df.withColumn(groupBy, normalizeRandomColUDF(df[groupBy]))
    log("Starting the groupby count", logger=logger, verbose=verbose)
    dfCount = df.groupBy(groupBy).count()
    filters = [[]]
    currentCount = 0
    groupByNumberOfItems = 0
    log("Starting to collect filters", logger=logger, verbose=verbose)
    for row in dfCount.collect():
        groupByNumberOfItems += 1
        filters[-1].append(row[groupBy])
        currentCount += row["count"]
        if currentCount >= chunksSize:
            currentCount = 0
            filters.append([])
    if len(filters[-1]) == 0:
        filters = filters[:-1]
    assert groupByNumberOfItems == len(flattenLists(filters))
    log("Starting to yield rows", logger=logger, verbose=verbose)
    for i in pb(range(len(filters)), len(filters), message="Collecting chunks", logger=logger, verbose=verbose):
        filter = filters[i]
        for row in df.where(col(groupBy).isin(filter)).collect():
            row = row.asDict()
            if defaultRandomGroupCol in row:
                del row[defaultRandomGroupCol]
            yield row


def dfChunkedSave(df, saveDir, groupBy=None, chunksSize=1000000, saveCallback=None, logger=None, verbose=True, cleanDirPath=True):
    def __saveCallback(tmpDF, dirPath):
        tmpDF.write.format("json").option("compression", "bzip2").mode("overwrite").save(dirPath)
    if saveCallback is None:
        saveCallback = __saveCallback
    if groupBy is None:
        groupBy = df.columns[0]
        logWarning("We will use " + groupBy + " as the group by column for `dfChunkedSave`.",
            logger=logger, verbose=verbose)
    if cleanDirPath:
        for current in sortedGlob(saveDir + "/*"):
            removeDirSecure(current, slashCount=4)
    dfCount = df.groupBy(groupBy).count()
    filters = [[]]
    currentCount = 0
    groupByNumberOfItems = 0
    for row in dfCount.collect():
        groupByNumberOfItems += 1
        filters[-1].append(row[groupBy])
        currentCount += row["count"]
        if currentCount >= chunksSize:
            currentCount = 0
            filters.append([])
    if len(filters[-1]) == 0:
        filters = filters[:-1]
    assert groupByNumberOfItems == len(flattenLists(filters))
    for i in pb(range(len(filters)), message="Storing chunks", logger=logger, verbose=verbose):
        dirPath = saveDir + "/" + str(i)
        mkdir(dirPath)
        filter = filters[i]
        tmpDF = df.where(col(groupBy).isin(filter))
        saveCallback(tmpDF, dirPath)


def groupAndSumSparseVectors(df, groupByColName="authorialDomain", targetColName="tf"):
    """
        This function take a dataframe, group on groupby column and sum SparseVectors in the target column
    """
    reduce = df.rdd.reduceByKey(sparseVectorAdd).toDF()
    reduce = reduce.withColumnRenamed("_1", groupByColName)
    reduce = reduce.withColumnRenamed("_2", targetColName)
    return reduce



def groupAndSumSparseVectors_old(df, groupByColName, targetColName):
    """
        This function take a dataframe, group on groupby column and sum SparseVectors in the target column
    """
    print("DEPRECATED groupAndSumSparseVectors_old")
    exit()
    goupedDF = df.groupBy(groupByColName).agg(collect_list(targetColName).alias(targetColName))
    sparseVectorSumUDF = udf(sparseVectorSummerizer, VectorUDT())
    goupedDF = goupedDF.withColumn(targetColName, sparseVectorSumUDF(goupedDF[targetColName]))
    return goupedDF



def computeTFIDF(df, inputColName="tf", outputColName="tfidf", logger=None, verbose=True, removeInputCol=True):
    """
        This add a tfidf column convert a dataframe to 
        A minDF of 2 will remove term that appear once in the whole corpus
    """
    idf = IDF(inputCol=inputColName, outputCol=outputColName)
    idfModel = idf.fit(df)
    tfidfDF = idfModel.transform(df)
    # We drop the tf column:
    if removeInputCol:
        try:
            tfidfDF = tfidfDF.drop(inputColName)
        except Exception as e:
            logException(e, logger, verbose=verbose)
    return tfidfDF

def getDistincts(df, col):
    return set(df.select(col).distinct().rdd.map(lambda r: r[0]).collect())

def dictToRow(d):
    return Row(**OrderedDict(sorted(d.items())))

def dictListToDataFrame(l, spark):
    return spark.sparkContext.parallelize(l).map(dictToRow).toDF()

def mergeDataframes(a, b):
    return pyspark.sql.DataFrame.unionAll(a, b)

