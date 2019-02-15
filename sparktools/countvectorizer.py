
from systemtools.logger import *
from systemtools.duration import *
from systemtools.file import *
from systemtools.basics import *
from systemtools.location import *
from systemtools.system import *
from datastructuretools.basics import ListChunker
from collections import defaultdict
from sparktools.utils import *
from pyspark.sql.functions import lit, rand, randn, col, udf, desc
from pyspark.sql.types import *
from pyspark.ml.feature import HashingTF, IDF, CountVectorizer



def addTermFrequencies(df, vocDir, inputCol="ngrams", targetCol="tf",
                       minDF=2, chunksSize=1000000,
                       logger=None, verbose=True, removeInputCol=False, pruneVoc=False):
    """
        The purpose of this function is to replace CountVectorizer which throw either:
         * Remote RPC client disassociated. Likely due to containers exceeding thresholds
         * Java heap space
        in cases the vocabulary is large (~1 000 000 000 for a Spark cluster of 30 nodes)
        or your amount of available RAM is low.
        For example when you want to use ngrams >= 2 as the vocabulary.
        The default CountVectorizer will share the vocabulary across nodes.
        Instead, this function will first split the voc and sum all frequencies for each
        vocabulary chunks.
        This function take a datframe, will add a "tf" column. You also have to give a directory where
        the vocabulary will be stored in multiple files (0.pickle, 1.pickle...).
        pruneVoc is very facultative and not mandatory in most cases.
    """
    # First we delete the vocabulary which already exists in the vocDir:
    for current in sortedGlob(vocDir + "/*.pickle"):
        removeFile(current)
    # We flat all the voc and remove duplicates:
    log("Starting getting the vocabulary", logger=logger, verbose=verbose)
    tt = TicToc(logger=logger, verbose=verbose)
    tt.tic(display=False)
    vocRDD = df.select(inputCol).rdd.flatMap(lambda x: list(set(x[0])))
    # We add a count to each row (each row is a ngram):
    vocRDD = vocRDD.map(lambda x: (x, 1))
    # Now we count document frequencies for each term:
    vocRDD = vocRDD.reduceByKey(lambda v1, v2: v1 + v2)
    # We keep only voc element which is >= minDF:
    whiteVocRDD = vocRDD.filter(lambda o: o[1] >= minDF)
    if pruneVoc:
        blackVocRDD = vocRDD.filter(lambda o: o[1] < minDF)
    # We collect and chunk the voc to do not share the entire voc across Spark nodes:
    if chunksSize is None:
        whiteVocChunks = [list(whiteVocRDD.keys().collect())]
        if pruneVoc:
            blackVocChunks = [list(blackVocRDD.keys().collect())]
        whiteVocSize = len(whiteVocChunks[0])
        if pruneVoc:
            blackVocSize = len(blackVocChunks[0])
    else:
        # ListChunker will serialize in batchs (chunks) to do not need to persist the whole content in memory
        # We use rddStreamCollect because the `collect` method of Dataframe load the entire voc in memory
        whiteVocChunks = ListChunker(chunksSize, rddStreamCollect(whiteVocRDD.keys(), chunksSize, logger=logger, verbose=verbose), logger=logger, verbose=verbose)
        if pruneVoc:
            blackVocChunks = ListChunker(chunksSize, rddStreamCollect(blackVocRDD.keys(), chunksSize, logger=logger, verbose=verbose), logger=logger, verbose=verbose)
        whiteVocSize = whiteVocChunks.getTotalSize()
        if pruneVoc:
            blackVocSize = blackVocChunks.getTotalSize()
    # We delete all ngrams which are not in collectedVoc:
    if pruneVoc:
        for blackVocChunk in pb(blackVocChunks, message="Prunning vocabulary black list", logger=logger, verbose=verbose):
            blackVocChunk = set(blackVocChunk)
            theUdf = udf(lambda ngrams: [token for token in ngrams if token not in blackVocChunk], ArrayType(StringType()))
            df = df.withColumn(inputCol, theUdf(df[inputCol]))
    # We fill the tf column with zeros:
    theUdf = udf(lambda: SparseVector(whiteVocSize, {}), VectorUDT())
    df = df.withColumn(targetCol, theUdf())
    # We define the udf function:
    def __sumTF(ngrams, vector, vocChunkDict, startIndex=0):
        """
            This function take ngrams and a vector, it will add frequencies of these ngrams in
            the vector at the right index according to the dictionnary of index vocChunkDict
            and the startIndex.
        """
        # We create a default dict of zero integers
        values = defaultdict(int) # from collections import defaultdict
        # For each ngram:
        for ngram in ngrams:
            # We check if the ngram exist in the voc:
            if ngram in vocChunkDict:
                # We find the right index from the entire vocabulary (not only this chunk, so we add startIndex):
                index = vocChunkDict[ngram] + startIndex
                # We add 1 frequency:
                values[index] += 1
        # We sum with the previous vector:
        vector = sparseVectorAdd(vector, SparseVector(vector.size, dict(values)))
        # We return the final vector:
        return vector
    # We create the start index of each chunk:
    startIndex = 0
    # For each white chunk (we use `pb` to see a progress bar):
    for whiteVocChunk in pb(whiteVocChunks, message="Summing term frequencies", logger=logger, verbose=verbose):
        # We construct the voc as a dict to have access to indexes in O(1):
        whiteVocChunkDict = dict()
        i = 0
        for current in whiteVocChunk:
            whiteVocChunkDict[current] = i
            i += 1
        # We create the udf and give whiteVocChunkDict and startIndex
        theUDF = udf(lambda col1, col2: __sumTF(col1, col2, whiteVocChunkDict, startIndex), VectorUDT())
        # We add all frequencies for the current voc chunk:
        df = df.withColumn(targetCol, theUDF(df[inputCol], df[targetCol]))
        # Here we force spark to execute the withColumn, instead it works lazy and
        # receive a lot of withColumn stage because of the `for` loop and crash:
        df.count()
        # And we continue to the next chunk:
        startIndex += len(whiteVocChunk)
    # We drop the ngrams columns:
    if removeInputCol:
        df = df.drop(inputCol)
    # We store and reset list chunkers:
    mkdir(vocDir)
    if isinstance(whiteVocChunks, ListChunker):
        if pruneVoc:
            blackVocChunks.reset()
        whiteVocChunks.copyFiles(vocDir)
        log("Voc size: " + str(whiteVocChunks.getTotalSize()), logger, verbose=verbose)
        whiteVocChunks.reset()
    else:
        serialize(whiteVocChunks[0], vocDir + "/0.pickle")
        log("Voc size: " + str(len(whiteVocChunks[0])), logger, verbose=verbose)
    # We log the end:
    tt.toc("We generated the voc and added term frequencies to the DF.")
    # We return all data:
    return df







def toNgramsFrequency_old(df, inputColName="ngrams", targetColName="tf",
                          minDF=2, vocabSize=2000000000,
                         removeInputCol=False):
    """
        This replace ngrams column by a CountVectorizer column sum n sparse vectors
    """
    cv = CountVectorizer(inputCol=inputColName, outputCol=targetColName, minDF=minDF, vocabSize=vocabSize)
    cvModel = cv.fit(df)
    voc = cvModel.vocabulary
    tfDF = cvModel.transform(df)
    # We drop the ngrams columns:
    if removeInputCol:
        try:
            tfDF = tfDF.drop(inputColName)
        except Exception as e:
            logException(e, logger)
    return (tfDF, voc)