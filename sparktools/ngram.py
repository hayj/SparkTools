from systemtools.logger import *
from systemtools.basics import *
from pyspark.ml.feature import NGram
from sparktools.utils import *


def toNgramDF(df, nbGrams, inputColName, addNbGramsToOutputCol=False,
    removeInputCol=True, logger=None, verbose=True):
    """
        This function convert a dataframe to a ngramDF on the given inputColName
    """
    if addNbGramsToOutputCol: 
        columnName = str(nbGrams) + "grams"
    else:
        columnName = "ngrams"
    ngram = NGram(n=nbGrams, inputCol=inputColName, outputCol=columnName)
    ngramDF = ngram.transform(df)
    # We drop the inputCol column:
    if removeInputCol:
        try:
            ngramDF = ngramDF.drop(inputColName)
        except Exception as e:
            logException(e, logger, verbose=verbose)
    return ngramDF