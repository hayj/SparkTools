
# nn -o ngram-tree-nohup.out pew in st-venv python ~/Workspace/Python/Utils/SparkTools/sparktools/ngram.py

from systemtools.logger import *
from systemtools.basics import *
if weAreBefore("21/02/2019"):
    print("WARNING we find spark in ngram.py " * 100)
    import findspark
    findspark.init(homeDir() + "/lib/spark-2.4.0-bin-hadoop2.7")
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


class NGramTree:
    def __init__(self, logger=None, verbose=True):
        self.logger = logger
        self.verbose = verbose
        self.vocSize = 0

    def load(self, path):
        self.tree = deserialize(path)

    def save(self, path=None):
        if self.vocSize != 0:
            if path is None:
                path = tmpDir() + "/ngramtree-" + str(self.vocSize) + ".pickle"
            serialize(self.tree, path)
            return path
        return None

    def generateFromVoc(self, files):
        if not isinstance(files, list):
            files = [files]
        self.tree = dict()
        self.vocSize = 0
        currentIndex = 0
        pbVerbose = len(files) > 3
        for filePath in pb(files, logger=self.logger, verbose=pbVerbose):
            currentVoc = deserialize(filePath)
            for ngram in currentVoc:
                ngram = ngram.split(" ")
                currentTree = self.tree
                for word in ngram[:-1]:
                    if word not in currentTree:
                        currentTree[word] = dict()
                    currentTree = currentTree[word]
                lastWord = ngram[-1]
                currentTree[lastWord] = currentIndex
                currentIndex += 1
                self.vocSize += 1

    def getIndex(self, ngram):
        words = ngram.split(" ")
        currentTree = self.tree
        for word in words:
            currentTree = currentTree[word]
        return currentTree

    def printSample(self, count=5):
        for i in range(count):
            currentTree = self.tree
            ngram = ""
            while not isinstance(currentTree, int):
                key = random.choice(list(currentTree.keys()))
                ngram += key + " "
                currentTree = currentTree[key]
            log(ngram + "--> " + str(currentTree), self)
            log("\n", self)

def test1():
    voc = ["a b c", "b c a", "b b a", "a c b", "a a a"]
    path = tmpDir() + "/voc-test-ngram-tree.pickle"
    serialize(voc, path)
    ngtree = NGramTree()
    ngtree.generateFromVoc(path)
    print(ngtree.getIndex(voc[0]))
    print(ngtree.getIndex(voc[2]))
    print(ngtree.getIndex(voc[3]))
    ngtree.printSample()

def test2():
    path = tmpDir("listchunker") + "/HEM7F3OVRI-15500777898835018"
    files = sortedGlob(path + "/*.pickle", sortBy=GlobSortEnum.NUMERICAL_NAME)
    ngtree = NGramTree()
    ngtree.generateFromVoc(files)
    ngtree.save()
    ngtree.printSample()

def test3():
    logger = Logger("ngram-tree-test.log")
    path = "/home/student/Data/Asa/3grams-test-data/voc"
    files = sortedGlob(path + "/*.pickle", sortBy=GlobSortEnum.NUMERICAL_NAME)
    log(reducedLTS(files), logger)
    ngtree = NGramTree(logger=logger)
    ngtree.generateFromVoc(files)
    ngtree.save()
    ngtree.printSample()

if __name__ == "__main__":
    test3()