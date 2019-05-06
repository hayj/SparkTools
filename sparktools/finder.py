def findSpark(logger=None, verbose=True):
	from systemtools.logger import log
	from systemtools.location import sortedGlob, homeDir
	import findspark
	sparkPath = sortedGlob(homeDir() + "/lib/spark-*2*")[-1]
	log("Spark path: " + str(sparkPath), logger)
	findspark.init(sparkPath)