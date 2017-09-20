from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark.ml.util import keyword_only

@inherit_doc
class KNNClassifier:
    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 seed=None, topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                 bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                 k=5, neighborsCol="neighbors", maxNeighbors=float("inf"), rawPredictionCol="rawPrediction",
                 probabilityCol="probability"):
                 
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.classification.KNNClassifier", self.uid)

        self.topTreeSize = Param(self, "topTreeSize", "number of points to sample for top-level tree")
        self.topTreeLeafSize = Param(self, "topTreeLeafSize",
                                     "number of points at which to switch to brute-force for top-level tree")
        self.subTreeLeafSize = Param(self, "subTreeLeafSize",
                                     "number of points at which to switch to brute-force for distributed sub-trees")
        self.bufferSize = Param(self, "bufferSize",
                                "size of buffer used to construct spill trees and top-level tree search")
        self.bufferSizeSampleSize = Param(self, "bufferSizeSampleSize",
                                          "number of sample sizes to take when estimating buffer size")
        self.balanceThreshold = Param(self, "balanceThreshold",
                                      "fraction of total points at which spill tree reverts back to metric tree if "
                                      "either child contains more points")
        self.k = Param(self, "k", "number of neighbors to find")
        self.neighborsCol = Param(self, "neighborsCol", "column names for returned neighbors")
        self.maxNeighbors = Param(self, "maxNeighbors", "maximum distance to find neighbors")

        self._setDefault(topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                         bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                         k=5, neighborsCol="neighbors", maxNeighbors=float("inf"))

        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)
