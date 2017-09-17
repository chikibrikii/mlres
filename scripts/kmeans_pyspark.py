from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row
from decimal import Decimal
from re import sub
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import numpy as np


def removeQuotes(s):
	""" Remove quotation marks from an input string
	Args:
	    s (str): input string that might have the quote "" characters
	Returns:
	    str: a string without the quote characters
	"""
	return ''.join(i for i in s if i!='"')

def mk_int(s):
	s = s.strip()
	return int(s) if s else 0

def mk_double(s):
	s = s.strip()
	return float(s) if s else 0.0

def mk_decimal(s):
	s = s.strip()
	return Decimal(s) if s else 0.0


def main():
	'Start up a spark context to interface with spark. First thing you need to do always.'
	sc = SparkContext(appName="CleanData")

	'Read the file with data and transform it into an RDD. I cache it to speed the calculations later'
	data_file = sc.textFile("./all_companies_utf8.txt").cache()

	'You need that if you want to work with DataFrames or if you want to run Spark SQL queries on data'
	sqlContext = SQLContext(sc)

	'Filter out the headers line, filter out not available data and remove quotes from words.'
	csv_data = (data_file.filter(lambda l: 'Company name' not in l)
				.filter(lambda l: ('n.a.' not in l) and ('n.s.' not in l))
			      .map(lambda l: removeQuotes(l).split('\t'))
			)

	'Drop RDD from memory'
	data_file.unpersist()

	'Transform data structure into list of Row objects in order to transform them to a Spark DataFrame later.'
	row_data = (csv_data.map(lambda p: Row( company_name=p[1], 
						town=p[9],
						FTSE_sector=p[16],
						num_of_empl=mk_int( sub(r'[^\d.]', '', p[6])) or "None",
						turnover=mk_double(sub(r'[^\d.]', '', p[3])) or "None",
						profit_margin=mk_double(p[20]) or "None",
						ROSF=mk_double( p[23]) or "None",
						ROCE=mk_double( p[24]) or "None",
						gear_ratio=mk_double( p[21]) or "None",
						liquidity_ratio=mk_double( p[22]) or "None",
						credit_score=mk_int( p[26]) or "None"
						)
					)
			)		

	'Found those by hand..based on output of the .describe() function of the DataFrame class.'
	MEAN_TURN = 103382.0
	STD_TURN = 1556565.0
	MEAN_GEAR = 103.0 
	STD_GEAR = 162.0

	'Make a DataFrame, try to eliminate outliers. For this context outliers are 
	companies with turnover and gearing ration larger than 1 or 1.5 times more than the average numbers'
	df1 = (sqlContext.createDataFrame(row_data)
		.cache())
	df2 = df1.filter(df1.turnover < MEAN_TURN + 1.*STD_TURN)
	df3 = df2.filter(df2.gear_ratio < MEAN_GEAR + 1.5*STD_GEAR)

	'I drop rows with NA values'
	df = (df3.replace(0.0, 'None').dropna()
		.cache())

	'Trying to structure the feature vector column that will be the input for the Kmeans algo. Commented out lines
	are features that could be used also '
	vecAssembler = (VectorAssembler(inputCols=["turnover",
						#	"num_of_empl", 
							#"profit_margin", 
							#"ROSF",
							#"ROCE",
							#"gear_ratio",
							#"liquidity_ratio", 
							"credit_score"], 
							outputCol="unscaled_features"))

	'Subtract mean and normalize by std'
	scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withMean=True, withStd=True)
	
	'Run kmeans for 5 clusters'
	km = KMeans(k=5, seed=101)

	'form a pipeline of algorithms where you first make a feature vector column, then you scale and then you run kmeans' 
	pipeline = Pipeline(stages=[vecAssembler, scaler, km, ])

	'actually run the pipeline'
	model = pipeline.fit(df)

	'get the centres'
	centers = model.stages[2].clusterCenters()
	mean = model.stages[1].mean
	std = model.stages[1].std
	denorm_centers = []
	for c in centers:
		denorm_centers.append( (c*std)+mean )
	print("mean: {}".format(mean))
	print("std: {}".format(std))
	print(centers)
	print(denorm_centers)

	tdf = model.transform(df)

	'If you register a table you can run SQL queries then. I didnt do that but you can if you need exploratory analysis'
	tdf.registerTempTable("ClusteredFirms")
	
	tdf.printSchema()
	tdf.describe().show()
	tdf.show(5)

	'coalesce gathers all data in one node and then I use the spark-csv package in order to save the clusters as 
	a txt file'
	tdf.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('ClusteredFirms.txt')

	'Save cluster centroids in a text file'
	with open("kmeansCenters.txt", 'w') as f:
		#for v in denorm_centers:
		np.savetxt(f, denorm_centers, fmt='%10.5f', delimiter=',')
		np.savetxt(f, centers, fmt='%10.5f', delimiter=',')
		
	'Stop Spark contecxt on the server side'
	sc.stop()


if __name__ == "__main__":
	main()