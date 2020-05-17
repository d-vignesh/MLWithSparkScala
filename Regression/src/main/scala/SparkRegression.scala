// implementation of regression models using spark to analyse the relationship 
// b/w advertisment strategies and sales.

import java.io.File

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, round}
import org.apache.spark.sql.types.{DoubleType}

import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, DecisionTreeRegressionModel}
import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel}
import org.apache.spark.ml.regression.{GBTRegressor, GBTRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator


object SparkRegression {

	// defining a case class for the advertisment data structure
	case class Advertisment(TV: Double, Radio: Double, Newspaper: Double, Sales:Double)


	def main(args: Array[String]): Unit = {

		Logger.getLogger("org").setLevel(Level.WARN)
		Logger.getLogger("akka").setLevel(Level.OFF)
		
		val spark = SparkSession
			.builder
			.appName("Regression Models")
			.getOrCreate()
		import spark.implicits._

		val adsDataHomeDir = "/home/vignesh/my_drive/Data Science/spark/MLWithSparkScala/Regression/"

		// loading the data
		val adsData = spark.read
			.format("csv")
			.option("header", "true")
			.load(new File(adsDataHomeDir, "advertising.csv").toString())
			.withColumn("TV", col("TV").cast(DoubleType))
			.withColumn("Radio", col("Radio").cast(DoubleType))
			.withColumn("Newspaper", col("Newspaper").cast(DoubleType))
			.withColumn("Sales", col("Sales").cast(DoubleType))
			.as[Advertisment]

		// println(adsData.printSchema())
		// println(adsData.show(5))

		// convert the data to feature vector
		val cols = Array("TV", "Radio", "Sales")
		val assembler = new VectorAssembler()
			.setInputCols(cols)
			.setOutputCol("vector_features")

		val adsDataVectorized = assembler.transform(adsData)

		// normalize the data to be all features to same scale 
		val scaler = new StandardScaler()
			.setInputCol("vector_features")
			.setOutputCol("features")
			.setWithStd(true)
			.setWithStd(true)

		val scalerModel = scaler.fit(adsDataVectorized)
		val adsDataScaled = scalerModel.transform(adsDataVectorized).selectExpr("features", "Sales as label")

		// println(adsDataScaled.show(20, false))

		// Split the data into training, validation and test sets 
		val Array(trainingData, validationData, testData) = adsDataScaled.randomSplit(Array(0.6, 0.2, 0.2))
		
		// println(adsData.count())
		// println("training : " + trainingData.count() + " validation : " + validationData.count() + " test : " + testData.count())


		val evaluator = new RegressionEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("rmse")
		val numIterations = Array(10, 30)


		// ***************************  Linear Regression  ************************************
	
		val regParams = Array(0.003, 0.1, 0.3, 1.0) // param to configure regularization effect.
		var bestLR_Iteration: Int = 0
		var bestRegParam: Double = 0.0
		var bestLRModel: LinearRegressionModel = null
		var bestLR_RMSE: Double = Double.MaxValue

		println("***************************  Linear Regression  ************************************")
		for (iteration <- numIterations; regParam <- regParams) {
			val lr = new LinearRegression()
				.setRegParam(regParam)
				.setMaxIter(iteration)

			// fit the model on training data 
			val model = lr.fit(trainingData)

			// evaluate the model on the validation data
			val predictions = model.transform(validationData)

			var rmse = evaluator.evaluate(predictions)

			if (rmse < bestLR_RMSE) {
				bestLR_RMSE = rmse
				bestRegParam = regParam
				bestLR_Iteration = iteration
				bestLRModel = model
			}
			println(s"for iteration : " + iteration + ", RegParam : " + regParam + " RMSE on validation data : " + "%1.4f".format(rmse))
		}

		println(s"the best model has iteration : " + bestLR_Iteration + ", RegParam : " + bestRegParam)
		println(s"coefficients: ${bestLRModel.coefficients} Intercept: ${bestLRModel.intercept}")
		
		// make predictions on the test data
		val predictions = bestLRModel.transform(testData)
		val lr_rmse = evaluator.evaluate(predictions)
		println("the rmse of best Linear regression model on test data : " + "%1.4f".format(lr_rmse))

		// ******************************** DecisionTree Regressor ******************************
		
		val maxDepths = Array(3, 5, 7)
		val minNodesPerInstance = Array(5, 10, 15)
		var bestMaxDepthDT: Int = 0
		var bestMinNodeDT: Int = 0
		var bestDTModel: DecisionTreeRegressionModel = null
		var bestDT_RMSE: Double = Double.MaxValue

		println("******************************** DecisionTree Regressor ******************************")
		for(maxDepth <- maxDepths; minNode <- minNodesPerInstance) {
			val dt = new DecisionTreeRegressor()
				.setMaxDepth(maxDepth)
				.setMinInstancesPerNode(minNode)

			val model = dt.fit(trainingData)

			val predictions = model.transform(validationData)

			var rmse = evaluator.evaluate(predictions)

			if (rmse < bestDT_RMSE) {
				bestDT_RMSE = rmse
				bestMaxDepthDT = maxDepth
				bestMinNodeDT = minNode
				bestDTModel = model
			}
			println("model with maxDepth : " + maxDepth + " minNodesPerInstance : " + minNode + " gives RMSE : " + "%1.4f".format(rmse))
		}

		println("best model has maxDepth : " + bestMaxDepthDT + ", minNodesPerInstance : " + bestMinNodeDT)
		println(bestDTModel.toDebugString)

		val predictions = bestDTModel.transform(testData)
		val dt_rmse = evaluator.evaluate(predictions)
		println("the rmse of best Decision Regressor Model on test data : " + "%1.4f".format(dt_rmse))  

		// ************************************ Random Forest Regressor ****************************
		val numTreesInForest = Array(5, 10, 15)
		var bestMaxDepthRT: Int = 0
		var bestMinNodeRT: Int = 0
		var bestNumTrees: Int = 0
		var bestRTModel: RandomForestRegressionModel = null
		var bestRT_RMSE: Double = Double.MaxValue

		println("************************************ Random Forest Regressor ****************************")
		for ( maxDepth <- maxDepths; minNode <- minNodesPerInstance; numTrees <- numTreesInForest) {
			val rt = new RandomForestRegressor()
				.setMaxDepth(maxDepth)
				.setMinInstancesPerNode(minNode)
				.setNumTrees(numTrees)

			val model = rt.fit(trainingData)

			val predictions = model.transform(validationData)

			var rmse = evaluator.evaluate(predictions)

			if (rmse < bestRT_RMSE) {
				bestRT_RMSE = rmse
				bestMaxDepthRT = maxDepth
				bestMinNodeRT = minNode
				bestNumTrees = numTrees
				bestRTModel = model
			}

			println("model with maxDepth : " + maxDepth + ", minNodesPerInstance : " + minNode + ", numTrees : " + numTrees + " gives RMSE : " + "%1.4f".format(rmse))
		}

		println("best model has maxDepth : " + bestMaxDepthRT + ", minNodesPerInstance : " + bestMinNodeRT + " bestNumTrees : " + bestNumTrees)

		val predictions = bestRTModel.transform(testData)
		val rt_rmse = evaluator.evaluate(predictions)
		println("the rmse of best Random Forest Model on test data : " + "%1.4f".format(rt_rmse))

		// ******************************************* Gradient Boosted Tree ********************************
		var bestMaxDepthGBT: Int = 0
		var bestMinNodeGBT: Int = 0
		var bestIteration_GBT: Int = 0
		var bestGBTModel: GBTRegressionModel = null
		var bestGBT_RMSE: Double = Double.MaxValue

		println("******************************************* Gradient Boosted Tree ********************************")
		for( iteration <- numIterations; maxDepth <- maxDepths; minNode <- minNodesPerInstance) {
			val gbt = new GBTRegressor()
				.setMaxDepth(maxDepth)
				.setMinInstancesPerNode(minNode)
				.setMaxIter(iteration)

			val model = gbt.fit(trainingData)

			val predictions = model.transform(validationData)

			val rmse = evaluator.evaluate(predictions)

			if (rmse < bestGBT_RMSE) {
				bestGBT_RMSE = rmse
				bestMaxDepthGBT = maxDepth
				bestMinNodeGBT = minNode
				bestIteration_GBT = iteration
				bestGBTModel = model
			}

			println("model with iteration : " + iteration + " maxDepth : " + maxDepth + ", minNodesPerInstance : " + minNode + " gives RMSE : " + "%1.4f".format(rmse))
		}

		println("best model has maxDepth : " + bestMaxDepthGBT + ", minNodesPerInstance : " + bestMinNodeGBT + ", MaxIterations: " + bestIteration_GBT)

		val predictions = bestGBTModel.transform(testData)
		val rmse_gbt = evaluator.evaluate(predictions)
		println("rmse of best GBT Model on test data : " + "%1.4f".format(rmse_gbt))
	}
}