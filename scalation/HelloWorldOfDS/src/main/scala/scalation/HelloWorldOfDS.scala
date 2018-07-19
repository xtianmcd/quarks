//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Christian McDaniel
 *  @version 1.0
 *  @date    Feb 5, 2018
 *  @see     LICENSE (MIT style license file)
 */

package scalation 

//import scala.collection.immutable.ListMap
//import scala.math

//import scalation.plot.Plot
//import scalation.random.CDF.studentTCDF
//import scalation.util.{banner, Error, time}
//import scalation.util.Unicode.sub
//import scalation.stat.StatVector.corr
//import scalation.util.getFromURL_File
//import scalation.util.Error

import scalation.analytics.Regression

import scalation.analytics.classifier.NaiveBayes
import scalation.columnar_db.Relation
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream
import scalation.linalgebra._
import scala.math.sqrt
import scala.collection.mutable.Set

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The 'HelloWorldOfDS' object uses the MatrixD and Regression classes
* to perform multiple regression and subsequent analysis on a numerical dataset, read in as an
* argument at the time of running the program.
*  > sbt "run <filepath>"
*/
object HelloWorldOfDS extends App
{
    private val BASE_DIR = "/Applications/scalation_1.4/data/analytics/"
    //private val WINE_URL = "https://github.com/scalation/scalation/blob/master/data/analytics/classifier/winequality-white.csv"


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepares csv file for feeding into Multiple Linear Regression model.
    *   @param fp     filepath
    *   @param skip   number of initial rows to skip (re: e.g., col names)
    *   @param yCol   column number for target values (y)
    */
    def prepReg (fp: String, skip: Int, yCol: Int) =
    {
      //val mlrData = getFromURL_File(WINE_URL)
      val df    = MatrixD(fp, skip)
      val df_x  = df.sliceExclude(df.dim1, yCol)     // X (data) matrix
      val df_y  = df.col(yCol)                       // y (target) vector
      val df_i  = VectorD.one(df.dim1)               // intercept
      val df_x1 = df_x.+^:(df_i)                     // prepend intercept to X
      (df_x1, df_y)
    } // prepReg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepares arff file for feeding into Naive Bayes (or other MatriI-
    *   accommodating model)
    *   @param fp   filepath
    */
    def prepBayes (fp: String) =
    {
      var data  = Relation            (fp, -1, null)
      val xy    = data.toMatriI2      (null)
      val x     = xy                  (0 until xy.dim1, 0 until xy.dim2 - 1)
      val y     = xy.col              (xy.dim2 - 1)
      val fn    = data.colName.slice  (0, xy.dim2 - 1).toArray
      val cn    = Array               ("p", "e")
      val k = 2
      (x, y, fn, cn, k)
    } // prepBayes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Instantiates, trains, and prints results for Multiple Linear Regression
    *   and subsequently tests for collinearity among predictors via VIF
    *   calculation.
    *   @param x  training data
    *   @param y  target values
    */
    def runReg (x: MatrixD, y: VectorD, dataset: String) =
    {
      val rg = new Regression(x, y)
      rg.train ().eval(y)
      println  ()
      println  ("|====================|")
      println (s"|  $dataset DATASET  |")
      println (s"|    MULT LIN REG    |")
      println  ("|====================|")
      println  ()
      rg.report()
      println  ("VIF: \n")
      println  (rg.vif)
      println  ()
    } // runReg

    def prunePreds (x: MatrixD, y: VectorD) =
    {
      val rg = new Regression (x, y)
      // source for fwd selection and backward elim: RegressionTest4 in Regression.scala 
      println ("Forward Selection Test")
      val fcols = Set (0)
      for (l <- 1 until x.dim2) {
        val (x_j, b_j, fit_j) = rg.forwardSel (fcols)        // add most predictive variable
        println (s"forward model: add x_j = $x_j with b = $b_j \n fit = $fit_j")
        fcols += x_j
      } // for

      println ("Backward Elimination Test")
      val bcols = Set (0) ++ Array.range (1, x.dim2)
      for (l <- 1 until x.dim2) {
        val (x_j, b_j, fit_j) = rg.backwardElim (bcols)     // eliminate least predictive variable
        println (s"backward model: remove x_j = $x_j with b = $b_j \n fit = $fit_j")
        bcols -= x_j
      } // for
    } // prunePreds

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute and return accuracy, recall, precision and F-Score from the
    *	provided statistics of the current fold, which are provided by the
    *	checkClass() method below.
    *	@param tp  the true  positives from current classifier during current fold
    *	@param tn  the true  negatives from current classifier during current fold
    *	@param fp  the false positives from current classifier during current fold
    *	@param fn  the false negatives from current classifier during current fold
    */
    def getStats (tp :Double, tn :Double, fp :Double, fn :Double): (Double, Double, Double, Double) =
    {
      var foldAcc    = (tp + tn) / (tp + tn + fp + fn)
      var foldRecall = 0.0
      var foldPrecis = 0.0
      var foldFScore = 0.0

      if (tp == 0.0) {											// set recall, precision and FScore
          if ((fp == 0.0) & (fn == 0.0)) {	// to predesignated value if tp == 0
              foldRecall = 1.0
              foldPrecis = 1.0
              foldFScore = 1.0
          } else {
              foldRecall = 0.0
              foldPrecis = 0.0
              foldFScore = 0.0
          }
      } else {
        foldRecall = tp / (tp + fn)
        foldPrecis = tp / (tp + fp)
        foldFScore = 2 * foldPrecis * foldRecall / (foldPrecis + foldRecall)
      } // else tp not 0

      (foldAcc, foldRecall, foldPrecis, foldFScore)
    } // getStats

  	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  	/** Determine the confusion matrix parameters, namely tp, tn, fp, fn, for
  	*	each example within the given fold's validation set for the current
  	*	classifier. Feed these parameters to getStats() above and return the
  	*	results.
  	*	@param predictArray  the predicted values for each validation example
  	*	@param actualArray   the ground truth label for each validation example
  	*	@param show          if true, prints some informative messages;
  	*						 false by default
  	*/
    def checkClass (predictArray: VectorD, actualArray: VectorD, show: Boolean = false): (Double, Double, Double, Double) =
    {
      if (show) {
        println(s"predictions: $predictArray")
        println(s"class: $actualArray")
      } // if show

      var fp = 0.0
      var fn = 0.0
      var tp = 0.0
      var tn = 0.0

      for (pred <- 0 until predictArray.dim)  {

        if (predictArray(pred) == 0.0) {
          if (actualArray(pred) == 0)      tn += 1.0
          else if (actualArray(pred) == 1) fn += 1.0
        } // if predict neg

        else if (predictArray(pred) == 1.0) {
          if (actualArray(pred) == 1) tp += 1.0
          else if (actualArray(pred) == 0) fp += 1.0
        } // if predict pos

      } // for

      if (show) {
        println(s"tp: $tp, fp: $fp, tn: $tn, fn: $fn")
      } // if show

      var (acc, recall, precision, fScore) = getStats(tp, tn, fp, fn)	// call the getStats() method above
      (acc, recall, precision, fScore)
    } // checkClass


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The main method for the `ClassifierTests` class. Performs six different
    *	classifiers on the provided data via k-fold cross validation. During
    *	each fold, feeds the classifications for each classifier to the method
    *	checkClass() above to calculate performance measures, and then
    *	collects these metrics for mean and standard deviation calculations.
    *	Prints the mean and standard deviation for each classifier to the
    *	console for review.
    *	@param dataset    short name for identifying the dataset during output
    *	@param datasetx   the unlabeled training data
    *	@param datasety   the ground trught labels for the training data
    *	@param datasetfn  the feature names for the dataset
    *	@param datasetcn  the class names for the dataset
    *	@param nx 		    the number of folds for k-fold cross validation
    *	@param show		    if true, prints some informative messages;
  	*		                false by default
    */
    def crossValidateAlgos (dataset :String, datasetx :MatriI, datasety :VectoI, datasetfn :Array[String], datasetcn :Array[String], k :Int, nx :Int, show: Boolean = false)
    {

      val vc          = (for(j <- datasetx.range2) yield datasetx.col(j).max() + 1).toArray		// calculate the value counts for each feature
      if (show) {for (i <- vc) println(s"value count at index $i: $vc(i)")}

      val permutedVec = PermutedVecI     (VectorI.range (0, datasetx.dim1), ranStream)
      val randOrder   = permutedVec.igen                       									// randomize integers 0 until size
      val itestA      = randOrder.split  (nx)                   									// make array of itest indices

      if (show) {for (i <- itestA) println(s"randomly generated list of indices for the fold-wise testing: $i")}

      var accNB       = new VectorD (0) // declare the containers for fold-specific performance metrics
      var recallNB    = new VectorD (0)
      var precisNB    = new VectorD (0)
      var fScoreNB    = new VectorD (0)

      for (it <- 0 until nx) {                													// for loop for cross validation
        var classesNB  = new VectorD (0)			// declare the containers for the fold-specific classifications

        val itest      = itestA(it)() 															// get array from it element
        if (show) {println(s"randomly generated validation set for current fold: $itest")}
        var rowsList   = Array[Int]() 															// the training row indices to be kept for training
        for (rowi <- 0 until datasetx.dim1) {
          if (!itest.contains(rowi)) {rowsList = rowsList :+ rowi} 							// build the fold-specific training set row indices
        } // for
        if (show) println(s"rows to include for training: length = $rowsList.length")

        val foldx  = datasetx.selectRows(rowsList)		
        val foldy  = datasety.select(rowsList)  										// build the fold-specific training set
        if (show) println(s"initial dataset has $datasetx.dim1 rows; fold training set has $foldx.dim1 rows")
        if (show) {println(s"MatrixI training set: $foldx")}

        var nb  = new NaiveBayes         (foldx,  foldy, datasetfn, k, datasetcn,    vc)		// providing only the training data/labels

        nb.train  ()

        var rowy :Int = 10000																	// initialize ground truth label to high number for debugging
        var yArray    = new VectorD (0)															// initialize container for ground truth labels (classifier never sees these)


        for (ic <- itest) {																		// for loop for classifying each example in the validation set

          var rowx = datasetx (ic)															// validation example
          rowy     = datasety (ic)															// ground truth label

          if (show) println(s"row: $rowx")

          var (iclassNB,  icnNB,  iprobNB)  = nb.classify  ( rowx )     // classify example for each classifier

          yArray     = yArray     ++ rowy														// append label/prediction to containers

          classesNB  = classesNB  ++ iclassNB

        } // for classify

        // get fold-specific performance measures for each classifier
        var (foldAccNB,  foldRecallNB,  foldPrecisionNB,  foldFScoreNB)  = checkClass (classesNB,  yArray, show)

        accNB     = accNB     ++ foldAccNB
        recallNB  = recallNB  ++ foldRecallNB
        precisNB  = precisNB  ++ foldPrecisionNB
        fScoreNB  = fScoreNB  ++ foldFScoreNB

      } // for cv

      // print the output (mean and std dev performance for each metric for each classifier)

      val formatHeader = "%10s %10s %10s %10s %10s\n"
      val formatMain   = "%-10s %10.9f %10.9f %10.9f %10.9f\n"
      val methodNB     = "NaiveBayes"

      println ()
      println ("|====================|")
      println(s"|  $dataset DATASET  |")
      println(s"|   $nx  FOLDS  CV    |")
      println(s"|    NAIVE  BAYES    |")
      println ("|====================|")
      println ()

      println(">meanCV")
      printf(formatHeader, "", "accuracy", "recall", "precision", "f-score")
      printf(formatMain, methodNB,  accNB.mean,  recallNB.mean,  precisNB.mean,  fScoreNB.mean)
      println("==========================================================")
      println()
      println(">stdCV")
      printf(formatHeader, "", "accuracy", "recall", "precision", "f-score")
      printf(formatMain, methodNB,  sqrt(accNB.variance),  sqrt(recallNB.variance),  sqrt(precisNB.variance),  sqrt(fScoreNB.variance))
      println("==========================================================")
      println()
    } // crossValidateRand


    val wineData = BASE_DIR + "winequality-white.csv"
    val bcData   = BASE_DIR + "classifier/breast-cancer.arff"

    val (x_mlr, y_mlr) = prepReg(wineData, 0, 11)            // prep data for mlr
    runReg(x_mlr, y_mlr, "WineQual")                                  // run initial regression

    // perform a transformation on "Alcohol content" predictor and rerun regression
    var col_bef   = x_mlr.col(11)
    var col_trans = col_bef.~^(2)
    //println("column before transformation: " + col_bef)
    //println("column after  transformation: " + col_trans)
    
    x_mlr.setCol(11, col_trans)         // substitute transformed col
    // println(x_mlr)
    runReg(x_mlr, y_mlr, "Wine-tf ")

    prunePreds(x_mlr, y_mlr)

    // drop cols suspected of not adding much information; rerun regression
    val df_dropped = x_mlr.sliceExclude(x_mlr.dim1, 3)//.sliceExclude(x_mlr.dim1, 5)//.sliceExclude(x_mlr.dim1, 7)
    runReg(df_dropped, y_mlr, "WineDrop")

    // Naive Bayes 
    val (x_bc, y_bc, fn_bc, cn_bc, k_bc) = prepBayes(bcData)

    // call the methods to run the classifiers with the given data for given folds:
    //10-fold CV
    crossValidateAlgos (" CANCER ", x_bc,  y_bc,  fn_bc,  cn_bc, k_bc,  10)

    //20-fold CV
    crossValidateAlgos (" CANCER ", x_bc,  y_bc,  fn_bc,  cn_bc, k_bc,  20)

}
