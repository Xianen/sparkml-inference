package sparkml.inference;

import org.apache.spark.SparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static org.junit.Assert.assertTrue;


public class SparkMLModelLoaderTest {
    private final static Logger logger = LoggerFactory.getLogger(SparkMLModelLoaderTest.class);
    private static String modelPath = "build/lrModel";

    @BeforeClass
    public static void trainLogisticModel() throws IOException {
        ClassLoader classLoader = SparkMLModelLoaderTest.class.getClassLoader();
        // the data file is downloaded from spark's github project
        String dataFilePath = classLoader.getResource("data/mllib/sample_libsvm_data.txt").getPath();
        logger.info("training a logistic model from data: " + dataFilePath);
        SparkSession spark = SparkSession.builder().master("local[1]").getOrCreate();
        SparkContext sc = spark.sparkContext();
        sc.setLogLevel("WARN");
        // Load training data
        Dataset<Row> training = spark.read().format("libsvm")
                .load(dataFilePath);


        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(training);

        // Print the coefficients and intercept for logistic regression
        logger.info("Coefficients: " + lrModel.coefficients()
                + " Intercept: " + lrModel.intercept());
        lrModel.write().overwrite().save(modelPath);
        spark.stop();
    }


    @Test
    public void testLoadProbabilisticClassificationModel() {
        ProbabilisticClassificationModel<Vector, LogisticRegressionModel> model = SparkMLModelLoader.loadProbabilisticClassificationModel(modelPath);
        int numFeatures = 692;
        Assert.assertEquals(numFeatures, model.numFeatures());
        Vector x = Vectors.sparse(numFeatures, new int[]{244 ,263}, new double[]{-1000.0, -2000.0});

        Vector predProbs = model.predictProbability(x);
        logger.info("prediction predProbs: " + predProbs);

        double predLabel = model.predict(x);
        logger.info("prediction label: " + predLabel);
        assertTrue(predLabel > 0);

        model.setThresholds(new double[] {0.1, 0.9});
        logger.info("new threshold: " + model.getThresholds()[1]);
        assertTrue(model.predict(x) < 1);
    }


}
