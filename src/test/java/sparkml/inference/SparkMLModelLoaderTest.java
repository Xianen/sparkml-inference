package sparkml.inference;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.spark.sql.RowFactory;

import java.util.ArrayList;
import java.util.List;

public class SparkMLModelLoaderTest {
    @Before
    public void trainLogisticModel() {
        SparkSession spark = SparkSession.builder().master("local[1]").getOrCreate();
        SparkContext sc = spark.sparkContext();
        sc.setLogLevel("INFO");
        Row row0 = RowFactory.create(1.0, Vectors.dense(1.0, 2.0));
        Row row1 = RowFactory.create(0.0, Vectors.dense(-1.0, -2.0));

        // TODO: create sample data set and train a model
        List<Row> list=new ArrayList<Row>();
        list.add(row0);
        list.add(row1);
        //spark
        //sc.parallelize(list, 1);
        spark.stop();
    }
    @Test
    public void testLoadProbabilisticClassificationModel() {
        ProbabilisticClassificationModel<Vector, LogisticRegressionModel> model = SparkMLModelLoader.loadProbabilisticClassificationModel("/Users/xiqiu/blor");

        double predLabel = model.predict(Vectors.dense(-1.1, 2.2));
        assertTrue(predLabel > 0);

        Vector predProbs = model.predictProbability(Vectors.dense(-1.1, 2.2));
        System.out.println("prediction label: " + predLabel);
        System.out.println("prediction predProbs: " + predProbs);

    }


}
