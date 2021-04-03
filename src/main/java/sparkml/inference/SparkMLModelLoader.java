package sparkml.inference;

import org.apache.spark.ml.classification.ProbabilisticClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Method;

public class SparkMLModelLoader {
    private static String getModelClassName(String path) throws IOException, ParseException {
        JSONParser jsonParser = new JSONParser();

        // TODO: what if multiple path-xxxxx in the path ?
        JSONObject jsonObject = (JSONObject) jsonParser.parse(new FileReader(path + "/metadata/part-00000"));

        String className = (String) jsonObject.get("class");
        System.out.println("className: " + className);
        return className;
    }

    public static <FeaturesType, M extends ProbabilisticClassificationModel<FeaturesType, M>>
    ProbabilisticClassificationModel<FeaturesType, M> loadProbabilisticClassificationModel(String path) {

        ProbabilisticClassificationModel<FeaturesType, M> model = null;
        try {
            SparkSession spark = SparkSession.builder().master("local[1]").getOrCreate();

            String className = getModelClassName(path);
            Method m = Class.forName(className).getMethod("load", String.class);
            Object modelObj = m.invoke(null, path); // load is a static method, no object is required to invoke

            model = (ProbabilisticClassificationModel<FeaturesType, M>) modelObj;

            spark.stop();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return model;
    }
}

