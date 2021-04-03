# sparkml-inference
SparkML Inference for Single Instance

```
String modelPath = "path-to-sparkml-model";
ProbabilisticClassificationModel<Vector, LogisticRegressionModel> model = SparkMLModelLoader.loadProbabilisticClassificationModel(modelPath);

// define an sparse feature vector example
Vector x = Vectors.sparse(model.numFeatures(), new int[]{244 ,263}, new double[]{-1000.0, -2000.0});

// get prediction as probability per class
Vector predProbs = model.predictProbability(x); // example result:  [0.3822148720234751,0.617785127976525]
```