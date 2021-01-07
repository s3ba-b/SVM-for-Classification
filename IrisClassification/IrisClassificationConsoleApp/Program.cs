using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MulticlassClassification_Wine.DataStructures;

namespace MulticlassClassification_Wine
{
    public static partial class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"../../../../Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/wine-train.txt";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/wine-test.txt";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static string BaseModelsRelativePath = @"../../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/WineQualityClassificationModel.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        private static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);

            //1.
            BuildTrainEvaluateAndSaveModel(mlContext);

            //2.
            TestSomePredictions(mlContext);

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<WineData>(TrainDataPath, separatorChar: ';', hasHeader: true);
            var testDataView = mlContext.Data.LoadFromTextFile<WineData>(TestDataPath, separatorChar: ';', hasHeader: true);
            

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(WineData.quality))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(WineData.fixedAcidity),
                                                                                   nameof(WineData.volatileAcidity),
                                                                                   nameof(WineData.citricAcid),
                                                                                   nameof(WineData.residualSugar),
                                                                                   nameof(WineData.chlorides),
                                                                                   nameof(WineData.freeSulfurDioxide),
                                                                                   nameof(WineData.totalSulfurDioxide),
                                                                                   nameof(WineData.density),
                                                                                   nameof(WineData.pH),
                                                                                   nameof(WineData.sulphates),
                                                                                   nameof(WineData.alcohol))
                                                                       .AppendCacheCheckpoint(mlContext)); 
                                                                       // Use in-memory cache for small/medium datasets to lower training time. 
                                                                       // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets. 

            // STEP 3: Set the training algorithm, then append the trainer to the pipeline  
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: nameof(WineData.quality) , inputColumnName: "KeyColumn"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

            // STEP 5: Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "quality", "Score");

            //Common.ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            //Test Classification Predictions with some hard-coded samples 
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<WineData, WinePrediction>(trainedModel);

            // During prediction we will get Score column with 3 float values.
            // We need to find way to map each score to original label.
            // In order to do that we need to get TrainingLabelValues from Score column.
            // TrainingLabelValues on top of Score column represent original labels for i-th value in Score array.
            // Let's look how we can convert key value for PredictedLabel to original labels.
            // We need to read KeyValues for "PredictedLabel" column.
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();

            // Since we apply MapValueToKey estimator with default parameters, key values
            // depends on order of occurence in data file. Which is "Iris-setosa", "Iris-versicolor", "Iris-virginica"
            // So if we have Score column equal to [0.2, 0.3, 0.5] that's mean what score for
            // Iris-setosa is 0.2
            // Iris-versicolor is 0.3
            // Iris-virginica is 0.5.
            //Add a dictionary to map the above float values to strings. 
            Dictionary<float, string> WineQualities = new Dictionary<float, string>();
            WineQualities.Add(0, "0");
            WineQualities.Add(1, "1");
            WineQualities.Add(2, "2");
            WineQualities.Add(3, "3");
            WineQualities.Add(4, "4");
            WineQualities.Add(5, "5");
            WineQualities.Add(6, "6");
            WineQualities.Add(7, "7");
            WineQualities.Add(8, "8");
            WineQualities.Add(9, "9");
            WineQualities.Add(10, "10");

            Console.WriteLine("=====Predicting using model====");
            //Score sample 1
            var resultprediction1 = predEngine.Predict(SampleWineData.Wine1);
            var indexOfHighestScoreForPrediction1 = GetIndexOfHigherScore(resultprediction1);

            Console.WriteLine($"Actual: 6.     Predicted label and score:  {WineQualities[labelsArray[indexOfHighestScoreForPrediction1]]}: {resultprediction1.Score[indexOfHighestScoreForPrediction1]:0.####}");
            Console.WriteLine();

            //Score sample 2
            var resultprediction2 = predEngine.Predict(SampleWineData.Wine2);
            var indexOfHighestScoreForPrediction2 = GetIndexOfHigherScore(resultprediction2);

            Console.WriteLine($"Actual: 7.   Predicted label and score:  {WineQualities[labelsArray[indexOfHighestScoreForPrediction2]]}: {resultprediction2.Score[indexOfHighestScoreForPrediction2]:0.####}");
            Console.WriteLine();

            //Score sample 3
            var resultprediction3 = predEngine.Predict(SampleWineData.Wine3);
            var indexOfHighestScoreForPrediction3 = GetIndexOfHigherScore(resultprediction3);

            Console.WriteLine($"Actual: 6.   Predicted label and score: {WineQualities[labelsArray[indexOfHighestScoreForPrediction3]]}: {resultprediction3.Score[indexOfHighestScoreForPrediction3]:0.####}");
            Console.WriteLine();
        }

        private static int GetIndexOfHigherScore(WinePrediction resultPrediction)
        {
            var maxScore = resultPrediction.Score.Max();
            int index = 0;

            foreach (var score in resultPrediction.Score)
                if (score == maxScore)
                    return index;
                else
                    index++;

            throw new ArgumentNullException();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
