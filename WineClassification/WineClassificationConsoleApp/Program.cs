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
            var mlContext = new MLContext(seed: 0);

            BuildTrainEvaluateAndSaveModel(mlContext);

            TestSomePredictions(mlContext);

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<WineData>(TrainDataPath, separatorChar: ';', hasHeader: true);
            var testDataView = mlContext.Data.LoadFromTextFile<WineData>(TestDataPath, separatorChar: ';', hasHeader: true);


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

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: nameof(WineData.quality), inputColumnName: "KeyColumn"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "quality", "Score");
            
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {trainer.ToString()} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<WineData, WinePrediction>(trainedModel);

            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();

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
            var resultprediction1 = predEngine.Predict(SampleWineData.Wine1);
            var indexOfHighestScoreForPrediction1 = GetIndexOfHigherScore(resultprediction1);

            Console.WriteLine($"Actual: 6.     Predicted label and score:  {WineQualities[labelsArray[indexOfHighestScoreForPrediction1]]}: {resultprediction1.Score[indexOfHighestScoreForPrediction1]:0.####}");
            Console.WriteLine();

            var resultprediction2 = predEngine.Predict(SampleWineData.Wine2);
            var indexOfHighestScoreForPrediction2 = GetIndexOfHigherScore(resultprediction2);

            Console.WriteLine($"Actual: 7.   Predicted label and score:  {WineQualities[labelsArray[indexOfHighestScoreForPrediction2]]}: {resultprediction2.Score[indexOfHighestScoreForPrediction2]:0.####}");
            Console.WriteLine();

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
