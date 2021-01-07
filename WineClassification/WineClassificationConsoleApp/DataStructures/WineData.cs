using Microsoft.ML.Data;

namespace MulticlassClassification_Wine.DataStructures
{
    /// <summary>
    /// "fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
    /// </summary>
    public class WineData
    {
        [LoadColumn(0)]
        public float fixedAcidity;
        
        [LoadColumn(1)]
        public float volatileAcidity;
        
        [LoadColumn(2)]
        public float citricAcid;
        
        [LoadColumn(3)]
        public float residualSugar;
        
        [LoadColumn(4)]
        public float chlorides;
        
        [LoadColumn(5)]
        public float freeSulfurDioxide;
        
        [LoadColumn(6)]
        public float totalSulfurDioxide;
        
        [LoadColumn(7)]
        public float density;
        
        [LoadColumn(8)]
        public float pH;
        
        [LoadColumn(9)]
        public float sulphates;
        
        [LoadColumn(10)]
        public float alcohol;
        
        [LoadColumn(11)]
        public float quality;
    }
}