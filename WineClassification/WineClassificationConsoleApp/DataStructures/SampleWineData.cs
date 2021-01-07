namespace MulticlassClassification_Wine.DataStructures
{
    public class SampleWineData
    {
        /// <summary>
        /// 6;0.21;0.38;0.8;0.02;22;98;0.98941;3.26;0.32;11.8;6
        /// </summary>
        internal static readonly WineData Wine1 = new WineData
        {
            fixedAcidity = 6,
            volatileAcidity = (float) 0.21,
            citricAcid = (float) 0.38,
            residualSugar = (float) 0.8,
            chlorides = (float) 0.02,
            freeSulfurDioxide = 22,
            totalSulfurDioxide = 98,
            density = (float) 0.98941,
            pH = (float) 3.26,
            sulphates = (float) 0.32,
            alcohol = (float) 11.8,
            quality = 6
        };
        
        /// <summary>
        /// 5.5;0.29;0.3;1.1;0.022;20;110;0.98869;3.34;0.38;12.8;7
        /// </summary>
        internal static readonly WineData Wine2 = new WineData
        {
            fixedAcidity = (float) 5.5,
            volatileAcidity = (float) 0.29,
            citricAcid = (float) 0.3,
            residualSugar = (float) 1.1,
            chlorides = (float) 0.022,
            freeSulfurDioxide = 20,
            totalSulfurDioxide = 110,
            density = (float) 0.98869,
            pH = (float) 3.34,
            sulphates = (float) 0.38,
            alcohol = (float) 12.8,
            quality = 7
        };
        
        /// <summary>
        /// 6.5;0.24;0.19;1.2;0.041;30;111;0.99254;2.99;0.46;9.4;6
        /// </summary>
        internal static readonly WineData Wine3 = new WineData
        {
            fixedAcidity = (float) 6.5,
            volatileAcidity = (float) 0.24,
            citricAcid = (float) 0.19,
            residualSugar = (float) 1.2,
            chlorides = (float) 0.041,
            freeSulfurDioxide = 30,
            totalSulfurDioxide = 111,
            density = (float) 0.99254,
            pH = (float) 2.99,
            sulphates = (float) 0.46,
            alcohol = (float) 9.4,
            quality = 6
        };
    }
}