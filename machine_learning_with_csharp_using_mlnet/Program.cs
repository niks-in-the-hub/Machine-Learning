using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLTutorial
{
    //iris data class for features
    public class IrisData
    {
        [Column("0")]
        public float SepalLength;
        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

        [Column("4")]
        [ColumnName("Label")]
        public string Label;
    }

    //IrisPrediction is the result returned prediction by the model
    //class to hold the label.
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }


    class Program
    {
        static void Main(string[] args)
        {
            //Create a Pipeline and Load the Data
            var pipeline = new LearningPipeline();
            string dataPath = "flowers.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            //Transform data from string to numeric
            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            //adding learning/training algorithm
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            //Train the model
            var model = pipeline.Train<IrisData, IrisPrediction>();

            //Using the model to make predictions
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 0.3f,
                SepalWidth = 0.6f,
                PetalLength = 1.2f,
                PetalWidth = 1.1f
            });
            Console.WriteLine($"Pridicted flower class is : {prediction.PredictedLabels}");
            Console.Read();
        }
    }
}
