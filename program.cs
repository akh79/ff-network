using System;
using System.IO;
using System.Globalization;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ff_neural_net
{
    // xTrain matrix contains images in rows, its dimensions (numberOfImages, 784), where 784 = 28 * 28
    // yTrain matrix contains one-hot-encoded labels, its dimensions (numberOfLabels, 10)
    // numberOfImages == numberOfLabels == 60000

    internal class Program
    {
        static bool IsValidFolderPath(string path)
        {
            try
            {
                string fullPath = Path.GetFullPath(path);
                return Path.IsPathRooted(fullPath) && Directory.Exists(fullPath);
            }
            catch
            {
                return false;
            }
        }

        // This function finds the index of a first nonzero entry
        static int IndexOfNonzeroElement(float[] array)
        {
            for (int i = 0; i < array.Length; ++i)
            {
                if (array[i] != 0.0f)
                    return i;
            }

            return -1;
        }

        static int IndexOfMaxElement(float[] array)
        {
            int index = 0;
            float maxVal = array[0];

            for (int i = 1; i < array.Length; ++i)
            {
                if (array[i] > maxVal)
                {
                    index = i;
                    maxVal = array[i];
                }
            }

            return index;
        }

        static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine($"There must be 3 input parameters on the command line. Given {args.Length}");
                return;
            }

            string basePath = args[0].Trim();

            if (!IsValidFolderPath(basePath))
            {
                Console.WriteLine("An invalid data base path is given. Terminating...");
                return;
            }

            if (!basePath.EndsWith("/") && !basePath.EndsWith("\\"))
            {
                basePath += "/";
            }

            if (!int.TryParse(args[1], out int minibatches))
            {
                Console.WriteLine("Number of minibatches isn't a valid integer. Terminating...");
                return;
            }

            if (!float.TryParse(args[2], NumberStyles.Float, CultureInfo.InvariantCulture, out float learningRate))
            {
                Console.WriteLine("Learning rate isn't a valid float number.Terminating...");
                return;
            }

            string trainImageFilePath = basePath + "train-images.idx3-ubyte";
            string trainLabelFilePath = basePath + "train-labels.idx1-ubyte";

            var trainData = MNISTReader.ReadImages(trainImageFilePath);
            float[][] trainLabels = MNISTReader.ReadLabels(trainLabelFilePath);

            NeuralNetwork net = new NeuralNetwork();
            net.Add(new FullyConnectedLayer(trainData.rows * trainData.cols, 100));
            net.Add(new ActivationLayer(100, 100));
            net.Add(new FullyConnectedLayer(100, 50));
            net.Add(new ActivationLayer(50, 50));
            net.Add(new FullyConnectedLayer(50, 10));
            net.Add(new ActivationLayer(10, 10));

            // Train
            net.Fit(trainData.images, trainLabels, minibatches, learningRate);

            string testImageFilePath = basePath + "t10k-images.idx3-ubyte";
            string testLabelFilePath = basePath + "t10k-labels.idx1-ubyte";

            var testData = MNISTReader.ReadImages(testImageFilePath);
            float[][] testLabels = MNISTReader.ReadLabels(testLabelFilePath);

            //Test
            int[][] cm = new int[10][];     // matrix of prediction results
            for (int i = 0; i < cm.Length; ++i)
            {
                cm[i] = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            }

            int count = testData.images.Length;

            for (int i = 0; i < count; ++i)
            {
                float[] output = net.Predict(testData.images[i]);

                int test = IndexOfNonzeroElement(testLabels[i]);
                int pred = IndexOfMaxElement(output);

                cm[test][pred] += 1;
            }

            // Accuracy
            int sumDiag = 0, sumTotal = 0;

            for (int i = 0; i < cm.Length; ++i)
            {
                for (int j = 0; j < cm[i].Length; ++j)
                {
                    if (i == j)
                        sumDiag += cm[i][j];

                    sumTotal += cm[i][j];
                }
            }

            float accuracy = (float)sumDiag / (float)sumTotal;

            // Print the cumulative result matrix cm
            for (int i = 0; i < cm.Length; ++i)
            {
                for (int j = 0; j < cm[i].Length; ++j)
                {
                    Console.Write($"{cm[i][j], 8}");
                }
                Console.WriteLine();
            }

            Console.WriteLine($"accuracy: {accuracy}");
        }
    }
}
