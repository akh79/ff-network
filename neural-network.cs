using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.ExceptionServices;
using System.IO;
using System.Reflection;

namespace ff_neural_net
{
    // All 1d arrays are row-vectors, multiplied by matrix on the right

    internal delegate float Activation(float x);

    public abstract class Layer
    {
        internal static Random rnd = new Random();

        internal readonly int mInputSize;
        internal readonly int mOutputSize;
        internal float[] mInputs;
        internal float[] mOutputs;
        internal float[] mInputError;

        internal Layer(int inputSize, int outputSize)
        {
            mInputSize = inputSize;
            mOutputSize = outputSize;
            mInputs = new float[mInputSize];
            mOutputs = new float[mOutputSize];
            mInputError = new float[mInputSize];
        }

        internal abstract float[] Forward(float[] inputData);

        internal abstract float[] Backward(float[] outputError);

        internal abstract void Step(float eta);
    }

    public class ActivationLayer : Layer
    {
        private Activation activationFunc;
        private Activation activationDerv;
        internal ActivationLayer(int inputSize, int outputSize)
            : base(inputSize, outputSize)
        {
            activationFunc = new Activation((float x) => 1.0f / (1.0f + (float)System.Math.Exp(-(double)x)));
            activationDerv = new Activation((float x) => activationFunc(x) * (1.0f - activationFunc(x)));
        }

        internal override float[] Forward(float[] inputData)
        {
            Debug.Assert(inputData.Length == mInputSize);
            Array.Copy(inputData, mInputs, mInputSize);     // no allocation

            for (int i = 0; i < mInputSize; ++i)
            {
                mOutputs[i] = activationFunc(mInputs[i]);
            }

            return mOutputs;
        }

        internal override float[] Backward(float[] outputError)
        {
            Debug.Assert(outputError.Length == mOutputSize);

            for (int i = 0; i < mOutputSize; ++i)
            {
                mInputError[i] = activationDerv(mInputs[i]) * outputError[i];
            }

            return mInputError;
        }

        internal override void Step(float eta)
        {
        }
    }

    public class FullyConnectedLayer : Layer
    {
        internal float[][] mWeights;
        internal float[] mBiases;
        internal float[][] mDeltaWeights;
        internal float[] mDeltaBiases;
        internal float[] mTempProd;
        internal int mPasses;
        internal bool mLoadParamsFromFile = false;
        private static int index = 0;

        internal void LoadParametersFromBinary(float[][] mat, float[] vec, string filePath)
        {
            using (var reader = new BinaryReader(File.Open(filePath, FileMode.Open)))
            {
                int rows = reader.ReadInt32();
                int cols = reader.ReadInt32();

                Debug.Assert(rows == mat.GetLength(0));
                Debug.Assert(cols == mat[0].Length);
                Debug.Assert(cols == vec.Length);

                // Weight matrix
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        mat[i][j] = reader.ReadSingle();
                    }
                }

                for (int i = 0; i < cols; ++i)
                {
                    vec[i] = reader.ReadSingle();
                }
            }
        }

        internal FullyConnectedLayer(int inputSize, int outputSize)
            : base(inputSize, outputSize)
        {
            mWeights = new float[inputSize][];
            for(int i = 0; i < mInputSize; ++i)
            {
                mWeights[i] = new float[mOutputSize];
                for (int j = 0; j < mOutputSize; ++j)
                {
                    if (!mLoadParamsFromFile)
                        mWeights[i][j] = (float)(rnd.NextDouble() - 0.5f);
                }
            }

            mBiases = new float[mOutputSize];
            for (int i = 0; i < mOutputSize; ++i)
            {
                if (!mLoadParamsFromFile)
                    mBiases[i] = (float)(rnd.NextDouble() * 0.1); 
            }

            mDeltaWeights = new float[mInputSize][];
            for (int i = 0; i < mInputSize; ++i)
            {
                mDeltaWeights[i] = new float[mOutputSize];
            }
            mDeltaBiases = new float[mOutputSize];

            mTempProd = new float[mOutputSize];

            mPasses = 0;

            if (mLoadParamsFromFile)
            {
                string pName = String.Format("C:/Users/akhru/Data/ff-neural-net-mathnet/params{0}.bin", index++);
                LoadParametersFromBinary(mWeights, mBiases, pName);
            }
        }

        internal override float[] Forward(float[] inputData)
        {
            Debug.Assert(inputData.Length == mInputSize);
            Array.Copy(inputData, mInputs, mInputSize);     // no allocation

            for(int i = 0; i < mOutputSize; ++i)
            {
                mTempProd[i] = mBiases[i];
            }

            for (int row = 0; row < mInputSize; ++row)
            {
                for (int col = 0; col < mOutputSize; ++col)
                {
                    mTempProd[col] += mInputs[row] * mWeights[row][col];
                }
            }

            return mTempProd;
        }

        internal override float[] Backward(float[] outputError)
        {
            Debug.Assert(outputError.Length == mOutputSize);
            Array.Clear(mInputError, 0, mInputSize);

            for (int row = 0; row < mOutputSize; ++row)
            {
                for (int col = 0; col < mInputSize; ++col)
                {
                    // Transposed weights matrix
                    mInputError[col] += outputError[row] * mWeights[col][row];
                }
            }

            // accumulate the error over the minibatch
            for (int i = 0; i < mInputSize; i++)
            {
                for (int j = 0; j < mOutputSize; j++)
                {
                    mDeltaWeights[i][j] += mInputs[i] * outputError[j];
                }
            }

            for (int i = 0; i < mOutputSize; ++i)
            {
                mDeltaBiases[i] += outputError[i];
            }

            ++mPasses;
            return mInputError;
        }

        internal override void Step(float eta)
        {
            // update the weights and biases by the mean error over the minibatch
            float factor = (eta / (float)mPasses);

            for (int i = 0; i < mInputSize; i++)
            {
                for (int j = 0; j < mOutputSize; j++)
                {
                    mWeights[i][j] -= factor * mDeltaWeights[i][j];
                }
            }

            for (int i = 0; i < mOutputSize; ++i)
            {
                mBiases[i] -= factor * mDeltaBiases[i];
            }

            // reset for the next minibatch
            for (int i = 0; i < mInputSize; ++i)
            {
                Array.Clear(mDeltaWeights[i], 0, mOutputSize);
            }

            Array.Clear(mDeltaBiases, 0, mOutputSize);

            mPasses = 0;
        }
    }

    public class NeuralNetwork
    {
        private List<Layer> mLayers;
        private bool mVerbose;

        public NeuralNetwork(bool verbose = true)
        {
            mLayers = new List<Layer>();
            mVerbose = verbose;
        }

        // TODO: Loss function can be inlined within Fit
        private float Loss(float[] train, float[] output)
        {
            Debug.Assert(train.Length == output.Length);

            float loss = 0.0f, diff = 0.0f;
            for (int i = 0; i < output.Length; ++i)
            {
                diff = train[i] - output[i];
                loss += 0.5f * diff * diff;
            }

            return loss / (float)(output.Length);
        }

        public void Add(Layer layer)
        {
            if (mLayers.Count > 0)
            {
                if (mLayers[mLayers.Count - 1].mOutputSize != layer.mInputSize)
                {
                    throw new ArgumentException("Layer cannot be added due to inputs/outputs mismatch", nameof(layer));
                }
            }

            mLayers.Add(layer);
        }

        public float[] Predict(float[] input)
        {
            float[] output = input;

            for (int i = 0; i < mLayers.Count; ++i)
            {
                output = mLayers[i].Forward(output);
            }

            return output;
        }

        // xTrain matrix contains images in rows, its dimensions (numberOfImages, 784), where 784 = 28 * 28
        // yTrain matrix contains one-hot-encoded labels, its dimensions (numberOfLabels, 10)
        // numberOfImages == numberOfLabels == 60000

        public void Fit(float[][] xTrain, float[][] yTrain, int minibatches, float rate, int batchSize = 64)
        {
            Random rand = new Random();

            Debug.Assert(xTrain.GetLength(0) == yTrain.GetLength(0));

            int inputSize = xTrain[0].Length;
            int outputSize = yTrain[0].Length;

            for (int i = 0; i < minibatches; ++i)
            {
                float err = 0.0f;

                int[] idx = Sampling.RandomKSubset(xTrain.GetLength(0), batchSize, rand);

                // process minibatch
                for (int j = 0; j < batchSize; ++j)
                {
                    // forward pass: use a row idx[j] from xTrain 
                    float[] output = xTrain[idx[j]];        // reference

                    for (int k = 0; k < mLayers.Count; ++k)
                    {
                        output = mLayers[k].Forward(output);
                    }

                    // accumulate error
                    err += Loss(yTrain[idx[j]], output);

                    // backward pass: initialize error vector with LossDerv
                    float[] error = output;                 // reference

                    // Error must be a reference to a valid output of the last layer
                    for (int k = 0; k < outputSize; ++k)
                    {
                        error[k] -= yTrain[idx[j]][k];
                    }

                    for (int k = mLayers.Count - 1; k >= 0; --k)
                    {
                        error = mLayers[k].Backward(error);
                    }
                }

                // update weights and biases
                for (int k = 0; k < mLayers.Count; ++k)
                {
                    mLayers[k].Step(rate);
                }

                // report mean loss over minibatch
                if (mVerbose && i % 10 == 0)
                {
                    err /= (float)batchSize;
                    Console.WriteLine($"Minibatch {i}/{minibatches} error: {err:F6}");
                }
            }
        }
    }
}
