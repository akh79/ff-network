# ff-network
Basic feed-forward neural network structured after https://github.com/rkneusel9/MathForDeepLearning and implemented in C#.

Feature vectors are represented by row-vectors. The fully connected layer computes $f(x) = x * W + b$. A fixed activation function, sigmoid, is used throughout. Minibatch training is used: updating gradient is done for each minibatch. The loss function used is *MSE*, mean squared error.

The emphasis is put on the reduction of array instancing. This is the reason for not using any matrix library. Matrices are represented by jagged arrays, allowing for efficient fetching of rows. 

A particular ff-network is built for recognition of MNIST digits with the full 28*28 pixel resolution. The main function uses up to 4 input parameters:

- base path to a directory containing the original MNIST data files,
- number of minibatches (epochs) used for training,
- learning rate,
- size of a minibatch (optional, 64 if not given)

MNIST data files can be loaded from here: https://github.com/cvdfoundation/mnist

My tests show that converges is moderate. Parameters, I used for testing minibatches = 10000, learning rate = 1.0, size of minibatch = 64. After approximately 5000 minibatch steps there is no significant reduction of error. The resulting accuracy on MNIST test data is 0.959 for that parameter set.

