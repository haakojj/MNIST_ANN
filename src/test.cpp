#include "libs.h"
#include "MNIST_database.h"
#include "Sample.h"
#include "MNIST_neuralNetwork.h"

int main()
{
	MNIST::Database trainDB("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	MNIST::Database t10kDB("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	std::vector<MNIST::Sample> training = trainDB.toSample();
	std::vector<MNIST::Sample> validate = std::vector<MNIST::Sample>(training.end() - 10000, training.end());
	training.erase(training.end() - 10000, training.end());
	std::vector<MNIST::Sample> t10k = t10kDB.toSample();

	std::vector<uint> layers;
	layers.push_back(training[0].input.size());
	layers.push_back(30);
	layers.push_back(training[0].output.size());
	MNIST::NeuralNetwork nn(layers);

	uint stepSize = 1;
	uint epoch = stepSize;
	while (true)
	{
		nn.gradientDecent(0.1, 5.0, 10, stepSize, training);
		uint right = 0;
		for (auto sample = t10k.begin(); sample != t10k.end(); sample++)
		{
			MNIST::Layer::Index res;
			MNIST::Layer::Index ans;
			nn(sample->input).maxCoeff(&res);
			sample->output.maxCoeff(&ans);
			if (res == ans) right++;
		}
		std::cout << "Epoch " << epoch << ": " << right << "/" << t10k.size() << std::endl;
		epoch += stepSize;
	}
    return 0;
}
