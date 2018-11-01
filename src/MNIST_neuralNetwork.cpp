#include "libs.h"
#include "MNIST_neuralNetwork.h"
#include "MNIST_image.h"

namespace MNIST
{
	// Initialize weights and biases randomly, with standard normal distribution.
	NeuralNetwork::NeuralNetwork(const std::vector<uint>& layerSizes)
	{
		rng = Utils::RNG();
		this->layerSizes = layerSizes;
		for (auto i = layerSizes.begin(); i != layerSizes.end() - 1; i++)
		{
			uint sx = *i;
			uint sy = *(i + 1);
			Weights m(sy, sx);
			Layer v(sy);

			for (uint y = 0; y < sy; y++)
			{
				v[y] = rng.getNormal(0.0, sqrt(1.0/sx));
				for (int x = 0; x < sx; x++) m(y, x) = rng.getNormal(0.0, sqrt(1.0 / sx));
			}

			weights.push_back(m);
			biases.push_back(v);
		}
	}

	// Run network on some input vector, and return output vector.
	Layer NeuralNetwork::operator() (const Layer& input) const
	{
		std::vector<Layer> tmp = feedForward(input);
		return sigmoid(tmp.back());
	}

	// Return weighed inputs for each layer.
	std::vector<Layer> NeuralNetwork::feedForward(const Layer& input) const
	{
		std::vector<Layer> res;
		Layer last = input;
		for (uint layer = 0; layer < weights.size(); layer++)
		{
			Layer tmp = weights[layer] * last + biases[layer];
			res.push_back(tmp);
			last = sigmoid(tmp);
		}
		return res;
	}

	// Backpropagate the error through the network.
	std::vector<Layer> NeuralNetwork::backPropagate(const std::vector<Layer>& wInn, const Layer& outError) const
	{
		std::vector<Layer> errors = wInn;
		errors.back() = outError;
		for (int layer = wInn.size() - 2; layer >= 0; layer--) errors[layer] = (weights[layer + 1].transpose() * errors[layer + 1]).cwiseProduct(diffSig(wInn[layer]));
		return errors;
	}

	// Apply stochastic gradient decent learing algorithm. 
	void NeuralNetwork::gradientDecent(float learningRate, float regParam, uint batchSize, uint epochs, const std::vector<Sample>& samples)
	{
		std::vector<Sample const*> samplePtrs;
		for (auto sample = samples.begin(); sample != samples.end(); sample++) samplePtrs.push_back(&(*sample));
		uint batchCount = samples.size() / batchSize;

		for (uint epoch = 0; epoch < epochs; epoch++)
		{
			std::shuffle(samplePtrs.begin(), samplePtrs.end(), rng.getGen());
			uint sampleIndex = 0;
			for (uint batch = 0; batch < batchCount; batch++)
			{
				std::vector<Weights> gradWeights;
				std::vector<Layer> gradBiases;
				for (auto i = layerSizes.begin(); i != layerSizes.end() - 1; i++)
				{
					uint sx = *i;
					uint sy = *(i + 1);
					Weights m = Weights::Zero(sy, sx);
					Layer v = Layer::Zero(sy);
					gradWeights.push_back(m);
					gradBiases.push_back(v);
				}

				for (uint batchIndex = 0; batchIndex < batchSize; batchIndex++)
				{
					Sample const* sample = samplePtrs[sampleIndex];

					std::vector<Layer> wInn = feedForward(sample->input);
					std::vector<Layer> errors = backPropagate(wInn, sigmoid(wInn.back()) - sample->output);

					gradWeights[0] += errors[0] * sample->input.transpose();
					gradBiases[0] += errors[0];
					for (uint layer = 1; layer < errors.size(); layer++)
					{
						gradWeights[layer] += errors[layer] * sigmoid(wInn[layer - 1]).transpose();
						gradBiases[layer] += errors[layer];
					}

					sampleIndex++;
				}

				float learnFac = learningRate / batchSize;
				float regFac = 1.0 - (learningRate * regParam / samples.size());
				for (uint layer = 0; layer < weights.size(); layer++)
				{
					weights[layer] = regFac * weights[layer] - learnFac * gradWeights[layer];
					biases[layer] = biases[layer] - learnFac * gradBiases[layer];
				}
			}
		}
	}

	float sigmoid(float x)
	{
		if (x > 20.0) return 1.0;
		else if (x < -20.0) return 0.0;
		return 1.0 / (1.0 + exp(-x));
	}

	Layer sigmoid(const Layer& x)
	{
		Layer res = x;
		for (uint i = 0; i < res.rows(); i++) res[i] = sigmoid(res[i]);
		return res;
	}

	float diffSig(float x)
	{
		if (x > 20.0 || x < -20.0) return 0.0;
		float tmp = exp(-x);
		return tmp / ((tmp + 1.0) * (tmp + 1.0));
	}

	Layer diffSig(const Layer& x)
	{
		Layer res = x;
		for (uint i = 0; i < res.rows(); i++) res[i] = diffSig(res[i]);
		return res;
	}
}
