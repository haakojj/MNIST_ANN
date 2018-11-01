#pragma once

#include "libs.h"
#include "Utils.h"
#include "Sample.h"

namespace MNIST
{
	class NeuralNetwork
	{
	public:
		NeuralNetwork(const std::vector<uint>& layerSizes);
		Layer operator() (const Layer& input) const;
		void gradientDecent(float learningRate, float regParam, uint batchSize, uint epochs, const std::vector<Sample>& samples);
	private:
		std::vector<uint> layerSizes;
		std::vector<Weights> weights;
		std::vector<Layer> biases;
		Utils::RNG rng;
		std::vector<Layer> feedForward(const Layer& input) const;
		std::vector<Layer> backPropagate(const std::vector<Layer>& wInn, const Layer& outError) const;
	};

	float sigmoid(float x);
	Layer sigmoid(const Layer& x);
	float diffSig(float x);
	Layer diffSig(const Layer& x);
}
