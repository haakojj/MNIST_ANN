#pragma once

#include "libs.h"

namespace MNIST
{
	typedef Eigen::VectorXf Layer;
	typedef Eigen::MatrixXf Weights;

	struct Sample {
		Layer input;
		Layer output;
	};
}
