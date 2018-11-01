#pragma once

#include "libs.h"

typedef uint32_t uint;
typedef uint8_t uint8;

namespace Utils
{
	struct MemBlock {
		char* mem;
		uint size;
	};

	class RNG {
	public:
		RNG();
		RNG(uint seed);
		float getNormal();
		float getNormal(float mean, float stddev);
		std::mt19937 getGen();
	private:
		std::mt19937 gen;
		std::normal_distribution<float> normal;
	};

	MemBlock openBinFile(const char* fileName);
	uint endianSwap32(uint n);
	uint buffToUint(const char* buff);
}
