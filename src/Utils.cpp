#include "libs.h"
#include "Utils.h"

#ifdef _WIN32
#include <intrin.h>
#endif

namespace Utils
{
	RNG::RNG()
	{
		std::random_device rd;
		gen = std::mt19937(rd());
		normal = std::normal_distribution<float>(0.0, 1.0);
	}

	RNG::RNG(uint seed)
	{
		gen = std::mt19937(seed);
		normal = std::normal_distribution<float>(0.0, 1.0);
	}

	std::mt19937 RNG::getGen() { return gen; }
	float RNG::getNormal() { return normal(gen); }
	float RNG::getNormal(float mean, float stddev)
	{
		std::normal_distribution<float> dist = std::normal_distribution<float>(mean, stddev);
		return dist(gen);
	}

	MemBlock openBinFile(const char* fileName)
	{
		MemBlock buff;
		buff.mem = nullptr;
		buff.size = 0;

		std::ifstream file;
		std::streampos size;
		file.open(fileName, std::ios::in | std::ios::binary | std::ios::ate);
		if (file.is_open())
		{
			size = file.tellg();
			buff.size = size;
			buff.mem = new char[size];
			file.seekg(0, std::ios::beg);
			file.read(buff.mem, size);
			file.close();
		}
		return buff;
	}

	uint endianSwap32(uint n)
	{
#ifdef _WIN32
		return _byteswap_ulong(n);
#elif __linux__
		return  __builtin_bswap32(n);
#endif
	}

	uint buffToUint(const char* buff)
	{
		return endianSwap32(*(reinterpret_cast<const uint*>(buff)));
	}
}
