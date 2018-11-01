#pragma once

#include "libs.h"
#include "Utils.h"
#include "MNIST_image.h"
#include "Sample.h"

namespace MNIST {

	class Database {
	public:
		Database();
		Database(uint size, Image* imgs);
		Database(const char* imgFileName, const char* labelFileName);
		~Database();
		Database(const Database& other);
		Database& operator=(Database other);
		Database(Database&& other);
		Database& operator=(Database&& other);
		friend void swapDB(Database& first, Database& second);

		std::vector<Sample> toSample();
		void openFile(const char* imgFileName, const char* labelFileName);
		Image operator[](uint n);
		uint getSize() const;
		Image* getRawImgPtr() const;
	private:
		Image* imgs;
		uint size;
	};
}
