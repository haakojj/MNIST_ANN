#pragma once

#include "libs.h"
#include "Utils.h"
#include "Sample.h"

namespace MNIST {
	class Image {
	public:
		Image();
		Image(uint width, uint height, uint8 label, const char* pixels);
		~Image();
		Image(const Image& other);
		Image& operator=(Image other);
		Image(Image&& other);
		Image& operator=(Image&& other);
		friend void swapImg(Image& first, Image& second);

		char* operator[](uint y);
		void writePNG(const char* filename) const;
		Sample toSample() const;

		uint getWidth() const;
		uint getHeight() const;
		uint8 getLabel() const;
		char* getRawPixelPtr() const;

	private:
		char* pixels;
		uint width;
		uint height;
		uint8 label;
	};
}
