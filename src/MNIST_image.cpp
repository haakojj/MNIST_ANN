#include "libs.h"
#include "MNIST_image.h"
#include "Utils.h"
#include "Sample.h"

namespace MNIST {
	Image::Image()
	{
		width = 0;
		height = 0;
		label = 0;
		pixels = nullptr;
	}

	Image::Image(uint width, uint height, uint8 label, const char* pixels)
	{
		this->width = width;
		this->height = height;
		this->label = label;
		this->pixels = new char[width * height];
		std::memcpy(this->pixels, pixels, width * height);
	}

	Image::~Image()
	{
		delete[] pixels;
	}

	Image::Image(const Image& other)
	{
		width = other.getWidth();
		height = other.getHeight();
		label = other.getLabel();
		pixels = new char[width * height];
		std::memcpy(pixels, other.getRawPixelPtr(), width * height);
	}

	Image& Image::operator=(Image other)
	{
		swapImg(*this, other);
		return *this;
	}

	Image::Image(Image&& other) : Image()
	{
		swapImg(*this, other);
	}

	Image& Image::operator=(Image&& other)
	{
		swapImg(*this, other);
		return *this;
	}

	void swapImg(Image& first, Image& second)
	{
		uint tmpWidth = first.width;
		uint tmpHeight = first.height;
		uint8 tmpLabel = first.label;
		char* tmpPixels = first.pixels;

		first.width = second.width;
		first.height = second.height;
		first.label = second.label;
		first.pixels = second.pixels;

		second.width = tmpWidth;
		second.height = tmpHeight;
		second.label = tmpLabel;
		second.pixels = tmpPixels;
	}

	char* Image::operator[](uint y)
	{
		return pixels + width * y;
	}

	void Image::writePNG(char const* const filename) const
	{
		png::image< png::gray_pixel > image(width, height);
		for (uint x = 0; x < width; x++)
		{
			for (uint y = 0; y < height; y++)
			{
				image[y][x] = png::gray_pixel(pixels[width * y + x]);
			}
		}
		image.write(filename);
	}

	// Return image as sample with pixel values in range [0.0, 1.0].
	Sample Image::toSample() const
	{
		Layer input(width * height);
		for (uint i = 0; i < input.rows(); i++)
		{
			uint8* tmp = reinterpret_cast<uint8*> (pixels + i);
			input[i] = (*tmp) / 255.0;
		}
		Layer output = Layer::Zero(10);
		output[label] = 1.0;
		Sample sample;
		sample.input = input;
		sample.output = output;
		return sample;
	}

	uint Image::getWidth() const
	{
		return width;
	}

	uint Image::getHeight() const
	{
		return height;
	}

	char* Image::getRawPixelPtr() const
	{
		return pixels;
	}

	uint8 Image::getLabel() const
	{
		return label;
	}
}
