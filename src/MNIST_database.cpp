#include "libs.h"
#include "MNIST_database.h"
#include "MNIST_image.h"
#include "Utils.h"
#include "Sample.h"


namespace MNIST {
	Database::Database()
	{
		size = 0;
		imgs = nullptr;
	}

	Database::Database(uint size, Image* imgs)
	{
		this->size = size;
		this->imgs = imgs;
	}

	Database::Database(const char* imgFileName, const char* labelFileName)
	{
		openFile(imgFileName, labelFileName);
	}

	Database::~Database()
	{
		delete[] imgs;
	}

	Database::Database(const Database& other)
	{
		size = other.size;
		std::memcpy(imgs, other.getRawImgPtr(), size);
	}

	Database& Database::operator=(Database other)
	{
		swapDB(*this, other);
		return *this;
	}

	Database::Database(Database&& other) : Database()
	{
		swapDB(*this, other);
	}

	Database& Database::operator=(Database&& other)
	{
		swapDB(*this, other);
		return *this;
	}

	void swapDB(Database& first, Database& second)
	{
		uint tmpSize = first.size;
		Image* tmpImgs = first.imgs;

		first.size = second.size;
		first.imgs = second.imgs;

		second.size = tmpSize;
		second.imgs = tmpImgs;
	}

	std::vector<Sample> Database::toSample()
	{
		std::vector<Sample> res;
		for (uint i = 0; i < size; i++) res.push_back(imgs[i].toSample());
		return res;
	}

	void Database::openFile(const char* imgFileName, const char* labelFileName)
	{
		Utils::MemBlock imgBuff = Utils::openBinFile(imgFileName);
		if (imgBuff.size == 0)
		{
			this->size = 0;
			this->imgs = nullptr;
			return;
		}

		Utils::MemBlock labelBuff = Utils::openBinFile(labelFileName);
		if (labelBuff.size == 0)
		{
			this->size = 0;
			this->imgs = nullptr;
			delete[] imgBuff.mem;
			return;
		}

		uint imgMagicNum = Utils::buffToUint(imgBuff.mem);
		uint imgSize = Utils::buffToUint(imgBuff.mem + 4);
		uint height = Utils::buffToUint(imgBuff.mem + 8);
		uint width = Utils::buffToUint(imgBuff.mem + 12);
		uint labelMagicNum = Utils::buffToUint(labelBuff.mem);
		uint labelSize = Utils::buffToUint(labelBuff.mem + 4);
		if (imgMagicNum != 0x00000803 || labelMagicNum != 0x00000801 || imgSize != labelSize)
		{
			this->size = 0;
			this->imgs = nullptr;
			delete[] imgBuff.mem;
			delete[] labelBuff.mem;
			return;
		}

		this->size = imgSize;
		this->imgs = new Image[imgSize];
		for (uint i = 0; i < imgSize; i++)
		{
			uint8 label = *(reinterpret_cast<uint8*>(labelBuff.mem + 8 + i));
			Image img(width, height, label, imgBuff.mem + 16 + i*width*height);
			this->imgs[i] = img;
		}

		delete[] imgBuff.mem;
		delete[] labelBuff.mem;
	}

	Image Database::operator[](uint n)
	{
		return *(imgs + n);
	}

	uint Database::getSize() const
	{
		return size;
	}

	Image* Database::getRawImgPtr() const
	{
		return imgs;
	}
}
