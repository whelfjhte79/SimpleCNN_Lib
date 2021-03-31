#ifndef CNN_H
#define CNN_H

#include<vector>
#include<iostream>
#include<fstream>
#include<random>
using namespace std;


using V3D = std::vector<std::vector<std::vector<double>>>;
using V2D = std::vector<std::vector<double>>;
using V1D = std::vector<double>;

using namespace std;

namespace preprocess {
	class Image {
		V3D imageData;
		V1D labelData;
		V2D onehotLabelData;
	public:
		Image();
		V3D getImageData();
		V2D getLabelData();
		void oneHotEncoding(V2D& onehotLabelData);

		int ReverseInt(int i);
		void readDataImage(int NumberOfImages, int DataOfAnImageY, int DataOfAnImageX, V3D& arr);
		void readDataLabel(V1D& arr);
	};
	Image::Image() {
		this->readDataImage(10000, 28, 28, this->imageData);
		this->readDataLabel(this->labelData);
		this->oneHotEncoding(this->onehotLabelData);
	}
	V3D Image::getImageData() {
		return this->imageData;
	}
	V2D Image::getLabelData() {
		return this->onehotLabelData;
	}
	void Image::oneHotEncoding(V2D& onehotLabelData) {
		V1D v;
		for (int i = 0; i < this->labelData.size(); ++i) {
			for (int j = 0; j < 10; ++j) {
				v.push_back(0.0);
			}
			v[this->labelData[i]] = 1;
			onehotLabelData.push_back(v);
			v.clear();
		}
	}
	int Image::ReverseInt(int i) {
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}
	void Image::readDataImage(int NumberOfImages, int DataOfAnImageY, int DataOfAnImageX, V3D& arr) {
		arr.resize(NumberOfImages, V2D(DataOfAnImageY,V1D(DataOfAnImageX)));
		ifstream file("train-images.idx3-ubyte", ios::binary);
		if (file.is_open()) {
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;
			file.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			file.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			file.read((char*)&n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);
			file.read((char*)&n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);

			for (int i = 0; i < 10000; ++i) {
				for (int r = 0; r < n_rows; ++r) {
					for (int c = 0; c < n_cols; ++c) {
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						arr[i][r][c] = (double)temp;
					}
				}
			}
		}
	}
	void Image::readDataLabel(V1D& arr) {
		ifstream file("train-labels.idx1-ubyte");
		for (int i = 0; i < 10000; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (i > 7) {
				arr.push_back((double)temp);
			}
		}
	}


}
namespace cnn {
	enum class StructType {
		None,
		Convolution,
		Padding,
		Pooling,
		Activation,
		Fullyconnected,
		Optimizer
	};

	class CNN {
	private:
		V3D filter;
		StructType type = StructType::None;

		V3D feature;

		//initialization
		V3D input;
		V2D label;
		int batchSize = 0;
		double learningRate = 0.0;
		int epoch = 0;
		int stride = 0;
		int paddingSize = 0;
		int outputSize = 0;
	public:
		CNN();
		CNN(V3D& input, V2D& label);
		CNN(V3D& input, V2D& label, int batchSize, double learningRate, int epoch, int stride, int paddingSize, int outputSize);
		void init(V3D& input, V2D& label);
		void init(V3D& input, V2D& label, int batchSize, double learningRate, int epoch, int stride, int paddingSize, int outputSize);
		void setStructure(V3D(*arg)(StructType&, V3D&, V3D&));
		template<typename T, typename... Args> void setStructure(T arg, Args... args);

		V3D& setRandomFilter(V3D& filter);
	};
	CNN::CNN() { 
		this->type = StructType::None;
		this->setRandomFilter(this->filter);
	}
	CNN::CNN(V3D& input, V2D& label) {
		this->input = input;
		this->label = label;
	}
	CNN::CNN(V3D& input, V2D& label, int batchSize, double learningRate, int epoch, int stride, int paddingSize, int outputSize) :
		input(input), label(label), batchSize(batchSize), learningRate(learningRate), epoch(epoch), stride(stride), paddingSize(paddingSize), outputSize(outputSize) {

		this->type = StructType::None;
		this->setRandomFilter(this->filter);
	}
	void CNN::init(V3D& input, V2D& label) {
		this->input = input;
		this->label = label;
	}
	void CNN::init(V3D& input, V2D& label, int batchSize, double learningRate, int epoch, int stride, int paddingSize, int outputSize) {
		this->input = input;
		this->label = label;
		this->batchSize = batchSize;
		this->learningRate = learningRate;
		this->epoch = epoch;
		this->stride = stride;
		this->paddingSize = paddingSize;
		this->outputSize = outputSize;
	}
	void CNN::setStructure(V3D(*arg)(StructType&, V3D&, V3D&)) {
		auto structType = arg;
		this->feature = structType(this->type, this->input, this->filter);
	}
	template<typename T, typename... Args> inline void CNN::setStructure(T arg, Args... args) {
		auto structType = arg;
		this->feature = structType(this->type, this->input, this->filter);
		this->setStructure(args...);
	}
	V3D& CNN::setRandomFilter(V3D& filter) {
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_int_distribution<int> uid(-10000, 10000);
		for (int i = 0; i < filter.size(); ++i) {
			for (int j = 0; j < filter[0].size(); ++j) {
				for (int k = 0; k < filter[0][0].size(); ++k) {
					filter[i][j][k] = (double)uid(mt) / 10000.0;
				}
			}
		}
		return filter;
	}
	// //////////////////////////////////////////////////
	// //////////////////////////////////////////////////

	V3D convolution(StructType& type, V3D& input, V3D& filter);
	V3D pooling(StructType& type, V3D& input, V3D& filter);
	V3D padding(StructType& type, V3D& input, V3D& filter);
	V3D activation(StructType& type, V3D& input, V3D& filter);
	V3D fullyconnected(StructType& type, V3D& input, V3D& filter);

	// 스트라이드 받아오는 방식 수정 필요.
	static int stride = 1;
	V3D convolution(StructType& type, V3D& input, V3D& filter) {
		type = StructType::Convolution;
		int W = input[0][0].size();
		int F = filter[0][0].size();
		int S = stride;
		int P = 0;

		int outputX = ((W - F + (2 * P)) / S) + 1;
		int outputY = input[0].size() -2;

		V2D conv2D(3, V1D(3));
		//(featImgSize * filter.size(), V2D(featSizeY, V1D(featSizeX, 0.0)));
		V3D temp(input.size() * filter.size(), V2D(input[0].size(), V1D(input[0][0].size(), 0.0)));
		if (stride <= 0) stride = 1;
		if (stride > 5) stride = 5;

		int cnt = 0;
		for (int i = 0; i < filter.size(); ++i) {
			for (int j = 0; j < input.size(); ++j) {
				for (int k = 0; k < outputY; ++k) {
					for (int l = 0; l < outputX; ++l) {
						for (int m = 0; m < filter[0].size(); ++m) {
							for (int n = 0; n < filter[0][0].size(); ++n) {
								conv2D[m][n] = input[j][k + m][l * stride] * filter[i][m][n];
								temp[cnt][k][l] += conv2D[m][n];
							}
						}
					}
				}
				cnt++;
			}
		}
		return input;
	}
	V3D pooling(StructType& type, V3D& input, V3D& filter) {
		type = StructType::Pooling;
		return input;
	}
	V3D padding(StructType& type, V3D& input, V3D& filter) {
		type = StructType::Padding;
		return input;
	}
	V3D activation(StructType& type, V3D& input, V3D& filter) {
		type = StructType::Activation;
		return input;
	}
	V3D fullyconnected(StructType& type, V3D& input, V3D& filter) {
		type = StructType::Fullyconnected;
		return input;
	}
}

#endif//