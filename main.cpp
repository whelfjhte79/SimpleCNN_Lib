#include"CNN.h"
#include<iostream>

using namespace std;
using namespace cnn;

int main(void) {
	cnn::CNN cnn;
	cnn.setStructure(convolution,activation,pooling,padding,convolution,activation,pooling,fullyconnected,activation);
}