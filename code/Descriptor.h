#ifndef Descriptor_h
#define Descriptor_h
#include "include.h"
class Descriptor{
public:
	float			xi, yi;	// position of the Feature
	vector<double>	fv;		// 'Feature's features :)
	Descriptor(){}
	Descriptor(float x, float y, vector<double> const& f){
		xi = x;
		yi = y;
		fv = f;
	}
};
#endif
