#ifndef Feature_h
#define Feature_h
#include "include.h"

class Feature{
	public:
		float			x;
		float			y;		// position of feature
		vector<double>	mag;	// magnitudes at (x,y)
		vector<double>	orien;	// orientations
		int				scale;	// The scale where feature was detected
	
		Feature() {}
		Feature(float x1, float y1) {x=x1; y=y1;}
		Feature(float x1, float y1, vector<double> const& m, vector<double> const& o, unsigned int s){
			x     = x1;
			y     = y1;
			mag   = m;
			orien = o;
			scale = s;
		}
};

#endif
