#ifndef SIFTransform_h
#define SIFTransform_h
#include "include.h"
#include "Feature.h"
#include "Descriptor.h"

struct extremaPoint{
	int octave;
	int interval;
	int y;
	int x;
	extremaPoint(int o,int i, int yi, int xi){octave=o; interval=i; x=xi; y=yi;}
};

class SIFTransform{

	private:
		//constants
		const static double CURVATURE_THRESHOLD = 5;
		const static double CONTRAST_THRESHOLD = 0.03;
	   	const static int NUM_BINS = 36;
	   	const static int MAX_KERNEL_SIZE = 20;
	   	const static int FEATURE_WINDOW_SIZE = 16;
	   	const static int FVSIZE = 128;
	   	const static float FV_THRESHOLD = 0.2;
	   	const static int DESC_NUM_BINS = 8;
	   	const static double PI = 3.141592653589793238462;
		
		//private data members
		IplImage* sourceImage;			// The image we're working on
		unsigned int octaves;			// The desired number of octaves
		unsigned int intervals;			// The desired number of intervals
		unsigned int num_features;		// The number of keypoints detected

		IplImage***	scaleSpace;			// A 2D array to hold the different gaussian blurred images
		IplImage*** diffGaussian;		// A 2D array to hold the different DoG images
		double**	blurSigma;			// A 2D array to hold the sigma used to blur a particular image
		list<extremaPoint> extremaList;
		vector<Feature> features;		// Holds each keypoint's basic info
		vector<Descriptor> descriptors;	// Holds each keypoint's descriptor
		
		// private methods
		void initializeMemory();
		void BuildScaleSpace();
		void LocateExtremas();
		void OrientationAssgn();
		void AssgnDescriptors();
		
		//helper functions
		int getKernelSize(double sigma, double cut_off=0.001);
		CvMat* BuildInterpolatedGaussianTable(unsigned int size, double sigma);
		double gaussian2D(double x, double y, double sigma);
   	
	public:
		SIFTransform(string img, int octaves, int intervals);
		~SIFTransform();
		
		void displayFeatures();
		vector<Descriptor> returnDescriptors();
};
#endif