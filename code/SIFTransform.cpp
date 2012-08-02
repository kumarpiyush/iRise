#include "SIFTransform.h"
//#include <cmath>
SIFTransform::SIFTransform(string img, int o, int i){
	IplImage* image=cvLoadImage(&img[0]);
	sourceImage = cvCloneImage(image);
	octaves = o;
	intervals = i;
	initializeMemory();
}


SIFTransform::~SIFTransform(){
	for(int i=0;i<octaves;i++){
		int j;
		// Release all images in octave i
		for(j=0;j<intervals+3;j++)	{cvReleaseImage(&scaleSpace[i][j]);}
		for(j=0;j<intervals+2;j++)	{cvReleaseImage(&diffGaussian[i][j]);}
		delete [] scaleSpace[i];
		delete [] diffGaussian[i];
		delete [] blurSigma[i];
	}
    delete [] scaleSpace;
    delete [] diffGaussian;
    delete [] blurSigma;
}


void SIFTransform::initializeMemory(){
	scaleSpace = new IplImage**[octaves];
	for(int i=0;i<octaves;i++)
		scaleSpace[i] = new IplImage*[intervals+3];

	diffGaussian = new IplImage**[octaves];
	for(int i=0;i<octaves;i++)
		diffGaussian[i] = new IplImage*[intervals+2];
		
	blurSigma = new double*[octaves];
	for(int i=0;i<octaves;i++)
		blurSigma[i] = new double[intervals+3];
}


// This function generates all the blurred out images for each octave and also the DoG images
void SIFTransform::BuildScaleSpace(){
	IplImage* imgGray = cvCreateImage(cvGetSize(sourceImage), IPL_DEPTH_32F , 1);
	IplImage* imgTemp = cvCreateImage(cvGetSize(sourceImage), 8 , 1);
	
	// If the image is colour, it is converted to grayscale
	if(sourceImage->nChannels==3){
		cvCvtColor(sourceImage, imgTemp, CV_BGR2GRAY);
	}
	else{
		cvCopy(sourceImage, imgTemp);
	}
	
	for(int x=0;x<imgTemp->width;x++){
		for(int y=0;y<imgTemp->height;y++){
			cvSetReal2D(imgGray, y, x, cvGetReal2D(imgTemp, y, x)/255.0); 				// to float image, divide each pixel by 255
		}
	}

	// smoothimg image, doubling it and again smoothing it increases stable keypoints
	cvSmooth(imgGray, imgGray, CV_GAUSSIAN, 0, 0, 0.5);
	scaleSpace[0][0] = cvCreateImage(cvSize(imgGray->width*2, imgGray->height*2), IPL_DEPTH_32F , 1);
	cvPyrUp(imgGray, scaleSpace[0][0]);
	cvSmooth(scaleSpace[0][0], scaleSpace[0][0], CV_GAUSSIAN, 0, 0, 1.0);

	double initSigma = sqrt(2.0);
	blurSigma[0][0] = initSigma * 0.5;

	for(int i=0;i<octaves;i++){
		double sigma = initSigma;
		CvSize currentSize = cvGetSize(scaleSpace[i][0]);

		for(int j=1;j<intervals+3;j++){
			scaleSpace[i][j] = cvCreateImage(currentSize, 32, 1);
			
			double sigma_f = sqrt(pow(2.0,2.0/intervals)-1)*sigma;						// sigma to be used for blur
            sigma = pow(2.0,1.0/intervals)*sigma;
			
			blurSigma[i][j] = sigma * 0.5 * pow(2.0f, (float)i);
			
			cvSmooth(scaleSpace[i][j-1], scaleSpace[i][j], CV_GAUSSIAN, 0, 0, sigma_f);	// create new image
			
			diffGaussian[i][j-1] = cvCreateImage(currentSize, 32, 1);					// actual image for feature detection
			cvSub(scaleSpace[i][j-1], scaleSpace[i][j], diffGaussian[i][j-1]);
		}
		
		if(i<octaves-1){
			currentSize.width/=2;
			currentSize.height/=2;														// reset size for next octave
			
			scaleSpace[i+1][0] = cvCreateImage(currentSize, 32, 1);
			cvPyrDown(scaleSpace[i][0], scaleSpace[i+1][0]);							// first element of new octave
			blurSigma[i+1][0] = blurSigma[i][intervals];								// blurSigma for next level
		}
	}
}


void SIFTransform::LocateExtremas(){
	double curvature_ratio, curvature_threshold;
	IplImage *middle, *up, *down;		// for extrema detection
	int scale;
	double dxx, dyy, dxy, trH, detH;

	unsigned int num=0;					// Number of keypoins detected
	unsigned int numRemoved=0;			// The number of key points rejected because they failed a test

	curvature_threshold = (CURVATURE_THRESHOLD+1)*(CURVATURE_THRESHOLD+1)/CURVATURE_THRESHOLD;

	for(int i=0;i<octaves;i++){
		scale = (int)pow(2.0, (double)i);
		
		for(int j=1;j<intervals+1;j++){

			// Images just above and below, in the current octave
			middle = diffGaussian[i][j];
			up = diffGaussian[i][j+1];
			down = diffGaussian[i][j-1];

			for(int x=1;x<diffGaussian[i][j]->width-1;x++){
			
				for(int y=1;y<diffGaussian[i][j]->height-1;y++){
					bool set = false;
					double pixelValue = cvGetReal2D(middle, y, x);

					// check for maximum
					if (pixelValue > cvGetReal2D(middle, y-1, x  )	&&
                        pixelValue > cvGetReal2D(middle, y+1, x  )  &&
                        pixelValue > cvGetReal2D(middle, y  , x-1)  &&
                        pixelValue > cvGetReal2D(middle, y  , x+1)  &&
                        pixelValue > cvGetReal2D(middle, y-1, x-1)	&&
                        pixelValue > cvGetReal2D(middle, y-1, x+1)	&&
                        pixelValue > cvGetReal2D(middle, y+1, x+1)	&&
                        pixelValue > cvGetReal2D(middle, y+1, x-1)	&&
                        pixelValue > cvGetReal2D(up, y  , x  )      &&
                        pixelValue > cvGetReal2D(up, y-1, x  )      &&
                        pixelValue > cvGetReal2D(up, y+1, x  )      &&
                        pixelValue > cvGetReal2D(up, y  , x-1)      &&
                        pixelValue > cvGetReal2D(up, y  , x+1)      &&
                        pixelValue > cvGetReal2D(up, y-1, x-1)		&&
                        pixelValue > cvGetReal2D(up, y-1, x+1)		&&
                        pixelValue > cvGetReal2D(up, y+1, x+1)		&&
                        pixelValue > cvGetReal2D(up, y+1, x-1)		&&
                        pixelValue > cvGetReal2D(down, y  , x  )    &&
                        pixelValue > cvGetReal2D(down, y-1, x  )    &&
                        pixelValue > cvGetReal2D(down, y+1, x  )    &&
                        pixelValue > cvGetReal2D(down, y  , x-1)    &&
                        pixelValue > cvGetReal2D(down, y  , x+1)    &&
                        pixelValue > cvGetReal2D(down, y-1, x-1)	&&
                        pixelValue > cvGetReal2D(down, y-1, x+1)	&&
                        pixelValue > cvGetReal2D(down, y+1, x+1)	&&
                        pixelValue > cvGetReal2D(down, y+1, x-1)   ){
							num++;
							set = true;
							extremaList.push_back(extremaPoint(i,j-1,y,x));
					}
					// check for minimum
					else if (pixelValue < cvGetReal2D(middle, y-1, x  )	&&
                        pixelValue < cvGetReal2D(middle, y+1, x  )  &&
                        pixelValue < cvGetReal2D(middle, y  , x-1)  &&
                        pixelValue < cvGetReal2D(middle, y  , x+1)  &&
                        pixelValue < cvGetReal2D(middle, y-1, x-1)	&&
                        pixelValue < cvGetReal2D(middle, y-1, x+1)	&&
                        pixelValue < cvGetReal2D(middle, y+1, x+1)	&&
                        pixelValue < cvGetReal2D(middle, y+1, x-1)	&&
                        pixelValue < cvGetReal2D(up, y  , x  )      &&
                        pixelValue < cvGetReal2D(up, y-1, x  )      &&
                        pixelValue < cvGetReal2D(up, y+1, x  )      &&
                        pixelValue < cvGetReal2D(up, y  , x-1)      &&
                        pixelValue < cvGetReal2D(up, y  , x+1)      &&
                        pixelValue < cvGetReal2D(up, y-1, x-1)		&&
                        pixelValue < cvGetReal2D(up, y-1, x+1)		&&
                        pixelValue < cvGetReal2D(up, y+1, x+1)		&&
                        pixelValue < cvGetReal2D(up, y+1, x-1)		&&
                        pixelValue < cvGetReal2D(down, y  , x  )    &&
                        pixelValue < cvGetReal2D(down, y-1, x  )    &&
                        pixelValue < cvGetReal2D(down, y+1, x  )    &&
                        pixelValue < cvGetReal2D(down, y  , x-1)    &&
                        pixelValue < cvGetReal2D(down, y  , x+1)    &&
                        pixelValue < cvGetReal2D(down, y-1, x-1)	&&
                        pixelValue < cvGetReal2D(down, y-1, x+1)	&&
                        pixelValue < cvGetReal2D(down, y+1, x+1)	&&
                        pixelValue < cvGetReal2D(down, y+1, x-1)   ){
							num++;
							set = true;
							extremaList.push_back(extremaPoint(i,j-1,y,x));
					}
					
					// contrast check
					if(set && fabs(cvGetReal2D(middle, y, x)) < CONTRAST_THRESHOLD){
						num--;
						numRemoved++;
						set=false;
						extremaList.pop_back();
					}

					// The edge check
					else if(set){
						dxx = (cvGetReal2D(middle, y-1, x) +
							  cvGetReal2D(middle, y+1, x) -
							  2.0*cvGetReal2D(middle, y, x));

						dyy = (cvGetReal2D(middle, y, x-1) +
							  cvGetReal2D(middle, y, x+1) -
							  2.0*cvGetReal2D(middle, y, x));

						dxy = (cvGetReal2D(middle, y-1, x-1) +
							  cvGetReal2D(middle, y+1, x+1) -
							  cvGetReal2D(middle, y+1, x-1) - 
							  cvGetReal2D(middle, y-1, x+1)) / 4.0;

						trH = dxx + dyy;
						detH = dxx*dyy - dxy*dxy;

						curvature_ratio = trH*trH/detH;
	
						if(detH<0 || curvature_ratio>curvature_threshold){
							num--;
							numRemoved++;
							set=false;
							extremaList.pop_back();
						}
					}
				}
			}
		}
	}

	num_features = num;
}


void SIFTransform::OrientationAssgn(){		// for all keypoints get orientation
	// These images hold the magnitude and direction of gradient 
	// for all blurred out images
	IplImage*** magnitude = new IplImage**[octaves];	// magnitudes of gradient for blurred images
	IplImage*** orientation = new IplImage**[octaves];	// orientations of gradient for blurred images

	for(int i=0;i<octaves;i++){
		magnitude[i] = new IplImage*[intervals];
		orientation[i] = new IplImage*[intervals];
	}	

	// These two loops are to calculate the magnitude and orientation of gradients
	for(int i=0;i<octaves;i++){
		for(int j=1;j<intervals+1;j++){
		
			magnitude[i][j-1] = cvCreateImage(cvGetSize(scaleSpace[i][j]), 32, 1);
			orientation[i][j-1] = cvCreateImage(cvGetSize(scaleSpace[i][j]), 32, 1);

			cvZero(magnitude[i][j-1]);
			cvZero(orientation[i][j-1]);
			
			for(int x=1;x<scaleSpace[i][j]->width-1;x++){
				for(int y=1;y<scaleSpace[i][j]->height-1;y++){
					double dx = cvGetReal2D(scaleSpace[i][j], y, x+1) - cvGetReal2D(scaleSpace[i][j], y, x-1);
					double dy = cvGetReal2D(scaleSpace[i][j], y+1, x) - cvGetReal2D(scaleSpace[i][j], y-1, x);
					
					cvSetReal2D(magnitude[i][j-1], y, x, sqrt(dx*dx + dy*dy));		// store magnitudes
					
					double ori=atan(dy/dx);					
					cvSet2D(orientation[i][j-1], y, x, cvScalar(ori));				// store magnitudes
				}
			}
		}
	}

	// The histogram with 8 bins
	double* hist_orient = new double[NUM_BINS];

	// Go through all octaves
	for(int i=0;i<octaves;i++){
		unsigned int scale = (unsigned int)pow(2.0, (double)i);
		unsigned int width = scaleSpace[i][0]->width;
		unsigned int height= scaleSpace[i][0]->height;

		// Go through all intervals in the current scale
		for(int j=1;j<intervals+1;j++){
			double abs_sigma = blurSigma[i][j];

			// This is used for magnitudes
			IplImage* imgWeight = cvCreateImage(cvSize(width, height), 32, 1);
			cvSmooth(magnitude[i][j-1], imgWeight, CV_GAUSSIAN, 0, 0, 1.5*abs_sigma);
			
			int neighbRange = getKernelSize(1.5*abs_sigma)/2;

			// Temporarily used to generate a mask of region used to calculate 
			// the orientations
			IplImage* imgMask = cvCreateImage(cvSize(width, height), 8, 1);
			cvZero(imgMask);

			for(int x=0;x<width;x++){
				for(int y=0;y<height;y++){
					if(extremaList.front().octave==i && extremaList.front().interval==j-1 && extremaList.front().y==y && extremaList.front().x==x){
						extremaList.pop_front();
						unsigned int k;
						for(k=0;k<NUM_BINS;k++){
							hist_orient[k]=0.0;
						}
						
						for(int kk=-neighbRange;kk<=neighbRange;kk++){
							for(int tt=-neighbRange;tt<=neighbRange;tt++){
								if(x+kk<0 || x+kk>=width || y+tt<0 || y+tt>=height){continue;}	// Ensure we're within the image

 								double sampleOrient = cvGetReal2D(orientation[i][j-1], y+tt, x+kk);

								if(sampleOrient <=-M_PI || sampleOrient>M_PI){
									printf("Bad Orientation: %f\n", sampleOrient);
								}
								
								sampleOrient+=M_PI;
								
								unsigned int sampleOrientDegrees = sampleOrient * 180 / M_PI;	// Convert to degrees
								hist_orient[(int)sampleOrientDegrees / (360/NUM_BINS)] += cvGetReal2D(imgWeight, y+tt, x+kk);
								cvSetReal2D(imgMask, y+tt, x+kk, 255);
							}
						}
						
						// now check for maxima
						double max_peak = hist_orient[0];
						unsigned int max_peak_index = 0;
						for(k=1;k<NUM_BINS;k++)
						{
							if(hist_orient[k]>max_peak)
							{
								max_peak=hist_orient[k];
								max_peak_index = k;
							}
						}

						// List of magnitudes and orientations at the current extrema
						vector<double> orien;
						vector<double> mag;
						for(k=0;k<NUM_BINS;k++){
							if(hist_orient[k]> 0.8*max_peak){
								double x1 = k-1;
								double y1;
								double x2 = k;
								double y2 = hist_orient[k];
								double x3 = k+1;
								double y3;

								if(k==0){
									y1 = hist_orient[NUM_BINS-1];
									y3 = hist_orient[1];
								}
								else if(k==NUM_BINS-1){
									y1 = hist_orient[NUM_BINS-1];
									y3 = hist_orient[0];
								}
								else{
									y1 = hist_orient[k-1];
									y3 = hist_orient[k+1];
								}
								/* now fit a parabola thru three points above
								  y=b . (x^2+x+1)	// b is 3-tuple
                                  b = inv(x) y
								*/

								double *b = new double[3];
								CvMat *X = cvCreateMat(3, 3, CV_32FC1);
								CvMat *matInv = cvCreateMat(3, 3, CV_32FC1);

								cvSetReal2D(X, 0, 0, x1*x1);
								cvSetReal2D(X, 1, 0, x1);
								cvSetReal2D(X, 2, 0, 1);

								cvSetReal2D(X, 0, 1, x2*x2);
								cvSetReal2D(X, 1, 1, x2);
								cvSetReal2D(X, 2, 1, 1);

								cvSetReal2D(X, 0, 2, x3*x3);
								cvSetReal2D(X, 1, 2, x3);
								cvSetReal2D(X, 2, 2, 1);

								// Invert the matrix
								cvInv(X, matInv);

								b[0] = cvGetReal2D(matInv, 0, 0)*y1 + cvGetReal2D(matInv, 1, 0)*y2 + cvGetReal2D(matInv, 2, 0)*y3;
								b[1] = cvGetReal2D(matInv, 0, 1)*y1 + cvGetReal2D(matInv, 1, 1)*y2 + cvGetReal2D(matInv, 2, 1)*y3;
								b[2] = cvGetReal2D(matInv, 0, 2)*y1 + cvGetReal2D(matInv, 1, 2)*y2 + cvGetReal2D(matInv, 2, 2)*y3;

								double x0 = -b[1]/(2*b[0]);
								
								if(fabs(x0)>2*NUM_BINS)
									x0=x2;

								while(x0<0)
									x0 += NUM_BINS;
								while(x0>= NUM_BINS)
									x0-= NUM_BINS;
									
								double x0_n = x0*(2*M_PI/NUM_BINS);	// normalize

								assert(x0_n>=0 && x0_n<2*M_PI);
								x0_n -= M_PI;
								assert(x0_n>=-M_PI && x0_n<M_PI);

								orien.push_back(x0_n);
								mag.push_back(hist_orient[k]);
							}
						}
						features.push_back(Feature(x*scale/2, y*scale/2, mag, orien, i*intervals+j-1));	// store the keypoint
					}
				}
			}
			cvReleaseImage(&imgMask);
		}
	}
	assert(features.size() == num_features);
	for(int i=0;i<octaves;i++){
		for(int j=1;j<intervals+1;j++){
			cvReleaseImage(&magnitude[i][j-1]);
			cvReleaseImage(&orientation[i][j-1]);
		}

		delete [] magnitude[i];
		delete [] orientation[i];
	}

	delete [] magnitude;
	delete [] orientation;
}


void SIFTransform::AssgnDescriptors(){
	IplImage*** imgInterpolatedMagnitude = new IplImage**[octaves];
	IplImage*** imgInterpolatedOrientation = new IplImage**[octaves];
	for(int i=0;i<octaves;i++){
		imgInterpolatedMagnitude[i] = new IplImage*[intervals];
		imgInterpolatedOrientation[i] = new IplImage*[intervals];
	}
	
	for(int i=0;i<octaves;i++){
		for(int j=1;j<intervals+1;j++){
			unsigned int width = scaleSpace[i][j]->width;
			unsigned int height =scaleSpace[i][j]->height;
			
			IplImage* imgTemp = cvCreateImage(cvSize(width*2, height*2), 32, 1);
			cvZero(imgTemp);
			
			cvPyrUp(scaleSpace[i][j], imgTemp);
			
			imgInterpolatedMagnitude[i][j-1] = cvCreateImage(cvSize(width+1, height+1), 32, 1);
			imgInterpolatedOrientation[i][j-1] = cvCreateImage(cvSize(width+1, height+1), 32, 1);
			cvZero(imgInterpolatedMagnitude[i][j-1]);
			cvZero(imgInterpolatedOrientation[i][j-1]);
			
			for(float xx=1.5;xx<width-1.5;xx++){
				for(float yy=1.5;yy<height-1.5;yy++){
					double dx = (cvGetReal2D(scaleSpace[i][j], yy, xx+1.5) + cvGetReal2D(scaleSpace[i][j], yy, xx+0.5))/2 - (cvGetReal2D(scaleSpace[i][j], yy, xx-1.5) + cvGetReal2D(scaleSpace[i][j], yy, xx-0.5))/2;
					double dy = (cvGetReal2D(scaleSpace[i][j], yy+1.5, xx) + cvGetReal2D(scaleSpace[i][j], yy+0.5, xx))/2 - (cvGetReal2D(scaleSpace[i][j], yy-1.5, xx) + cvGetReal2D(scaleSpace[i][j], yy-0.5, xx))/2;

					int xdash = xx+1;
					int ydash = yy+1;
					assert(xdash<=width && ydash<=height);

					// Set the magnitude and orientation
					cvSetReal2D(imgInterpolatedMagnitude[i][j-1], ydash, xdash, sqrt(dx*dx + dy*dy));
					cvSetReal2D(imgInterpolatedOrientation[i][j-1], ydash, xdash, (atan2(dy,dx)==M_PI)? -M_PI:atan2(dy,dx) );
				}
			}

			// Pad the edges with zeros
			for(int x=0;x<width+1;x++){
				cvSetReal2D(imgInterpolatedMagnitude[i][j-1], 0, x, 0);
				cvSetReal2D(imgInterpolatedMagnitude[i][j-1], height, x, 0);
				cvSetReal2D(imgInterpolatedOrientation[i][j-1], 0, x, 0);
				cvSetReal2D(imgInterpolatedOrientation[i][j-1], height, x, 0);
			}

			for(int y=0;y<height+1;y++){
				cvSetReal2D(imgInterpolatedMagnitude[i][j-1], y, 0, 0);
				cvSetReal2D(imgInterpolatedMagnitude[i][j-1], y, width, 0);
				cvSetReal2D(imgInterpolatedOrientation[i][j-1], y, 0, 0);
				cvSetReal2D(imgInterpolatedOrientation[i][j-1], y, width, 0);
			}
			cvReleaseImage(&imgTemp);

		}
	}

	/* Interpolated Gaussian Table of size FEATURE_WINDOW_SIZE
	 Lowe suggests sigma should be half the window size
	 This is used to construct the "circular gaussian window" to weight 
	 magnitudes
	*/
	CvMat *G = BuildInterpolatedGaussianTable(FEATURE_WINDOW_SIZE, 0.5*FEATURE_WINDOW_SIZE);
	
	vector<double> hist(DESC_NUM_BINS);
	
	for(int ikp = 0;ikp<num_features;ikp++){
		unsigned int scale = features[ikp].scale;
		float kpx = features[ikp].x;
		float kpy = features[ikp].y;

		float descx = kpx;
		float descy = kpy;

		unsigned int ii = (unsigned int)(kpx*2) / (unsigned int)(pow(2.0, (double)scale/intervals));
		unsigned int jj = (unsigned int)(kpy*2) / (unsigned int)(pow(2.0, (double)scale/intervals));

		unsigned int width = scaleSpace[scale/intervals][0]->width;
		unsigned int height = scaleSpace[scale/intervals][0]->height;

		vector<double> orien = features[ikp].orien;
		vector<double> mag = features[ikp].mag;

		// Find the orientation and magnitude that have the maximum impact on the feature
		double main_mag = mag[0];
		double main_orien = orien[0];
		for(unsigned int orient_count=1;orient_count<mag.size();orient_count++)
		{
			if(mag[orient_count]>main_mag)
			{
				main_orien = orien[orient_count];
				main_mag = mag[orient_count];
			}
		}

		unsigned int neighbRange = FEATURE_WINDOW_SIZE/2;
		CvMat *weight = cvCreateMat(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE, CV_32FC1);
		vector<double> fv(FVSIZE);

		for(int i=0;i<FEATURE_WINDOW_SIZE;i++)
		{
			for(int j=0;j<FEATURE_WINDOW_SIZE;j++)
			{
				if(ii+i+1<neighbRange || ii+i+1>width+neighbRange || jj+j+1<neighbRange || jj+j+1>height+neighbRange){
                    cvSetReal2D(weight, j, i, 0);
                }
				else{
					cvSetReal2D(weight, j, i, cvGetReal2D(G, j, i)*cvGetReal2D(imgInterpolatedMagnitude[scale/intervals][scale%intervals], jj+j+1-neighbRange, ii+i+1-neighbRange));}
			}
		}
		
		for(int i=0;i<FEATURE_WINDOW_SIZE/4;i++){	// These loops are for splitting the 16x16 window into sixteen 4x4 blocks
		
			for(int j=0;j<FEATURE_WINDOW_SIZE/4;j++){
				for(int t=0;t<DESC_NUM_BINS;t++){
					hist[t]=0.0;
				}
				int starti = (int)ii - (int)neighbRange + 1 + (int)(neighbRange/2*i);
				int startj = (int)jj - (int)neighbRange + 1 + (int)(neighbRange/2*j);
				int limiti = (int)ii + (int)(neighbRange/2)*((int)(i)-1);
				int limitj = (int)jj + (int)(neighbRange/2)*((int)(j)-1);
				
				for(int k=starti;k<=limiti;k++){
				
					for(int t=startj;t<=limitj;t++){
					
						if(k<0 || k>width || t<0 || t>height)
							continue;

						// This is where rotation invariance is done
						double sample_orien = cvGetReal2D(imgInterpolatedOrientation[scale/intervals][scale%intervals], t, k);
						sample_orien -= main_orien;

						while(sample_orien<0)
							sample_orien+=2*M_PI;

						while(sample_orien>2*M_PI)
							sample_orien-=2*M_PI;

						// This should never happen
						if(!(sample_orien>=0 && sample_orien<2*M_PI))
							printf("BAD: %f\n", sample_orien);
						assert(sample_orien>=0 && sample_orien<2*M_PI);

						unsigned int sample_orien_d = sample_orien*180/M_PI;
						assert(sample_orien_d<360);

						unsigned int bin = sample_orien_d/(360/DESC_NUM_BINS);					// The bin
						double bin_f = (double)sample_orien_d/(double)(360/DESC_NUM_BINS);		// The actual entry

						assert(bin<DESC_NUM_BINS);
						assert(k+neighbRange-1-ii<FEATURE_WINDOW_SIZE && t+neighbRange-1-jj<FEATURE_WINDOW_SIZE);

						// Add to the bin
						hist[bin]+=(1-fabs(bin_f-(bin+0.5))) * cvGetReal2D(weight, t+neighbRange-1-jj, k+neighbRange-1-ii);
					}
				}

				// Keep adding these numbers to the feature vector
				for(int t=0;t<DESC_NUM_BINS;t++){
					fv[(i*FEATURE_WINDOW_SIZE/4+j)*DESC_NUM_BINS+t] = hist[t];
				}
			}
		}

		//Normalize the feature vector for illumination independence
		double norm=0;
		
		for(int t=0;t<FVSIZE;t++){
			norm+=pow(fv[t], 2.0);
		}
			
		norm = sqrt(norm);

		for(int t=0;t<FVSIZE;t++){
			fv[t]/=norm;
		}

		// Now, threshold the vector
		for(int t=0;t<FVSIZE;t++){
			if(fv[t]>FV_THRESHOLD){
				fv[t] = FV_THRESHOLD;
			}
		}
		norm=0;
		for(int t=0;t<FVSIZE;t++){
			norm+=pow(fv[t], 2.0);
		}
		norm = sqrt(norm);
		for(int t=0;t<FVSIZE;t++){
			fv[t]/=norm;
		}

		//Store the descriptor into the descriptors vector
		descriptors.push_back(Descriptor(descx, descy, fv));
	}

	assert(descriptors.size()==num_features);
	
	for(int i=0;i<octaves;i++){
		for(int j=1;j<intervals+1;j++){
			cvReleaseImage(&imgInterpolatedMagnitude[i][j-1]);
			cvReleaseImage(&imgInterpolatedOrientation[i][j-1]);
		}

		delete [] imgInterpolatedMagnitude[i];
		delete [] imgInterpolatedOrientation[i];
	}

	delete [] imgInterpolatedMagnitude;
	delete [] imgInterpolatedOrientation;
}

int SIFTransform::getKernelSize(double sigma, double cut_off){		// Returns the size of the kernal for the Gaussian blur given the sigma and
																	// cutoff value
    int i;
    for (i=0;i<MAX_KERNEL_SIZE;i++)
        if (exp(-((double)(i*i))/(2.0*sigma*sigma))<cut_off)
            break;
    unsigned int size = 2*i-1;
    return size;
}

CvMat* SIFTransform::BuildInterpolatedGaussianTable(unsigned int size, double sigma){	// This function actually generates the bell curve like 
																						//image for the weighted addition earlier.
	double kernel_half_size = size/2 - 0.5;

	double sog=0;
	CvMat* retMat = cvCreateMat(size, size, CV_32FC1);

	assert(size%2==0);

	double temp=0;
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			temp = gaussian2D(i-kernel_half_size, j-kernel_half_size, sigma);
			cvSetReal2D(retMat, j, i, temp);
			sog+=temp;
		}
	}

	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++){
			cvSetReal2D(retMat, j, i, 1.0/sog * cvGetReal2D(retMat, j, i));
		}
	}

	return retMat;
}


double SIFTransform::gaussian2D(double x, double y, double sigma){
	double ret = 1.0/(2*PI*sigma*sigma) * exp(-(x*x+y*y)/(2.0*sigma*sigma));
	return ret;
}


void SIFTransform::displayFeatures(){			// debugger :)
	IplImage* img = cvCloneImage(sourceImage);
	for(int i=0;i<num_features;i++){
		Feature ft = features[i];
		cvLine(img, cvPoint(ft.x, ft.y), cvPoint(ft.x, ft.y), CV_RGB(255,0,0), 3);
		cvLine(img, cvPoint(ft.x, ft.y), cvPoint(ft.x+10*cos(ft.orien[0]), ft.y+10*sin((double)ft.orien[0])), CV_RGB(255,255,255), 1);
	}
}


vector<Descriptor> SIFTransform::returnDescriptors(){
	BuildScaleSpace();
	LocateExtremas();
	OrientationAssgn();
	AssgnDescriptors();
	return descriptors;
}
