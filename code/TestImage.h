/*
 takes an image from the user, gets its features, and returns them for comparison
*/

#ifndef TestImage_h
#define TestImage_h

#include "include.h"
#include "Feature.h"
#include "Descriptor.h"
#include "SIFTransform.h"

class TestImage{
    private:
        string	         imgName;
        SIFTransform*    sift;
        const static int numOctaves   = 4;
        const static int numIntervals = 2;
        
    public:
        TestImage(string image);
        queue<float> imgCompQ();    // the comparator queue for this image
        ~TestImage();
};

#endif
