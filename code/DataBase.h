// this is the controller class
// takes in test image, compares it with image signature it imports, and shows found images

#ifndef DataBase_h
#define DataBase_h

#include "include.h"
#define ACCURACY 0.1
class DataBase{
	private:
		string namesOfImages;
        string directory_path_sorDists;
        string directory_path_images;
        queue<float> test_queue;
        
		// private methods
		bool isRight(int numMatched, int numTest, int numFound);
		int numberMatched(queue<float> q1, queue<float> q2);
        
    public:
        DataBase(string noi, string dirSorDists, string dirImg, queue<float> givenQ);
        list<string> findSimilar();    				// searches database and returns images
};
#endif