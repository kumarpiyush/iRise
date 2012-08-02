#include "DataBase.h"

DataBase::DataBase(string noi,string dirSorDists, string dirImg, queue<float> givenQ){
     namesOfImages=noi;
	 directory_path_sorDists=dirSorDists;
     directory_path_images=dirImg;
     test_queue=givenQ;
}

int compare(float a, float b){		//returns 0 if equal, 1 if a is larger, 2 if b is larger
	int i1=(int)(a/ACCURACY);
	int i2=(int)(b/ACCURACY);
	if(i1==i2) return 0;
	else if(i1>i2) return 1;
	else return 2;
}

bool DataBase::isRight(int nMatches, int testQSz, int currImgSz){
     //cout<<nMatches<<"  "<<testQSz<<"  "<<currImgSz<<endl;
     int min=testQSz;
     if(testQSz>currImgSz){min=currImgSz;}
     if(((float)nMatches/(float)min) > 0.6){
          return 1;
     }
     return 0;
}

int DataBase::numberMatched(queue<float> q1, queue<float> q2){ //do not change to queue<float>& since the queues are changed
	int count=0;
	while((!q1.empty()) && (!q2.empty())){
		float e1=q1.front();
		float e2=q2.front();
		int c=compare(e1,e2);
		if(c==0){
			count++;
			q1.pop();
			q2.pop();
		}
		else if(c==1){
			q2.pop();
		}
		else{
			q1.pop();
		}
	}
	return count;
}

list<string> DataBase::findSimilar(){
    list<string> imageList;
    FILE* imageNames=fopen(&namesOfImages[0],"r");
    while(!feof(imageNames)){
        char currentImage[100];
        fscanf(imageNames,"%s\n",currentImage);
        
        char currentImage1[100];
        strcpy(currentImage1,&directory_path_sorDists[0]);
        strcat(currentImage1,currentImage);
        
        int len=strlen(currentImage1);
        
        currentImage1[len-1]='t';
        currentImage1[len-2]='x';
        currentImage1[len-3]='t';
        //cout<<currentImage1<<endl;
        
        FILE* cImage=fopen(currentImage1,"r");
        queue<float> q_current_image;
        while(!feof(cImage)){
            float value;
            fscanf(cImage, "%f\n", &value);
            q_current_image.push(value);
        }
        fclose(cImage);
        int num_matched=numberMatched(test_queue, q_current_image);
        if(isRight(num_matched, test_queue.size(), q_current_image.size())){
            imageList.push_back(currentImage);
        }
    }
    fclose(imageNames);
    return imageList;
}