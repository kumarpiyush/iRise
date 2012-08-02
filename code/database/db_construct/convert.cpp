#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <queue>
#include <vector>
#include <utility>
#include <list>
#include <cv.h>
#include <highgui.h>
#include <cxtypes.h>
using namespace cv;
using namespace std;

int main(){
    system("python convert.py");
    FILE* alpha;
    alpha=fopen("names","r");
    
    char name[100],name1[100];
    int ind=1;
    while(!feof(alpha)){
        fscanf(alpha,"%s\n",name);
        cout<<name<<endl;
        IplImage* img=cvLoadImage(name);
        sprintf(name1,"imagesDone/%d.png",ind);
        cvSaveImage(name1,img);
        ind++;
    }
    fclose(alpha);
    system("rm names");
    return 0;
}
