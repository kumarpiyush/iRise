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
    system("python thumbnails.py");
    FILE* alpha;
    alpha=fopen("names","r");
    
    char name[100],name1[100];
    int ind=1;
    while(!feof(alpha)){
        fscanf(alpha,"%s\n",name);
        cout<<name<<endl;
        IplImage* img=cvLoadImage(name);
        float ratio=300.0/img->height;
        IplImage* img1=cvCreateImage(Size(img->width*ratio,300.0),8,3);
        cvResize(img,img1,CV_INTER_AREA);
        sprintf(name1,"../thumbnails/%d.png",ind);
        cvSaveImage(name1,img1);
        ind++;
    }
    fclose(alpha);
    system("rm names");
    return 0;
}
