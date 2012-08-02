#include "include.h"
#include "DataBase.h"
#include "TestImage.h"

int main(int ints,char** chars){
    string testImgName=chars[1];
    TestImage tImg(testImgName);
    queue<float> testImgQ=tImg.imgCompQ();
    
    DataBase db("database/images.txt","database/sortedDistances/","database/images/",testImgQ);
    list<string> imgs=db.findSimilar();
    
    list<string>::iterator itr;
    system("rm -r -f temp/*");
    int i=1;
    for(itr=imgs.begin();itr!=imgs.end();itr++){
        char name[100];
        strcpy(name,"database/images/");
        strcat(name,&(*itr)[0]);
        
        char command[100];
        sprintf(command,"link %s temp/%d.png",name,i);
        i++;
        system(command);
    }
    system("nautilus temp");
    return 0;
}