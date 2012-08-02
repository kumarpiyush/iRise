#include <iostream>
#include <cstdio>
#include <queue>
#include <cstring>

using namespace std;


int main(){
    int i=56;
    while(i-- && i!=0){
        char file_name[100];
        sprintf(file_name,"distances/%d.txt\0",i);
        cout<<file_name<<endl;
        FILE* alpha= fopen(file_name,"r");
       
        if(alpha==NULL){
            cout<<"alpha NULL"<<endl;
             return 1;
        }
       
        char output[100];
        sprintf(output,"sortedDistances/%d.txt\0",i);
       
        FILE* beta = fopen(output,"w");
       
        if(beta==NULL){
            cout<<"beta NULL"<<endl;
             return 1;
        }
       
        priority_queue<float,vector<float>, greater<float> > pq;
       
        while(!feof(alpha)){
            float value;
            fscanf(alpha, "%f\n", &value);
            pq.push(value);
        }
        //queue<float> q;
        while(!pq.empty()){
            float val=pq.top();
            fprintf(beta,"%f\n",val);
            //q.push(pq.top());
            pq.pop();
        }
        fclose(alpha);
        fclose(beta);
    }
    return 0;
}
