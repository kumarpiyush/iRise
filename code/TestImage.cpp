#include "TestImage.h"
#include "SIFTransform.h"


TestImage::TestImage(string name){
	sift = new SIFTransform(name, numOctaves, numIntervals);
}

queue<float> TestImage::imgCompQ(){
	vector<Descriptor> vec= sift->returnDescriptors();
	int iMax=vec.size();
	
	FILE* alpha;
	alpha=fopen("out1","w");
	for(int i=0;i<iMax;i++){
		for(int j=0;j<128;j++){		// loopong over descriptor
			fprintf(alpha,"%f ",(vec[i].fv)[j]);
		}
		fputc('\n',alpha);
	}
	fclose(alpha);
	system("python distances.py");
	
	priority_queue<float,vector<float>, greater<float> > pq;
	alpha=fopen("out2","r");
	while(!feof(alpha)){
		float value;
		fscanf(alpha, "%f\n", &value);
		pq.push(value);
	}
	fclose(alpha);
	system("rm out1 out2");		// useless now
	
	queue<float> comparator_queue;
	while(pq.size()!=0){
		comparator_queue.push(pq.top());
		pq.pop();
	}
	return comparator_queue;
}

TestImage::~TestImage(){
	delete sift;
}