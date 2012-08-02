void copyToROI(Mat& canvas,Mat& img,Rect r){
    if(canvas.type()!=img.type()){
        // only problem is when canvas is type 16 and img of type 0
        Mat img2;
        cvtColor(img, img2, CV_GRAY2RGB);
        
        float scaleX=(float)img2.cols/(float)(r.width-r.x);
        float scaleY=(float)img2.rows/(float)(r.height-r.y);
        
        for(int i=r.x;i<r.width;i++){
            for(int j=r.y;j<r.height;j++){
                canvas.at<Vec3b>(j,i)=img2.at<Vec3b>((float)(j-r.y)*scaleY,(float)(i-r.x)*scaleX);
            }
        }
    }
    else{
    
        float scaleX=(float)img.cols/(float)(r.width-r.x);
        float scaleY=(float)img.rows/(float)(r.height-r.y);
        
        for(int i=r.x;i<r.width;i++){
            for(int j=r.y;j<r.height;j++){
                canvas.at<Vec3b>(j,i)=img.at<Vec3b>((float)(j-r.y)*scaleY,(float)(i-r.x)*scaleX);
            }
        }
    }
}
