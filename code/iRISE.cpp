#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Shared_Image.H>
#include <FL/Fl_Image.H>
#include <Fl/Fl_PNG_Image.H>
#include <Fl/Fl_Button.H>
#include <Fl/Fl_File_Chooser.H>
#include <Fl/Fl_Input.H>
#include <iostream>
#include <cstring>
using namespace std;

Fl_Window win(1000,650,"iRISE");
Fl_Input* inp;
Fl_PNG_Image* img;
Fl_Box* box;

void changeImage(Fl_Widget* w,void* data){
    char command[100]={"./main "};
    strcat(command,inp->value());
    system(command);
}

int main(){
    box=new Fl_Box(0,0,1000,650);
    
    img=new Fl_PNG_Image("LOGO.png");
    box->image(img);
    
    inp=new Fl_Input(450,550,100,30,"Enter path:");
    Fl_Button but(450,600,100,30,"Enter image");
    but.callback(changeImage);
    
    win.show();
    return(Fl::run());
}