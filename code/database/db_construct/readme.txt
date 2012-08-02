this is to convert images to .png for storing in database (note that .gif images showed errors, and also for images with spaces in names :-/ )
HowTo:
convert.cpp makes convert.py to make a file of all images in images/ folder, and then puts .png version of them in imagesout. folder.
Just run ./convert

mysiftcontroller runs MySIFT on images in imagesout and puts their unsorted list of intermediate keypoint Eucledian distances in vectors/ folder.
Note that this was done on more of a manual method, so mysiftcontroller.py should be used at your own risk, unless you can modify it :)
