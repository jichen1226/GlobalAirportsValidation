## Annotation
Annotation file put in this folder.  
More information about the format [here](https://github.com/qqwweee/keras-yolo3).   

### Format
One row for one image;  
Row format: image_file_path box1 box2 ... boxN;  
Box format: x_min,y_min,x_max,y_max,class_id (no space).  
  
Example:
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3  
    path/to/img2.jpg 120,300,250,600,2  
    ...  
