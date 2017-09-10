#!/usr/bin/python
from PIL import Image
import os, sys

path = "~/Desktop/test"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = resizeimage.resize_cover(img, [299, 299])
            imResize.save(f + ' resized.png', 'PNG', quality=100)

resize()



# from PIL import Image
# from resizeimage import resizeimage

# fd_img = open('test-image.jpeg', 'r')
# img = Image.open(fd_img)
# img = resizeimage.resize_cover(img, [200, 100])
# img.save('test-image-cover.jpeg', img.format)
# fd_img.close()