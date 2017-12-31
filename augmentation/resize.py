import os
from resizeimage import resizeimage
from PIL import Image

for img in os.listdir('one'):
    with open(str('one/'+img), 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [640, 480])
            cover.save(str('one/'+img), image.format)
    
