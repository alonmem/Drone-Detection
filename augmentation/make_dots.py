import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import os
import xml.etree.ElementTree as ET
from os.path import basename
import itertools


ymin = 0
xmin = 0
xmax = 0
ymax = 0
c=0
cc=0

for f in os.listdir('one/output'):
    if f.endswith('.BMP'):

        img = Image.open('one/output/'+f)
        pixels = img.load()
        
        
        #my_file = open(str('images_with_dots/'+f), "r")
        tree = ET.parse('one/output/'+os.path.splitext(basename(f))[0]+'.xml')
        xmin= int(tree.find('object').find('bndbox').find('xmin').text)
        ymin= int(tree.find('object').find('bndbox').find('ymin').text)
        xmax= int(tree.find('object').find('bndbox').find('xmax').text)
        ymax= int(tree.find('object').find('bndbox').find('ymax').text)
        
        draw = ImageDraw.Draw(img)
        draw.rectangle(((xmin,ymin), (xmax,ymax)))
        img.save("one/output/what/"+os.path.splitext(basename(f))[0]+'.BMP')



        
       
        '''
        #img.save("one/what/"+f)
        
        

        
        draw = ImageDraw.Draw(img)

        
        a = []
        
        
        for i in range(img.size[0]):    # for every col
            for j in range(img.size[1]):    # For every row
                if pixels[i,j][0]>190 and pixels[i,j][1]>190 and pixels[i,j][2]<110:
                    c+=1
                    a.append([int(i),int(j)])
                    #pixels[i,j] = (0,0,0)

        #pixels[xmin,ymin] = (255, 255, 0)
        #pixels[xmax,ymax] = (255, 255, 0)
                    
        max_distance=0
        b = []
        
        if c>1:
            for i in a:
                for j in a:
                    if i==j:
                        continue
                    distance = int(abs(i[0]-j[0])) * int(abs(i[1]-j[1]))
                    if distance>max_distance:
                        max_distance = distance
                        b = []
                        b.append(i)
                        b.append(j)
                    
            if len(b)==0:
                if(len(a)==2):
                    b=a
                else:
                    break

            d = int(abs(b[0][0]-b[1][0])) * int(abs(b[0][1]-b[1][1]))
            if (d>15) :
                draw.rectangle(((b[0][0],b[0][1]), (b[1][0],b[1][1])))
                img.save("one/output/what/"+os.path.splitext(basename(f))[0]+'.BMP')

                
                                            
            #draw.rectangle(((b[0][0],b[0][1]), (b[1][0],b[1][1])))
        c=0
        '''

        



        
     
