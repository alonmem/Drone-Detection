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
my_xml = open('one/output/example.xml', "r")

for f in os.listdir('one/output/to_train'):
    if f.endswith('.BMP'):
        img = Image.open('one/output/to_train/'+f)
        pixels = img.load()

        a=[]
        for i in range(img.size[0]):    # for every col
            for j in range(img.size[1]):    # For every row
                if pixels[i,j][0]>190 and pixels[i,j][1]>190 and pixels[i,j][2]<110:
                    c+=1
                    a.append([int(i),int(j)])

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
                    continue
                
            d = int(abs(b[0][0]-b[1][0])) * int(abs(b[0][1]-b[1][1]))
            if (d>15) :
                xmin = str(b[0][0])
                ymin = str(b[0][1])
                xmax = str(b[1][0])
                ymax = str(b[1][1])
            else:
                continue
        else:
            continue
        c=0
    
        tree = ET.parse('one/output/example.xml')
        
        tree.find('size').find('width').text = str(img.size[0])
        tree.find('size').find('height').text = str(img.size[1])
        
        tree.find('object').find('bndbox').find('xmin').text = xmin
        tree.find('object').find('bndbox').find('xmax').text = xmax
        tree.find('object').find('bndbox').find('ymin').text = ymin
        tree.find('object').find('bndbox').find('ymax').text = ymax
        tree.find('filename').text = str(os.path.splitext(basename(f))[0])
        
        tree.write("one/output/to_train/"+os.path.splitext(basename(f))[0]+'.xml')

