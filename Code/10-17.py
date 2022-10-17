import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
# %matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio.v2 as imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil

images = []
for index, file in enumerate(glob.glob('../CHSTrafficCapstone/data/images/train/*.jpg')):
    images.append(imageio.imread(file))
    
# how many images we have
print('We have {} images'.format(len(images)))

# what are the sizes of the images
for index, file in enumerate(glob.glob('../CHSTrafficCapstone/data/images/train/*.jpg')):
    print('Image {} have size of {}'.format(file[7:], images[index].shape))

# XML file names correspond to the image file names
for index, file in enumerate(glob.glob('../CHSTrafficCapstone/data/images/train/*.xml')):
    print(file[7:])
    
shutil.copy('../CHSTrafficCapstone/data/images/train/1.xml', '../CHSTrafficCapstone/data/images/train/im1_10-17.txt')
annotation_text = open("../CHSTrafficCapstone/data/images/train/im1_10-17.txt", "r")
print(annotation_text.read())
annotation_text.close() 

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '../CHSTrafficCapstone/data/images/train/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# apply xml_to_csv() function to convert all XML files in images/ folder into labels.csv
labels_df = xml_to_csv('../CHSTrafficCapstone/data/images/train/')
labels_df.to_csv(('labels2.csv'), index=None)
print('Successfully converted xml to csv.')

#in previuos cell we also put all annotation in labels_df
# let's see what's inside
# each bounding box has a separate row
# 2 pictures have two red pandas in it
labels_df