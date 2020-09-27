import os
from time import time
import numpy as np
from openslide import OpenSlide
import cv2
from readxml import readxml
from PIL import Image

def xml2mask_v1(svs_path,xml_path,mag=5):
    Image.MAX_IMAGE_PIXELS = 999999999
    defstart = time()
    # open svs and xml
    fn = os.path.split(xml_path)[-1]
    fn = os.path.splitext(fn)[0]
    svs_src = os.path.split(svs_path)[0]
    xml_src = os.path.split(xml_path)[0]
    if os.path.exists(svs_path): svs = OpenSlide(svs_path)
    if os.path.exists(xml_path): xml = readxml(xml_path)
    else:
        try: xml = readxml(os.path.join(*[xml_src,'binary annotation',fn+'.xml']))
        except:
            print(os.path.join(*[xml_src,'binary annotation',fn+'.xml']))
            print('check xml exists')

    # find which layer is collagen
    obj_count_per_class = [len(np.unique(xml['objID'][xml['classID']==classid])) for classid in np.unique(xml['classID'])]
    # first element in this list is collagen index
    sort_index = np.argsort(obj_count_per_class)

    # find reticular dermis of each section
    start = time()
    maglist = np.array([20, 5, 1])
    magidx = np.argwhere(maglist == mag)[0][0]
    mask = np.zeros(svs.level_dimensions[magidx][::-1])
    for classidx, classid in enumerate(np.unique(xml['classID'])[sort_index]):
        for objidx, objid in enumerate(np.unique(xml['objID'])):
            object = xml[xml['classID']==classid]
            object = object[xml['objID']==objid]
            if len(object)<1: continue
            # x = object['x']
            # y = object['y']
            x = object['x']/svs.level_downsamples[magidx]
            y = object['y']/svs.level_downsamples[magidx]
            vertex = np.array([[i,j] for i,j in zip(x,y)]).astype(np.int32)
            if classidx == 0: intensity = 255
            else: intensity = 0
            cv2.fillPoly(mask,[vertex],color=intensity)
    print("mask generation: {:.2f} sec elapsed".format(time()-start))
    # mask = cv2.resize(mask, dsize=svs.level_dimensions[0], interpolation=cv2.INTER_CUBIC)
    # mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    return mask
