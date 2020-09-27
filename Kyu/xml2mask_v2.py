import os
from time import time
import numpy as np
from openslide import OpenSlide
import cv2
from readxml import readxml
from PIL import Image

def xml2mask_v2(svs_path,xml_path,mag=5,sort_index=[0,1,9,3,2,4,5,6,7,8,10,11],binary=False):
    defstart = time()
    Image.MAX_IMAGE_PIXELS = 999999999
    # split path
    xml_src, xmlfn = os.path.split(xml_path)
    svs_src, svsfn = os.path.split(svs_path)
    fn = os.path.splitext(xmlfn)[0]

    # open svs and xml
    if os.path.exists(svs_path): svs = OpenSlide(svs_path)
    if os.path.exists(xml_path): xml = readxml(xml_path)
    else:
        try: xml = readxml(os.path.join(*[xml_src,'multiclass annotation',fn+'.xml']))
        except:
            print(os.path.join(*[xml_src,'multiclass annotation',fn+'.xml']))
            print('check xml exists')

    if binary:
        obj_count_per_class = [len(np.unique(xml['objID'][xml['classID'] == classid])) for classid in
                               np.unique(xml['classID'])]
        sort_index = np.argsort(obj_count_per_class)

    # find reticular dermis of each section
    maglist = np.array([20, 5, 1])
    magidx = np.argwhere(maglist == mag)[0][0]
    mask = np.zeros(svs.level_dimensions[magidx][::-1])
    for classid in sort_index:
        classid = classid+1
        if binary:
            if classid == sort_index[0]+1:
                intensity = 10
            else:
                intensity = 0
        else:
            intensity = classid

        try: object = xml[xml['classID'] == classid]
        except: print('xml does not exist',xml_path)

        for objid in np.unique(object['objID']):
            object2 = object[object['objID']==objid]
            if len(object2)<1:
                print('objID {:d} skip'.format(objid))
                continue
            x = object2['x']/svs.level_downsamples[magidx]
            y = object2['y']/svs.level_downsamples[magidx]
            vertex = np.array([[i,j] for i,j in zip(x,y)]).astype(np.int32)
            cv2.fillPoly(mask,[vertex],color=intensity)
    print("mask generation: {:.2f} sec elapsed".format(time()-defstart))
    return mask
