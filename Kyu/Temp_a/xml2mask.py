import os
from time import time
import numpy as np
from openslide import OpenSlide
import cv2
from readxml import readxml
from PIL import Image

# use xml to
def xml2mask(svs_src,fn,dst):
    Image.MAX_IMAGE_PIXELS = 999999999
    defstart = time()
    # make dst
    if not os.path.exists(dst): os.mkdir(dst)
    # open svs and xml
    svs_path = os.path.join(svs_src,fn+'.svs')
    xml_path = os.path.join(svs_src,fn+'.xml')
    if os.path.exists(svs_path): svs = OpenSlide(svs_path)
    if os.path.exists(xml_path): xml = readxml(xml_path)
    else:
        try: xml = readxml(os.path.join(*[svs_src,'binary annotation',fn+'.xml']))
        except: print('check xml exists')

    # find which layer is collagen
    obj_count_per_class = [len(np.unique(xml['objID'][xml['classID']==classid])) for classid in np.unique(xml['classID'])]
    # first element in this list is collagen index
    sort_index = np.argsort(obj_count_per_class)

    # find reticular dermis of each section
    start = time()
    mask = np.zeros(svs.level_dimensions[-1][::-1])
    for classidx, classid in enumerate(np.unique(xml['classID'])[sort_index]):
        for objidx, objid in enumerate(np.unique(xml['objID'])):
            if os.path.exists(os.path.join(dst, fn + '_tissue_roi_{:d}.tif'.format(objidx))): break
            object = xml[xml['classID']==classid]
            object = object[xml['objID']==objid]
            if len(object)<1: continue
            x = object['x']/svs.level_downsamples[-1]
            y = object['y']/svs.level_downsamples[-1]
            vertex = np.array([[i,j] for i,j in zip(x,y)]).astype(np.int32)
            if classidx == 0: intensity = 255
            else: intensity = 0
            cv2.fillPoly(mask,[vertex],color=intensity)
    print("mask generation: {:.2f} sec elapsed".format(time()-start))

    # save region with actual tissue section
    if not os.path.exists(os.path.join(dst,fn+'_tissue_mask.tif')):
        mask_img = Image.fromarray(mask).convert('1')
        mask_img.save(os.path.join(dst,fn+'_tissue_mask.tif'), compression=None)

    # iterate each collagen (which is supposedly entire reticular dermis of a section)
    xml_col = xml[xml['classID']==np.unique(xml['classID'])[sort_index][0]]
    region_arrs=[]
    for objidx, objid in enumerate(np.unique(xml_col['objID'])):
        if os.path.exists(os.path.join(dst,fn+'_tissue_roi_{:d}.tif'.format(objidx))):
            print('Saved ROI tiff file already exists. This file will be loaded')
            region_arr = Image.open(os.path.join(dst,fn+'_tissue_roi_{:d}.tif'.format(objidx)))
            region_arr = np.array(region_arr)
            region_arrs.append(region_arr)
            continue

        object = xml_col[xml_col['objID']==objid]
        x = object['x']
        y = object['y']
        [xmin,xmax,ymin,ymax] = [np.min(x),np.max(x),np.min(y),np.max(y)]
        [width,height] = [xmax-xmin,ymax-ymin]
        start=time()
        region = svs.read_region(location=(xmin,ymin),level=0,size=(width,height)).convert("RGB")
        region.save(os.path.join(dst,fn+'_tissue_region_{:d}.tif'.format(objidx)))
        print("openslide: {:.2f} sec elapsed".format(time()-start))

        start=time()
        # define bbox at lowest magnification
        [xmin_small,xmax_small,ymin_small,ymax_small] = [np.around(_/svs.level_downsamples[-1]).astype(int)
                                                         for _ in [xmin,xmax,ymin,ymax]]
        # crop mask that is already made for lowest magnification size
        region_mask = mask[ymin_small:ymax_small,xmin_small:xmax_small]
        # size up mask to highest magnification
        region_mask = cv2.resize(region_mask, dsize=region.size, interpolation=cv2.INTER_NEAREST)
        ret,region_mask = cv2.threshold(region_mask,0.5,1,cv2.THRESH_BINARY)
        region_mask = np.repeat(region_mask[:, :, np.newaxis], 3, axis=2)
        # apply the mask
        region_arr = np.array(region).astype(np.uint8)
        region_arr[region_mask==0] = 255 #numpy way of masking with condition
        region_arrs.append(region_arr)
        # save the masked roi
        region_arr_img = Image.fromarray(region_arr)
        region_arr_img.save(os.path.join(dst,fn+'_tissue_roi_{:d}.tif'.format(objidx)))
        print("removing non-collagen region: {:.2f} sec elapsed".format(time()-start))
    print("xml2mask: {:.2f} sec elapsed \n".format(time() - defstart))
    return region_arrs

