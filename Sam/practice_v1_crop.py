import os
# from os.path import isfile, join
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = 999999999

#for every parameter spreadsheet, find accompanying tissue image
src = r'\\motherserverdw\Kyu_Sync\server for undergrads\SAM\fibroblast_detection_v1'
csvs = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and "parameter" in f]

#iterate through each spread sheet find the centroid of each fibroblast
#open image and use centroid coordinate to make tiles
for x in csvs:
    csv = pd.read_csv(os.path.join(src, x))
    x_cent = csv['x']
    y_cent = csv['y']
    img = x.replace("parameters", "tissue_region")
    img = img.replace("csv", "tif")
    bw = img.replace('tissue_region','filtered')

    fn, ext = os.path.splitext(img)

    img = Image.open(os.path.join(src,img))
    bw = Image.open(os.path.join(src,bw))

    dst = os.path.join(src,fn+'_tile')
    dst_img = os.path.join(dst,'image')
    dst_bw = os.path.join(dst,'mask')
    if not os.path.exists(dst): os.mkdir(dst)
    if not os.path.exists(dst_img): os.mkdir(dst_img)
    if not os.path.exists(dst_bw): os.mkdir(dst_bw)

    for idx,(x,y) in enumerate(zip(x_cent,y_cent)):
        (left, upper, right, lower) = (x-50, y-50, x+50, y+50)
        im_crop = img.crop((left, upper, right, lower))
        bw_crop = bw.crop((left,upper,right,lower))
        im_crop.save(os.path.join(dst_img,'tile_{:d}.png'.format(idx)))
        bw_crop.save(os.path.join(dst_bw,'tile_{:d}.png'.format(idx)))
