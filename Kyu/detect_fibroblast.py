import os
from time import time
import numpy as np
import scipy
# image processing
import cv2
from skimage.measure import label,find_contours,regionprops
from skimage.morphology import remove_small_objects
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
# custom function
from xml2mask import xml2mask
from rgb2hed_v1 import separate_stains
from PIL import Image
import pandas as pd
from copy import deepcopy

svs_src = r'\\kukissd\research\aging\undergrad_replicate_data\svs\heejae\age_group2'
fns = [os.path.splitext(_)[0] for _ in os.listdir(svs_src) if _.lower().endswith('svs')]
dst = os.path.join(svs_src,'roi')

fns = ['5619_Wirtz.Denis_OTS-19_5619-034']
for fn in fns:
    rois = []
    rois = xml2mask(svs_src,fn,dst)
    for regionidx,region in enumerate(rois[2:3]):
        regionidx=2
        # RGB to Haematoxylin-Eosin-DAB (HED) color space conversion.
        # Hematoxylin + Eosin + DAB
        start=time()
        # rgb_from_hed = np.array([[0.650, 0.704, 0.286],
        #                          [0.072, 0.990, 0.105],
        #                          [0.268, 0.570, 0.776]])
        rgb_from_hed = np.array([[0.650, 0.704, 0.286],
                                 [0.072, 0.990, 0.105],
                                 [0.0, 0.0, 0.0]])
        rgb_from_hed[2, :] = np.cross(rgb_from_hed[0, :], rgb_from_hed[1, :])
        hed_from_rgb = scipy.linalg.inv(rgb_from_hed)
        Hema = separate_stains(region, hed_from_rgb)[:, :, 0]
        print("color deconvolution: {:.2f} sec elapsed".format(time()-start))
        # normalize image between 0 to 255
        Hematoxylin = cv2.normalize(Hema, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Kmean_pixels = [0]
        # while np.max(Kmean_pixels)==0:
        #     start = time()
        #     # 1 K-means intensity thresholding
        #     vectorized = Hematoxylin.reshape((-1, 1))
        #     vectorized = np.float32(vectorized)
        #     print(np.min(vectorized), np.mean(vectorized), np.max(vectorized))
        #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #     K = 6
        #     attempts = 10
        #     ret, Klabel, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        #     center = np.uint8(center)
        #     Kmean_pixels = []
        #     fig, axes = plt.subplots(2, 3, figsize=(30, 20), sharex=True, sharey=True)
        #     ax = axes.ravel()
        #     ax[0].imshow(Hematoxylin[2000:3000, 2000:3000], cmap='gray')
        #     ax[0].set_title("Original image")
        #     for idx, aa in enumerate(ax[1:]):
        #         Klabel2 = deepcopy(Klabel)
        #         Klabel2[Klabel2 != idx] = 0
        #         res = center[Klabel2.flatten()]
        #         result_image = res.reshape((Hematoxylin.shape))
        #         print(np.unique(result_image)[-1])
        #         Kmean_pixels.append(np.unique(result_image)[-1])
        #         aa.imshow(result_image[2000:3000, 2000:3000], cmap='gray')
        #         aa.set_title("K={:d}".format(idx + 1))
        #     for aa in ax.ravel():
        #         aa.axis('off')
        #     fig.savefig(os.path.join(dst, fn + '_K-mean_{:d}.svg'.format(regionidx)), format='svg')
        #     plt.close()
        #     print('Kmean_pixels max:',np.max(Kmean_pixels))
        #     print("K-means segmentation: {:.2f} sec elapsed".format(time() - start))
        # Klabel[Klabel != np.argwhere(Kmean_pixels == np.max(Kmean_pixels))[0][0]] = 0
        # res = center[Klabel.flatten()]
        # picked_image = res.reshape(Hematoxylin.shape)
        plt.figure(figsize=(12, 12))
        plt.hist(Hematoxylin.ravel(), 255, [0, 255])
        plt.xticks(range(0,256,5))
        plt.show()
        threshold = input("Enter threshold (0-255):")
        threshold = int(threshold)
        Hematoxylin_temp = deepcopy(Hematoxylin)
        Hematoxylin_temp[Hematoxylin < threshold] = 0
        Hematoxylin_temp[Hematoxylin > threshold] = 1
        # label the objects
        labeled_bw = label(Hematoxylin_temp)
        #2 size filter
        start=time()
        remove_small_objects(labeled_bw,min_size=30,connectivity=1,in_place=True)
        print("number of nucleus as of now:",len(np.unique(labeled_bw)))
        print("size filter {:.2f} sec elapsed".format(time()-start))
        #2 save bw before distance filter
        bw = (labeled_bw > 1) * 255
        plt.figure(figsize=(12,12))
        plt.imshow(bw[0:6000,0:6000],cmap='gray')
        plt.title(fn+str(regionidx))
        plt.figure(figsize=(12, 12))
        plt.imshow(region[0:6000, 0:6000])
        plt.title(fn+str(regionidx))
        plt.show()
        ########
        threshold = input("Enter threshold (0-255):")
        threshold = int(threshold)
        Hematoxylin[Hematoxylin < threshold] = 0
        Hematoxylin[Hematoxylin > threshold] = 1
        # label the objects
        labeled_bw = label(Hematoxylin)
        # 2 size filter
        start = time()
        remove_small_objects(labeled_bw, min_size=30, connectivity=1, in_place=True)
        print("number of nucleus as of now:", len(np.unique(labeled_bw)))
        print("size filter {:.2f} sec elapsed".format(time() - start))
        # 2 save bw before distance filter
        bw = (labeled_bw > 1) * 255
        plt.figure(figsize=(12, 12))
        plt.imshow(bw[0:6000, 0:6000], cmap='gray')
        plt.title(fn + str(regionidx))
        plt.figure(figsize=(12, 12))
        plt.imshow(region[0:6000, 0:6000])
        plt.title(fn + str(regionidx))
        plt.show()
        ########
        bw_img = Image.fromarray(bw).convert('1')
        bw_img.save(os.path.join(dst, fn + '_SZ_filtered_{:d}.tif'.format(regionidx)))
        #3 AR filter
        minAR = 2.5
        maxAR = 15
        start=time()
        props = regionprops(labeled_bw)

        def ARfilter(x):
            if (x['minor_axis_length']!=0): AR = x['major_axis_length']/x['minor_axis_length']
            else: AR = 0
            if AR<minAR: labeled_bw[labeled_bw==x.label]=0
            if AR>maxAR: labeled_bw[labeled_bw==x.label]=0
        Parallel(n_jobs=-4, prefer="threads")(delayed(ARfilter)(x) for x in props)
        # for idx,prop in enumerate(props):
        #     if idx%300==0: print('AR',idx)
        #     if (prop['minor_axis_length']!=0): AR = prop['major_axis_length']/prop['minor_axis_length']
        #     else: AR=0
        #     if AR<minAR: labeled_bw[labeled_bw==prop.label]=0
        #     if AR>maxAR: labeled_bw[labeled_bw==prop.label]=0
        print("number of nucleus as of now:", len(np.unique(labeled_bw)))
        print("AR filter {:.2f} sec elapsed".format(time()-start))
        #3 save bw before distance filter
        bw = (labeled_bw > 0) * 255
        bw_img = Image.fromarray(bw).convert('1')
        bw_img.save(os.path.join(dst, fn + '_AR_filtered_{:d}.tif'.format(regionidx)))
        #4 SF filter (keep cells in the range)
        start = time()
        minSF = 0.1
        maxSF = 0.7
        for prop in regionprops(labeled_bw):
            SF = 4*np.pi*prop['area']/prop['perimeter']**2
            if SF<minSF: labeled_bw[labeled_bw==prop.label]=0;
            if SF>maxSF: labeled_bw[labeled_bw==prop.label]=0;
        print("number of nucleus as of now:", len(np.unique(labeled_bw)))
        print("SF filter {:.2f} sec elapsed".format(time() - start))
        #4 save bw before distance filter
        bw = (labeled_bw>0)*255
        bw_img = Image.fromarray(bw).convert('1')
        bw_img.save(os.path.join(dst,fn+'_SF_filtered_{:d}.tif'.format(regionidx)))
        #5 distance filter
        start = time()
        min_dist_to_neighbor = 100
        prop = regionprops(labeled_bw)
        Y = [_.centroid for _ in prop]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Y)
        distances, indices = nbrs.kneighbors(Y)
        close_objects = np.array(prop)[distances[:,1]<min_dist_to_neighbor]
        for close_object in close_objects:
            labeled_bw[labeled_bw==close_object.label]=0
        print("number of nucleus as of now:", len(np.unique(labeled_bw)))
        print("Distance filter {:.2f} sec elapsed".format(time() - start))
        #5 save bw before distance filter
        bw = (labeled_bw > 0) * 255
        bw_img = Image.fromarray(bw).convert('1')
        bw_img.save(os.path.join(dst,fn+'_Dist_filtered_{:d}.tif'.format(regionidx)))
        #6 export datasheet
        prop = regionprops(labeled_bw)
        xs = [np.around(_['centroid'][0]) for _ in prop]
        ys = [np.around(_['centroid'][1]) for _ in prop]
        area = [np.sum(x._label_image[x._slice] == x.label) for x in prop]
        ARs = [np.around(_['major_axis_length']/_['minor_axis_length'],decimals=3) for _ in prop]
        SFs = [np.around(4*np.pi*_['area']/_['perimeter']**2,decimals=3) for _ in prop]
        dict = {'x':xs,'y':ys,'area':area,'aspect_ratio':ARs,'circularity':SFs}
        df = pd.DataFrame(dict)
        df.to_csv(os.path.join(dst,fn+'_parameters_{:d}.csv'.format(regionidx)))