#%%

import os
from xml2mask_v1 import xml2mask_v1
from matplotlib import pyplot as plt
import cv2
from time import time
import numpy as np
from PIL import Image
import pandas as pd

DL_src = r'\\kukissd\Kyu_Sync\deeplab with ashley\training_set\5x\classification_for_DL_skin_v3'
DL_results = [os.path.join(DL_src,_) for _ in os.listdir(DL_src) if _.endswith('tif')]
DL_results = [_ for _ in DL_results if not _.endswith(('024.tif','002.tif'))]

svs_src = r'\\kukissd\Kyu_Sync\deeplab with ashley\training_set'
svss = [os.path.join(svs_src,_) for _ in os.listdir(svs_src) if _.endswith('svs')]
svss = [_ for _ in svss if not _.endswith(('024.svs','002.svs'))]
xmls = [_.replace('.svs','.xml') for _ in svss]
masks=[]
DL_result_imgs=[]


for svs,xml,DL in zip(svss,xmls,DL_results):
    mask = xml2mask_v1(svs,xml,mag=5)
    masks.append(mask)
    DL_result_img = np.array(Image.open(DL))
    DL_result_img[(DL_result_img!=10)&(DL_result_img!=12)]=0
    DL_bw = (DL_result_img>0)*255
    DL_result_imgs.append(DL_bw)

accuracies = []

for idx,(DL_result_img,mask) in enumerate(zip(DL_result_imgs,masks)):
    # fig,ax=plt.subplots(1,2)
    # fig.set_figheight(12)
    # fig.set_figwidth(24)
    # ax[0].imshow(mask,cmap='gray')
    # ax[1].imshow(DL_result_img,cmap='gray')
    # fig.savefig(os.path.join(DL_src,'svg/manual_vs_deeplab_collagen_{:d}.svg'.format(idx)),format='svg')
    # plt.close(fig)
    mask = cv2.resize(mask, dsize=(DL_result_img.shape[::-1]), interpolation=cv2.INTER_CUBIC)


    DL_result_img[mask!=255]=0
    a=(np.sum(np.sum(DL_result_img,axis=1)/1)*1)
    b=(np.sum(np.sum(((mask>0)*255),axis=1)/1)*1)
    accuracies.append(a/b)

    # fig2,ax2=plt.subplots(1,2)
    # fig2.set_figheight(12)
    # fig2.set_figwidth(24)
    # ax2[0].imshow((mask>0)*255,cmap='gray')
    # ax2[1].imshow(DL_result_img,cmap='gray')
    # ax2[1].set_title('accuracy = '+str(a/b))
    # fig2.savefig(os.path.join(DL_src,'svg/manual_vs_deeplab_collagen_within_manual_{:d}.svg'.format(idx)),format='svg')
    # plt.close(fig2)

svsfn = [_ for _ in os.listdir(svs_src) if _.endswith('svs')]
svsfn = [_ for _ in svss if not _.endswith(('024.svs', '002.svs'))]

dict = {'Acc':accuracies,'Filename':svsfn}
df = pd.DataFrame(dict)
df.to_csv(os.path.join(DL_src,'accuracies.csv'))