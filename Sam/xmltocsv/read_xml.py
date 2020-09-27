import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.io import savemat
import os
from time import sleep

def xml2mat(xmlfile):
   tree = ET.parse(xmlfile)
   root = tree.getroot()
   x = np.array([])
   y = np.array([])
   obj = np.array([])
   label = np.array([])
   for Annotation in root.iter('Annotation'):
      for Region in Annotation.iter('Region'):
         xx = np.array([int(Vertex.get('X')) for Vertex in Region.iter('Vertex')])
         yy = np.array([int(Vertex.get('Y')) for Vertex in Region.iter('Vertex')])
         objj = np.array([int(Region.get('Id'))]*len(xx))
         labell = np.array([int(Annotation.get('Id'))]*len(xx))
         x = np.concatenate((x,xx),axis=None)
         y = np.concatenate((y,yy),axis=None)
         obj = np.concatenate((obj,objj),axis=None)
         label = np.concatenate((label,labell),axis=None)
   print('number of coordinates in annotation : ',len(x))
   mdict = {'x':x,'y':y,'objID':obj,'label':label}
   return mdict

dir = r'C:\Users\kyuha\Desktop\Skin\SVS\scan 3'
xmls = [os.path.join(dir,_) for _ in os.listdir(dir) if _.endswith('.xml')]
for xml in xmls:
   mdict = xml2mat(xml)
   fname = os.path.basename(xml)
   (file, ext) = os.path.splitext(fname)
   savemat(os.path.join(dir,file+'.mat'),mdict=mdict,do_compression=True)