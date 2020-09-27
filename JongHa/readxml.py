import xml.etree.ElementTree as ET
from time import time
import numpy as np
import pandas as pd

# xml_path = absolute filepath of xml
# mdict = 2D coordinates of annotation by class and object
def readxml(xml_path):
    start = time()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    x = np.array([])
    y = np.array([])
    obj = np.array([])
    classID = np.array([])
    for Annotation in root.iter('Annotation'):
      for Region in Annotation.iter('Region'):
         xx = np.array([float(Vertex.get('X')) for Vertex in Region.iter('Vertex')])
         yy = np.array([float(Vertex.get('Y')) for Vertex in Region.iter('Vertex')])
         objj = np.array([float(Region.get('Id'))]*len(xx))
         classIDD = np.array([float(Annotation.get('Id'))]*len(xx))
         x = np.concatenate((x,xx),axis=None)
         y = np.concatenate((y,yy),axis=None)
         obj = np.concatenate((obj,objj),axis=None)
         classID = np.concatenate((classID,classIDD),axis=None)
    # print('number of coordinates in annotation : ',len(x))
    x = x.astype(int)
    y = y.astype(int)
    obj = obj.astype(int)
    classID = classID.astype(int)
    mdict = {'x':x,'y':y,'objID':obj,'classID':classID}
    print('readxml took {:.2f} sec'.format(time()-start))
    return pd.DataFrame(mdict)