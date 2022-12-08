# A file with all the required functions

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.cluster import KMeans
from PIL import Image
import os
from cellpose import io
from pathlib import Path

def specificOutline(imgarr, roi): # takes in an imagearr and the roi
    empty = np.zeros_like(imgarr) 
    specific = np.where(imgarr == roi, imgarr, empty)
    return specific

def getROIName(npyDir, basedir = 1, widthdiv = 2, heightdiv = 2): 
    if basedir == 1: 
        basedir = os.path.join(Path.cwd(), npyDir)
    files = io.get_image_files(basedir, "")
    dir = []
    maskList = []
    for f in files: 
        dir.append((os.path.join(basedir, f) + "_seg.npy").replace(".tif", ""))
    roi_name = [] # a list containing the name of the roi we need
    for i in range(len(dir)):  
        dat = np.load(dir[i], allow_pickle=True).item()
        masks = dat['masks'] # Using the masks gives us a larger target to select from than outlines
        filenam = dat['filename']
        width, height = masks.shape
        neededROI = masks[width//widthdiv][height//heightdiv] # the general area of where our ROI will always be
        # the below code saves the outline in an image file
        maskList.append(masks)
        if neededROI != 0: 
            roi_name.append([i, neededROI])
    return roi_name, files, dir, maskList

def findVertex(masks, j, i, k = 3):  
    surr =[masks.item((j+k, i-k)),
    masks.item((j+k, i)),
    masks.item((j+k, i+k)),
    masks.item((j, i-k)),
    masks.item((j, i)),
    masks.item((j, i+k)),
    masks.item((j-k, i-k)),
    masks.item((j-k, i)),
    masks.item((j-k, i+k))]
    if len(list(set(surr)))>2: 
        return [j, i]

def dist(f, g): # displacement 
    return np.sqrt((f[0] - g[0])**2 + (f[1] - g[1])**2)

def GetPointsList(roi_name, dir): 
    M = [] # list containing all the masks
    O = [] # list containing all the outlines
    for k in range(len(roi_name)): 
        num, roi = roi_name[k]
        dat = np.load(dir[num], allow_pickle=True).item()
        masks = dat['masks']
        outlines = dat['outlines']
        M.append(masks)
        O.append(specificOutline(outlines, roi)) 
    pointsList = []

    for k in range(len(M)): 
        points = []
        o = O[k]
        height, width = o.shape
        for i in range(height): 
            for j in range(width): 
                if o[i][j] != 0: 
                    points.append([j, i])
        pointsList.append(points)
    return pointsList, M, O

def GetVertexList(roi_name, pointsList, masks): 
    mainVertexList = []
    for q in range(len(roi_name)): 
        points = pointsList[q] # specific set of points
        vertexes = []
        for k in range(len(points)): 
            i, j = points[k]
            vert = findVertex(masks[0], j, i) # This finds if a particular point is a vertex or not
            if vert: 
                vertexes.append(vert)
        mainVertexList.append(vertexes) # a list of the vertexes
    return mainVertexList
    
def GetClusters(numOfClusters, mainVertexList): 
    kmeans = KMeans(n_clusters= numOfClusters, random_state=42) 
    a = []
    x = []
    for i in range(len(mainVertexList)): 
        X = mainVertexList[i]
        if len(X) > 1: 
            clusts = kmeans.fit_predict(X)
            df = pd.DataFrame(X)
            df["labels"] = clusts
            df["frame"] = i 
            x.append(df)
    allx = pd.concat(x)
    C = list(zip(allx[1], allx[0]))
    clusts = kmeans.fit_predict(C)
    allx["clusterNum"] = clusts
    return  allx

def ShowClusterPlot(allx, files, clr = "white"): 
    for k in range(max(allx["clusterNum"])): 
        clus1 = allx.loc[allx["clusterNum"] == k] # select the cluster you want to look at
        plt.scatter(clus1[1], clus1[0])
        plt.text(np.array(clus1[1])[0], np.array(clus1[0])[0], k, c = clr)
        plt.imshow(Image.open(files[0]))

