import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from pathlib import Path
import os
import numpy 

def getFileDir(seg = False): # Returns a list with the png file names, or the seg.npy file names
    basedir = os.path.join(Path.cwd(), "Images")
    files = io.get_image_files(basedir, "")
    if seg == False: 
        return files
    else:
        dir = []
        for f in files: 
            path = os.path.join(basedir, f) + "_seg.npy"
            dir.append(path.replace(".png", ""))
        return dir

def getOutlines(fileOne, fileTwo, Mask = False): # takes in the directory to the first and second files
    # Import the first oultine

    datOne = np.load(fileOne, allow_pickle=True).item()
    masksOne = datOne['masks']
    outlinesOne = datOne['outlines']

    datTwo = np.load(fileTwo, allow_pickle=True).item()
    masksTwo = datTwo['masks']
    outlinesTwo = datTwo['outlines']
    if Mask == False: 
        return  (outlinesOne, outlinesTwo)
    elif Mask == True: 
        return (masksOne, masksTwo)


def evalOne(filename, newmask, defaultmodel = True): 
    img = imread(filename)
    # if defaultmodel == True: 
    model = models.CellposeModel(gpu = True, model_type= "cellpose_other") # We initilize the model and give the directory to our model
    # elif defaultmodel == False: 
    #     modeldir = os.path.join(Path.cwd(),"Images","models","Rebirth_2") 
    #     model = models.CellposeModel(gpu = True, pretrained_model = modeldir)
    masks, flows, styles = model.eval(img, diameter=None, channels=[0,0] )
    # save results so you can load in gui
    io.masks_flows_to_seg(img, newmask, flows, styles, filename, [0,0])


def getROI(Segdir, all = False): # Requires directory to the segmentation.npy files 
    from statistics import mode
    max = []
    for d in Segdir: 
        dat = np.load(d, allow_pickle=True).item()
        max.append(np.amax(dat['masks']))
    if all == False: 
        return mode(max)
    elif all == True: 
        return max

# Returns a specific outline given an roi 
def specificOutline(imgarr, roi): # takes in an imagearr and the roi
    empty = np.zeros_like(imgarr) 
    specific = np.where(imgarr == roi, imgarr, empty)
    return specific

def plotSide(a, b): # plots two outlines side by side. Made for convenience
    full, fullarr = plt.subplots(1,2)
    fullarr[0].imshow(a)
    fullarr[1].imshow(b)



def FixROI(A, B, roi): # Takes in an Array and the outline u want
    # initializing
    sigma = 0 
    count = 0 
    loopNum = 0 

    a = specificOutline(A, roi)
    b = specificOutline(B, roi)

    while (checkROI(a, b) == False) and  (0 < roi + sigma < np.amax(B)):
        sigma = count * (-1)**loopNum  # Lets us check the ROI above and below 
        b = specificOutline(B, roi + sigma)
        if loopNum % 2 != 0: # number is odd
            countFinder = loopNum - 1 
        else: 
            countFinder = loopNum
        count = 0.5 * countFinder + 1
        loopNum += 1
        
    return (roi + sigma)

def fileNameFromPath(dre): 
    a = (os.path.basename(os.path.normpath(dre)))
    count = 0 
    char = a[0]
    num = ""

    while char != "_": 
        count += 1
        num += char
        char = a[count]
    
    return int(num)



def findNonMatching(directory, lower, upper):
    
    weird = [] # Stores the weird/not working files. Namely the ones that dont' have matching ROIs
    BadOnes = [] # stores how many weird ones there are per frame 

    # Following code gives back a tuple with (index number, number of incorrectly labeled ROIs, an array with the outlines that aren't working)
    for i in range(lower, upper): # start from 1 as 0 doesn't matter
        outlinesFirst, outlinesLast = getOutlines(directory[0],  directory[i])
        weird = []
        
        for out in range(1, np.amax(outlinesLast)): 
            outA = specificOutline(outlinesFirst, out)
            outB = specificOutline(outlinesLast, out)
            check = checkROI(outA, outB)
            if check == False:
                weird.append(out)
            # full, fullarr = plt.subplots(1,2)
            # fullarr[0].imshow(outA)
            # fullarr[1].imshow(outB)
          
        if len(weird) > 0: 
            BadOnes.append((fileNameFromPath(directory[i]), weird))
    return BadOnes

# Bad = findNonMatching(dir, len(dir))
# Function to see if the ROI Matches
def checkROI(outA, outB): # input two outlines as matrixes
    row, col = outA.shape
    count = 0
    outCount = 0 
    # We iterate through the entire matrix which represents the two outlines. 
    for i in range(row): 
        for j in range(col):
            if (outA[i][j] != 0):  # Find how many pixels are used to make the outlines
                outCount += 1      # outCount gives the number of pixels to create the outlines
                if outB[i][j] != 0:
                    count +=1      # The number of overlapping pixels in the first and second outlines
    if outCount == 0:
        # print("no overlap:") 
        return False
    elif count/outCount * 100 >= 10: 
        # print("Overlap:", count/outCount * 100 ) # If more that 10% of it overlaps, we say it's the same cell
        return True
    elif count > 0: 
        print("Some Overlap:", count/outCount * 100) 
        return False 
    else: 
        return False


def smoother(x,window_len=11,window='hanning'):

    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def GetPositions(r): 
    top = bot = left = right = 0
    a, b = r.shape
    for i in range(1, b): 
        if sum(r[i]) != 0 and sum(r[i-1]) == 0: 
            top = i 
        if sum(r[i]) != 0 and sum(r[i + 1] == 0):
            bot = i 
        if sum(r[:, i]) != 0 and sum(r[:, i-1]) == 0: 
            left = i
        if sum(r[:, i]) != 0 and sum(r[:, i + 1]) == 0:
            right = i
    
    return top, bot, left, right 

def splitImg(r, vert = False): 
    m = r
    h = len(m[0])//2
    w = len(m)//2
    if vert == False: 
        return m[0:w], m[w::]
    else: 
        return m[0:h], m[h::]
  
def Shrink(r, w, top = 0, bot = 0, left = 0, right = 0, manual = False): 
    if manual == False: 
        top, bot, left, right = GetPositions(r)
        
    top = top - w
    bot = bot + w
    left = left - w
    right = right + w
        
    matrix = np.array(r)
    return matrix[top:bot, left:right]