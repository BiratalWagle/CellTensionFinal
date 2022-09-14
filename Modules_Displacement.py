import numpy as np
import cv2
from sklearn.metrics import mean_squared_error

def lefCol(img): 
    img = img//np.amax(img)
    sum = -1 
    count = -1 
    while sum <=0: 
        count +=1 
        col = img[:, count] # find the first column 
        sum = np.sum(col)   
    vertex1 = sum//2
    pint = 0  
    for i in range(len(col)): 
        if col[i]!=0: 
            break 
        else: 
            pint += 1 
    vertex1 = pint + sum//2 
    return (count, vertex1)

def rightCol(img):   
    img = img//np.amax(img)
    sum = -1
    width, height = img.shape 
    count =  height
    while sum <=0: 
        count -= 1
        col = img[:, count] # find the first column 
        sum = np.sum(col)   
    vertex2 = sum//2
    pint = 0  
    for i in range(len(col)): 
        if col[i]!=0: 
            break 
        else: 
            pint += 1 
    vertex2 = pint + sum//2 
    return (count, vertex2)

def toprow(img):
    img = img//np.amax(img)
    sum = -1
    count = -1
    while sum <=0: 
        count += 1
        row = img[count] # find the first row 
        sum = np.sum(row)   
    pint = 0  
    for i in range(len(row)): 
        if row[i]>0: 
            break 
        else: 
            pint += 1 
    vertex3 = pint + sum//2 
    return (count, vertex3)


def botrow(img):
    img = img//np.amax(img)
    sum = -1
    width, height = img.shape 
    count =  width
    while sum <=0: 
        count -= 1
        row = img[count] # find the last row 
        sum = np.sum(row)   

    pint = 0  
    for i in range(len(row)): 
        if row[i]!=0: 
            break 
        else: 
            pint += 1 
    vertex4 = pint + sum//2 
    return (vertex4, count)

def getEdges(test): 
    if len(test.shape) > 2: 
        img_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    else: 
        img_gray = test
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = img_gray.copy()
    img = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image=img, contours=contours, contourIdx=-1, color= (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    c = max(contours, key=cv2.contourArea)

    c = max(contours, key=cv2.contourArea)

    # Obtain outer coordinates
    l1, l2 = tuple(c[c[:, :, 0].argmin()][0])
    r1, r2 = tuple(c[c[:, :, 0].argmax()][0])
    t1, t2 = tuple(c[c[:, :, 1].argmin()][0])
    b1, b2 = tuple(c[c[:, :, 1].argmax()][0])
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.scatter(l1, l2)
    # plt.scatter(r1, r2)
    # plt.scatter(t1, t2)
    # plt.scatter(b1, b2)
    return l1, l2, r1, r2, t1, t2, b1, b2

def GetDisplacement(e1, e2): 
    dis = []
    for i in range(len(e1) - 1): 
        dis.append(np.sqrt((e1[i] - e1[i + 1])**2 + (e2[i] - e2[i+1])**2))
        
    return dis

def GetMSE(e1): 
    dis = []
    for i in range(len(e1) - 1): 
        dis.append(mean_squared_error(e1[i], e1[i + 1]))
    return dis

def selfGetEdges(img):
    if len(img.shape) > 2: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: 
        gray = img
    x1, y1 = lefCol(gray)
    x2, y2 = rightCol(gray)
    x3, y3 = toprow(gray)
    x4, y4 = botrow(gray)
    return x1, y1, x2, y2, x3, y3, x4, y4

def listGetEdges(imgs):
    a = []
    for i in range(len(imgs)):
        a.append(getEdges(imgs[i]))
    return a