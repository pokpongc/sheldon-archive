import cv2
import numpy as np
import imutils
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 2500
SC_CARD_MAX_AREA = 500000
SC_CARD_MIN_AREA = 100000
BKG_THRESH = 80
SC_BKG_THRESH = 30
CARD_THRESH = 30
class Query_card:
    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.pts = [0,0,0,0]
        self.coner4pts=[0,0,0,0]
        self.mask=[]
def preprocess_image(image,state=0):
    a = BKG_THRESH
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv,(5,5),0.33)
    text = open("./imagesnap/background.txt", "r")
    background_range = text.readline()
    bg = list(map(int,background_range.split(',')))
    text.close()
    
    text_U = open("./imagesnap/background_U.txt", "r")
    background_range_U = text_U.readline()
    bg_U = list(map(int,background_range_U.split(',')))
    # print(bg,bg_U)
    text_U.close()
    if(state==1):
        lower = np.array([[bg[0],bg[2],bg[4]]])#default value = 155
        upper = np.array([[bg[1],bg[3],bg[5]]])
        thresh_L = cv2.inRange(blur,lower,upper)
        cv2.imshow('a',thresh_L)
        lower_U = np.array([[bg_U[0],bg_U[2],bg_U[4]]])#default value = 155
        upper_U = np.array([[bg_U[1],bg_U[3],bg_U[5]]])
        thresh_U = cv2.inRange(blur,lower_U,upper_U)
        cv2.imshow('aaab',thresh_U)
        thresh = cv2.bitwise_or(thresh_L,thresh_U)
        # cv2.imshow('aa',thresh)
        # thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,np.ones((1,1),np.uint8), iterations = 1)
        # thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((8,8),np.uint8), iterations = 5)
        # thresh = cv2.subtract(thresh,cv2.morphologyEx(thresh,cv2.MORPH_HITMISS,np.ones((2,2),dtype='uint8')))
        cv2.imshow('adca',thresh)
        # cv2.waitKey()
        return thresh
    if(state==0):
        # lower = np.array([[20,2,100]])#default 13,10,155
        # upper = np.array([[70,13,255]])
        lower = np.array([[bg[0],bg[2],bg[4]]])#default value = 155
        upper = np.array([[bg[1],bg[3],bg[5]]])
        thresh_L = cv2.inRange(blur,lower,upper)
        cv2.imshow('ab',thresh_L)
        lower_U = np.array([[bg_U[0],bg_U[2],bg_U[4]]])#default value = 155
        upper_U = np.array([[bg_U[1],bg_U[3],bg_U[5]]])
        thresh_U = cv2.inRange(blur,lower_U,upper_U)
        thresh = cv2.bitwise_or(thresh_L,thresh_U)
        cv2.imshow('aba',thresh_U)
        
        edges = cv2.Canny(thresh,0,30)
        ed=edges
        kernel = np.ones((5,5),np.uint8)
        edges =cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        _,cnts,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i in range(len(cnts)):
                if(cv2.contourArea(cnts[i])<10000000 and cv2.contourArea(cnts[i])>1000):
                        
                        x,y,w,h = cv2.boundingRect(cnts[i])
                        cv2.rectangle(blur,(x,y),(x+w,y+h),(255,0,0))
                        cv2.drawContours(edges,cnts,i,(255,255,255),-1)    
        # thresh = cv2.bitwise_or(thresh,edges)
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,np.ones((3,3),np.uint8), iterations = 2)
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_ERODE,np.ones((1,1),np.uint8), iterations = 4)
    # thresh = cv2.subtract(thresh,cv2.morphologyEx(thresh,cv2.MORPH_HITMISS,np.ones((2,2),dtype='uint8')))
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations = 1)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8), iterations = 3)
    # thresh = cv2.subtract(thresh,cv2.morphologyEx(thresh,cv2.MORPH_HITMISS,np.ones((2,2),dtype='uint8')))
    cv2.imshow('acca',thresh)
    
    return thresh

def find_cards(thresh_image):
    
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    if len(cnts) == 0:
        return [], []
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])


    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
           
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card
def findscale_cards(thresh_image):
    
    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)
    if len(cnts) == 0:
        return [], []
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < SC_CARD_MAX_AREA) and (size > SC_CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
           
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card
def flattener(image, pts, w, h,cards):
    temp_rect = np.zeros((4,2), dtype = "float32")
    s = np.sum(pts, axis = 2)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left
            
            #direction=0
            
        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
    # cards.coner4pts=(tl,bl,tr,br)
    # print(tl,bl,tr,br,"g",temp_rect[0], temp_rect[3], temp_rect[1], temp_rect[2])
    cards.coner4pts=( [temp_rect[0]], [temp_rect[3]], [temp_rect[1]], [temp_rect[2]])
    maxWidth = 601
    maxHeight = 601
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)   
    return warp
def calXYZ(card,cscale=1):
    w_scale = 250/601
    tl,bl,tr,br=card.coner4pts
    return (int(tl[0][0]*w_scale*cscale),int(tl[0][1]*w_scale*cscale)),(int(bl[0][0]*w_scale*cscale),int(bl[0][1]*w_scale*cscale)),(int(tr[0][0]*w_scale*cscale),int(tr[0][1]*w_scale*cscale)),(int(br[0][0]*w_scale*cscale),int(br[0][1]*w_scale*cscale))
def squareframe(x,y,w,h,card,crop_image):
    blank_image = np.zeros(shape=[601,601, 3], dtype=np.uint8)
    scale = min(601/w,601/h)
    w=int(w*scale)
    h=int(h*scale)
    tl,bl,tr,br=card.coner4pts
    cv2.rectangle(blank_image,(300-int(w/2),300-int(h/2)),(300+int(w/2),300+int(h/2)),(0,255,255))

    resized = cv2.resize(crop_image, (w,h), interpolation = cv2.INTER_AREA)
    p1,p2,p3,p4=calXYZ(card,scale)
    blank_image2 = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)

    cv2.imshow('s',resized)
    return resized
def preprocess_card(contour, image,state=0):
    qCard = Query_card()

    qCard.contour = contour
    
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    x,y,w,h = cv2.boundingRect(contour)
    x-=1
    y-=1
    w+=0
    h+=4
    qCard.width, qCard.height = w, h
    scale = min(601/w,601/h)
    qCard.pts[0]=x
    qCard.pts[1]=y
    qCard.pts[2]=w
    qCard.pts[3]=h
    if(state==0):
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0))
        qCard.pts[0]=x
        qCard.pts[1]=y
        qCard.pts[2]=w
        qCard.pts[3]=h
        w=w*scale
        h=h*scale
    else:
        pass

    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    qCard.warp = flattener(image, pts, w, h,qCard)
    
    cv2.imshow('warp',qCard.warp)
    return qCard

def draw_results(image, qCard,state=0):

    if(state==2):
        return image
    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)
    if(state==1):
        x = qCard.center[0]
        y = qCard.center[1]
        cv2.circle(image,(x,y),5,(255,0,255),-1)
        p1,p2,p3,p4=calXYZ(qCard)
        tl,bl,tr,br=qCard.coner4pts
        cv2.putText(image,str(p1),(int(tl[0][0]+20),int(tl[0][1]+20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,100))
        cv2.putText(image,str(p2),(int(bl[0][0]+20),int(bl[0][1]-20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,100))
        cv2.putText(image,str(p3),(int(tr[0][0]-200),int(tr[0][1]+20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,100))
        cv2.putText(image,str(p4),(int(br[0][0]-200),int(br[0][1]-20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,100))

    return image