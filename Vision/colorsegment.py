import cv2
from scipy import stats
import operator
import numpy as np
from math import sqrt
class Cards:
    def __init__(self, image, shape,countour,imagesnap,debug=False):
        self.img = image
        self.imgsnap = imagesnap
        self.shape = shape
        self.countour = countour
        self.preprocess = self.preprocess()
        self.incard_mask = []
        self.color_ctr=[0,0,0,0,0,0]
        self.color_area=[0,0,0,0,0,0]
        self.color_allarea=[0,0,0,0,0,0]
        self.mask=[0,0,0,0,0,0]
        self.color_pos=[0,0,0,0,0,0]
        self.edgemask=[0,0,0,0,0,0]
        self.finalmask=[0,0,0,0,0,0]
        self.debug=debug
        
    def _setwhite(self):
        lab = cv2.cvtColor(self.preprocess,cv2.COLOR_BGR2HSV)
        lower = np.array(self.color[5][0])
        upper = np.array(self.color[5][1])
        mask = cv2.inRange(lab,lower,upper) 
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8), iterations = 2)
        mask = cv2.bitwise_and(mask,mask,mask=self.incard_mask)
        self.mask[5] = mask    
        # cv2.imshow('d',mask)
        # cv2.waitKey()   
        # cv2.destroyAllWindows()
    def preprocess(self):
        result = cv2.fastNlMeansDenoisingColored(self.img,None,20,10,7,21)
        return result    
    def hsvsegment(self):
        for i in range(5):
            if(i==4):
                self.newsetwhite()
            
            im2, ctrs, hier = cv2.findContours(self._hsvsegment(i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_area=0
            dummy_mask=self._hsvsegment(i).copy()
            point=0
            for j, ctr in enumerate(ctrs):
                # Get bounding box
                if(cv2.contourArea(ctr)>=largest_area):
                    
                    self.color_ctr[i]=ctr
                    largest_area=cv2.contourArea(ctr)
                    
                    point = j
            cv2.drawContours(dummy_mask,ctrs,point,(0,0,0),-1)
            # cv2.imshow(str(i),dummy_mask)
            final_mask=cv2.bitwise_and(cv2.bitwise_not(dummy_mask),self._hsvsegment(i))
            _, w_ctrs, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            try:
                self.color_area[i]=cv2.contourArea(w_ctrs[0])
                x, y, w, h = cv2.boundingRect(self.color_ctr[i])
                self.color_pos[i]=[x+int(w/2),y+int(h/2)]
                self.mask[i]=final_mask
                print(final_mask.count())
            except:
                pass
    def newsetwhite(self):
        gray = cv2.cvtColor(self.preprocess,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
    def _hsvsegment(self,index_color):
        lab = cv2.cvtColor(self.preprocess,cv2.COLOR_BGR2HSV)
        if(index_color == 0):
            lower = np.array(self.color[index_color][0])
            upper = np.array(self.color[index_color][1])
            mask1 = cv2.inRange(lab,lower,upper) 
            lower = np.array(self.color[index_color][2])
            upper = np.array(self.color[index_color][3])
            mask2 = cv2.inRange(lab,lower,upper) 
            mask = cv2.bitwise_or(mask1,mask2)
        elif(index_color==4):   
            lower = np.array(self.color[index_color][0])
            upper = np.array(self.color[index_color][1])
            mask = cv2.inRange(lab,lower,upper) 
           
            mask = cv2.bitwise_and(mask,mask,mask=self.incard_mask)
            color = cv2.bitwise_or(self.mask[0],self.mask[1])
            color = cv2.bitwise_or(color,self.mask[2])
            color = cv2.bitwise_or(color,self.mask[3])
            self.notcolor = cv2.bitwise_not(color)
            mask = cv2.bitwise_and(mask,cv2.bitwise_not(color))
            mask = cv2.bitwise_and(mask,cv2.bitwise_not(self.mask[5]))
            mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE,np.ones((5,5),np.uint8), iterations = 2)
            mask = cv2.bitwise_and(mask,mask,mask=self.incard_mask)
            return mask   
        else:
            lower = np.array(self.color[index_color][0])
            upper = np.array(self.color[index_color][1])
            mask = cv2.inRange(lab,lower,upper)
            mask = cv2.bitwise_and(mask,mask,mask=self.incard_mask)
            mask = cv2.bitwise_and(mask,cv2.bitwise_not(self.mask[5]))    
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8), iterations = 2)
        mask = cv2.bitwise_and(mask,mask,mask=self.incard_mask)     
        self.color_allarea[index_color]=mask
        return mask
    def edgedetectranking(self,img):
        bg = open("./imagesnap/"+"con_background"+".txt","r")
        msg_bg = list(map(int,bg.readline().split(',')))
        imgcon = self.contrast(img,msg_bg[0], msg_bg[1])
        hsvcon = cv2.cvtColor(imgcon,cv2.COLOR_BGR2HSV)
        lower = np.array([[msg_bg[2],msg_bg[4],msg_bg[6]]])
        
        upper = np.array([[msg_bg[3],msg_bg[5],msg_bg[7]]])
        mask = cv2.inRange(hsvcon,lower,upper) 
        newimgcon = cv2.bitwise_and(imgcon,imgcon,mask=cv2.bitwise_not(mask))
        cv2.imshow('contrassst',imgcon)
        # cv2.waitKey()
        imgcon = newimgcon.copy()
        def auto_canny(image, sigma=0.33):
            v = np.median(image)
            lower = int(max(0, (1.0 - sigma) * v)/10)
            upper = int(min(255, (1.0 + sigma) * v)/10)
            edged = cv2.Canny(image, lower, upper)
            return edged
        imgc=imgcon.copy()
        imgd=imgc.copy()
        edge = auto_canny(imgc)

        edge = cv2.subtract(edge,cv2.morphologyEx(edge,cv2.MORPH_HITMISS,np.ones((19,19),dtype='uint8')))
        edge = cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8), iterations = 1)
        cv2.imshow('eddd',edge)
        im2, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imgd=imgc.copy()
        for i,cnt in enumerate(contours):
            cv2.drawContours(edge, contours, i, (255,255,255), thickness=-1)
        cv2.imshow('edd',edge)
        if(self.debug):
            cv2.waitKey()
        im2, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obj = []
        for i,cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 1200:
                obj.append(cnt) 
        c = 0
        listobj = []
        listcolor = []
        for i,ob in enumerate(obj):
            
            colormask = edge.copy()
            x,y,w,h = cv2.boundingRect(ob)

            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
            bgr = img.copy()
            c = c+1
            cv2.drawContours(colormask, obj,i,(0,0,0) , thickness=-1)
            
            ob_h,ob_s,ob_v = cv2.split(hsv[y:y+h,x:x+w])
            ob_l,ob_a,ob_b = cv2.split(lab[y:y+h,x:x+w])
            ob_blue,ob_green,ob_red = cv2.split(bgr[y:y+h,x:x+w])

            fmask = cv2.bitwise_xor(colormask,edge)
            listcolor.append(fmask)
            # print(listcolor)
            erode = cv2.morphologyEx(listcolor[i],cv2.MORPH_ERODE,np.ones((3,3),dtype='uint8'))
            thin = cv2.subtract(erode,cv2.morphologyEx(erode,cv2.MORPH_HITMISS,np.ones((9,9),dtype='uint8')))
            thin = cv2.subtract(thin,cv2.morphologyEx(thin,cv2.MORPH_HITMISS,np.ones((9,9),dtype='uint8')))
            
            cv2.imshow('liost',thin)
            
            # x,y,w,h = cv2.boundingRect(thin)
            TARGET = ((x+x+w)//2,(y+y+h)//2)
            def find_nearest_white(img, target):
                nonzero = cv2.findNonZero(img)
                distances = np.sqrt((nonzero[:,:,0] - TARGET[0]) ** 2 + (nonzero[:,:,1] - TARGET[1]) ** 2)
                nearest_index = np.argmin(distances)
                return nonzero[nearest_index]
            select_x = find_nearest_white(thin, TARGET)[0][0]
            select_y = find_nearest_white(thin, TARGET)[0][1]
            # print(select_x,select_y)
            M = cv2.moments(listcolor[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv2.circle(newimgcon,(select_x,select_y),2,(255,0,255))
            
            
            r,c = np.where(listcolor[i]>0)
            # print(len(r),len(c))
            hu=[]
            s=[]
            v=[]
            l=[]
            a=[]
            b=[]
            bl=[]
            gr=[]
            re=[]
            for le in range(len(r)//100,len(r),len(r)//100):
            #     for co in c:
            #         if(listcolor[i][ro][co]>0):
                hu.append(hsv[r[le],c[le]][0])
                s.append(hsv[r[le],c[le]][1])
                v.append(hsv[r[le],c[le]][2])
                l.append(lab[r[le],c[le]][0])
                a.append(lab[r[le],c[le]][1])
                b.append(lab[r[le],c[le]][2])
                bl.append(bgr[r[le],c[le]][0])
                gr.append(bgr[r[le],c[le]][1])
                re.append(bgr[r[le],c[le]][2])
                cv2.circle(newimgcon,(c[le],r[le]),2,(255,0,255))
            hu = np.array(hu)
            s = np.array(s)
            v = np.array(v)
            l = np.array(l)
            a = np.array(a)
            b = np.array(b)
            bl = np.array(bl)
            gr = np.array(gr)
            re = np.array(re)
            cv2.imshow("cir",newimgcon)
            if self.debug==1:
                cv2.waitKey()
            listobj.append([c,[[np.median(hu),np.median(s),np.median(v)],
                            [np.median(l),np.median(a),np.median(b)],
                            [np.median(bl),np.median(gr),np.median(re)]],
                            ((x+x+w)//2,(y+y+h)//2),(0,0,0),listcolor[i],[0,0,0,0,0]])
            cv2.putText(imgd, str(c), (x,y), cv2.FONT_HERSHEY_SIMPLEX,  
                        2, (0,0,0), 5, cv2.LINE_AA)
            cv2.drawContours(imgd, ob,-1,(np.median(ob_h),np.median(ob_s),np.median(ob_v)) , thickness=20)
        cv2.destroyAllWindows()
        suspectcolor=[]
        mean_rl=[[5,100,147],[127,130,150],[176,176,180]]
        mean_ru=[[174,100,147],[127,150,127],[99,95,147]]
        mean_yellow=[[26,100,147],[130,128,150],[142,154,149]]
        mean_green=[[55,100,147],[128,124,130],[125,135,103]]
        mean_blue=[[110,100,147],[128,127,110],[141,100,63]]
        mean_black=[[50,10,147],[50,128,128],[89,89,87]]
        listmeancolor=[mean_rl,mean_ru,mean_yellow,mean_green,mean_blue,mean_black]
        for item in listobj:
            rl_hsv = ((1-(abs(item[1][0][0]-mean_rl[0][0]))/180)*1+(1-(abs(item[1][0][1]-mean_rl[0][1]))/255)*1+(1-(abs(item[1][0][2]-mean_rl[0][2]))/255)*0.5)/(1+1+0.5)
            ru_hsv = ((1-(abs(item[1][0][0]-mean_ru[0][0]))/180)*1+(1-(abs(item[1][0][1]-mean_ru[0][1]))/255)*1+(1-(abs(item[1][0][2]-mean_ru[0][2]))/255)*0.5)/(1+1+0.5)
            rl_lab = ((1-(abs(item[1][1][0]-mean_rl[1][0]))/255)*0.5+(1-(abs(item[1][1][1]-mean_rl[1][1]))/255)*1+(1-(abs(item[1][1][2]-mean_rl[1][2]))/255)*1)/(0.5+1+1)
            ru_lab = ((1-(abs(item[1][1][0]-mean_ru[1][0]))/255)*0.5+(1-(abs(item[1][1][1]-mean_ru[1][1]))/255)*1+(1-(abs(item[1][1][2]-mean_ru[1][2]))/255)*1)/(0.5+1+1)
            rl_bgr = ((1-(abs(item[1][2][0]-mean_rl[2][0]))/255)*1+(1-(abs(item[1][2][1]-mean_rl[2][1]))/255)*1+(1-(abs(item[1][2][2]-mean_rl[2][2]))/255)*1)/(1+1+1)
            ru_bgr = ((1-(abs(item[1][2][0]-mean_ru[2][0]))/255)*1+(1-(abs(item[1][2][1]-mean_ru[2][1]))/255)*1+(1-(abs(item[1][2][2]-mean_ru[2][2]))/255)*1)/(1+1+1)
            p_ru = (ru_hsv+ru_lab+ru_bgr*0.2)/2.2
            p_rl = (rl_hsv+rl_lab+rl_bgr*0.2)/2.2
            p_r = max(p_rl,p_ru)

            # print(rl_hsv,ru_hsv)
            item[5][0]=p_r
            ye_hsv = ((1-(abs(item[1][0][0]-mean_yellow[0][0]))/180)*1+(1-(abs(item[1][0][1]-mean_yellow[0][1]))/255)*1+(1-(abs(item[1][0][2]-mean_yellow[0][2]))/255)*0.5)/(1+1+0.5)
            ye_lab = ((1-(abs(item[1][1][0]-mean_yellow[1][0]))/255)*0.5+(1-(abs(item[1][1][1]-mean_yellow[1][1]))/255)*1+(1-(abs(item[1][1][2]-mean_yellow[1][2]))/255)*1)/(0.5+1+1)
            ye_bgr = ((1-(abs(item[1][2][0]-mean_yellow[2][0]))/255)*1+(1-(abs(item[1][2][1]-mean_yellow[2][1]))/255)*1+(1-(abs(item[1][2][2]-mean_yellow[2][2]))/255)*1)/(1+1+1)
            p_ye = (ye_hsv+ye_bgr*0.2+ye_lab)/2.2
            item[5][1]=p_ye
            gre_hsv = ((1-(abs(item[1][0][0]-mean_green[0][0]))/180)*1+(1-(abs(item[1][0][1]-mean_green[0][1]))/255)*1+(1-(abs(item[1][0][2]-mean_green[0][2]))/255)*0.5)/(1+1+0.5)
            gre_lab = ((1-(abs(item[1][1][0]-mean_green[1][0]))/255)*0.5+(1-(abs(item[1][1][1]-mean_green[1][1]))/255)*1+(1-(abs(item[1][1][2]-mean_green[1][2]))/255)*1)/(0.5+1+1)
            gre_bgr = ((1-(abs(item[1][2][0]-mean_green[2][0]))/255)*1+(1-(abs(item[1][2][1]-mean_green[2][1]))/255)*1+(1-(abs(item[1][2][2]-mean_green[2][2]))/255)*1)/(1+1+1)
            p_gre = (gre_hsv+gre_bgr*0.2+gre_lab)/2.2
            item[5][2]=p_gre
            blue_hsv = ((1-(abs(item[1][0][0]-mean_blue[0][0]))/180)*1+(1-(abs(item[1][0][1]-mean_blue[0][1]))/255)*1+(1-(abs(item[1][0][2]-mean_blue[0][2]))/255)*0.5)/(1+1+0.5)
            blue_lab = ((1-(abs(item[1][1][0]-mean_blue[1][0]))/255)*0.5+(1-(abs(item[1][1][1]-mean_blue[1][1]))/255)*1+(1-(abs(item[1][1][2]-mean_blue[1][2]))/255)*1)/(0.5+1+1)
            blue_bgr = ((1-(abs(item[1][2][0]-mean_blue[2][0]))/255)*1+(1-(abs(item[1][2][1]-mean_blue[2][1]))/255)*1+(1-(abs(item[1][2][2]-mean_blue[2][2]))/255)*1)/(1+1+1)
            p_blue = (blue_hsv+blue_bgr*0.2+blue_lab)/2.2
            item[5][3]=p_blue
            black_hsv = ((1-(abs(item[1][0][0]-mean_black[0][0]))/180)*0.25+(1-(abs(item[1][0][1]-mean_black[0][1]))/255)*2.5+(1-(abs(item[1][0][2]-mean_black[0][2]))/255)*0.5)/(0.25+2.5+0.5)
            black_lab = ((1-(abs(item[1][1][0]-mean_black[1][0]))/255)*0.5+(1-(abs(item[1][1][1]-mean_black[1][1]))/255)*1+(1-(abs(item[1][1][2]-mean_black[1][2]))/255)*1)/(0.5+1+1)
            black_bgr = ((1-(abs(item[1][2][0]-mean_black[2][0]))/255)*1+(1-(abs(item[1][2][1]-mean_black[2][1]))/255)*1+(1-(abs(item[1][2][2]-mean_black[2][2]))/255)*1)/(1+1+1)
            p_black = (black_hsv+black_bgr*0.2+black_lab)/2.2
            item[5][4]=p_black
            print(item[1], item[5],item[2],item[5].index(max(item[5])))
            item[0]=item[5].index(max(item[5]))
            suspectcolor.append(item[0])
        for i in range(5):
            colo={}
            if(suspectcolor.count(i)>1):
                for inde,j in enumerate(listobj):
                    print('j',j[0])
                    if j[0]==i :
                        x = j
                        colo[inde] =sorted(x[5],reverse=True)
                for inde in colo:
                    colo[inde] = (colo[inde][0]-colo[inde][1],colo[inde])
                while(len(colo)>0):    
                    # print(colo)
                    # print(operator.itemgetter(1)(colo))
                    print(max(colo.items(), key=operator.itemgetter(1))[0])
                    del colo[max(colo.items(), key=operator.itemgetter(1))[0]]
                    print(colo)
                    for ab in colo:
                        print("aa",listobj[ab][5].index(colo[ab][1][1]))
                        listobj[ab][0]=listobj[ab][5].index(colo[ab][1][1])

        for i in listobj :
            if i[0] == 0:
                i[0]='red'
                
            if i[0] == 1 :
                i[0]='yellow'
            if i[0] == 2:
                i[0]='green'
            if i[0] == 3:
                i[0]='blue'
            if i[0] == 4:
                i[0]='black'
            print(i[0],i[1][0])
        
        self.edgemask=listobj
        return listobj
    def readtext(self,color,line=0):
        f = open("./imagesnap/"+color+".txt","r")
        msg = list(map(int,f.readline().split(',')))
        return msg
    def sethsv(self):
        msg = self.readtext('red_L')
        msg_u = self.readtext('red_U')
        self.red = [[msg[0],msg[2],msg[4]],[msg[1],msg[3],msg[5]],[149,70,0],[180,255,255]]#160-10
        msg = self.readtext('yellow')
        self.yellow = [[msg[0],msg[2],msg[4]],[msg[1],msg[3],msg[5]]]
        msg = self.readtext('green')
        self.green = [[msg[0],msg[2],msg[4]],[msg[1],msg[3],msg[5]]]
        msg = self.readtext('blue')
        self.blue = [[msg[0],msg[2],msg[4]],[msg[1],msg[3],msg[5]]]
        msg = self.readtext('black')
        self.black = [[msg[0],msg[2],msg[4]],[msg[1],msg[3],msg[5]]]
        msg = self.readtext('background')
        self.white = [[msg[0],msg[2],msg[4]],[msg[1],msg[3],msg[5]]]#default value = 155
        self.color = [self.red,self.yellow,self.green,self.blue,self.black,self.white]
    
    def drawbox(self,l,u,s=1,placemode='centroid'):
        pos_list = {}
        self.drawgrid(self.imgsnap,25,25)
        for i in range(l,u,s):
            
            if i !=5 :
                cv2.imshow("fma",self.finalmask[i])
                if (self.debug):
                    cv2.waitKey()
                _,contour,hir=cv2.findContours(self.finalmask[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                M = cv2.moments(contour[0])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                x,y,w,h=cv2.boundingRect(contour[0])
                if(placemode=='centroid'):
                    self.color_pos[i]=[cX,cY]
                elif(placemode=='center'):
                    self.color_pos[i]=[x+int(w/2),y+int(h/2)]
                cv2.drawContours(self.imgsnap,contour[0], -1, (255, 255, 0), 4)
                if i ==0 :
                    cv2.circle(self.imgsnap, (cX, cY), 3, (0,0,128), 2)
                    cv2.circle(self.imgsnap,(self.color_pos[i][0],self.color_pos[i][1]),int(20*601/250),(0,0,255),2)
                    pos_list['red']= ([self.color_pos[i][0],self.color_pos[i][1]])
                if i ==1 :
                    cv2.circle(self.imgsnap, (cX, cY), 3, (0,128,128), 2)
                    cv2.circle(self.imgsnap,(self.color_pos[i][0],self.color_pos[i][1]),int(20*601/250),(0,255,255),2)
                    pos_list['yellow']=([self.color_pos[i][0],self.color_pos[i][1]])
                if i ==2 :
                    cv2.circle(self.imgsnap, (cX, cY), 3, (0,128,0), 2)
                    cv2.circle(self.imgsnap,(self.color_pos[i][0],self.color_pos[i][1]),int(20*601/250),(0,255,0),2)
                    pos_list['green']= ([self.color_pos[i][0],self.color_pos[i][1]])
                if i ==3 :
                    cv2.circle(self.imgsnap, (cX, cY), 3, (128,0,0), 2)
                    cv2.circle(self.imgsnap,(self.color_pos[i][0],self.color_pos[i][1]),int(20*601/250),(255,0,0),2)
                    pos_list['blue']= ([self.color_pos[i][0],self.color_pos[i][1]])
                if i ==4 :
                    cv2.circle(self.imgsnap, (cX, cY), 3, (128,128,128), 2)
                    cv2.circle(self.imgsnap,(self.color_pos[i][0],self.color_pos[i][1]),int(20*601/250),(0,0,0),2)
                    pos_list['black']=([self.color_pos[i][0],self.color_pos[i][1]])
        return pos_list
    def doublecheck(self,m=0):
        sequence = ['red','yellow','green','blue','black']
        error = []
        newlist=[]
        for i in sequence:
            for j in range(len(self.edgemask)):

                if(type(self.edgemask[j][0]) == type('s') ):
                    if(i==self.edgemask[j][0]):

                        newlist.append(self.edgemask[j])
                        break
                    elif(len(self.edgemask)==j+1):
                        error.append(i)
        print('error',error)
        for i in error:
            for j in range(len(self.edgemask)):
                if('unknow'==self.edgemask[j][0]):

                    if self.edgemask[j][2]!=(0,0):
                        if((self.edgemask[j][1][0]<=20 and self.edgemask[j][1][0]>=0 )or(self.edgemask[j][1][0]<=180 and self.edgemask[j][1][0]>=140 ) ):
                            self.edgemask[j][0]='red'
                            newlist.append(self.edgemask[j])
                            print(self.edgemask[j][1])
                            error.remove('red')
                            break
                        elif(self.edgemask[j][1][1]<=17 and self.edgemask[j][1][1]>=0  ):
                            self.edgemask[j][0]='black'
                            newlist.append(self.edgemask[j])
                            
                            error.remove('black')
                            break                   
                        elif(self.edgemask[j][1][0]<=45 and self.edgemask[j][1][0]>=21  ):
                            self.edgemask[j][0]='yellow'
                            newlist.append(self.edgemask[j])
                            print('yellowhere')
                            error.remove('yellow')
                            break
                        elif(self.edgemask[j][1][0]<=90 and self.edgemask[j][1][0]>=46  ):
                            self.edgemask[j][0]='green'
                            newlist.append(self.edgemask[j])
                            
                            error.remove('green')
                            break
                        elif(self.edgemask[j][1][0]<=120 and self.edgemask[j][1][0]>=91  ):
                            self.edgemask[j][0]='blue'
                            newlist.append(self.edgemask[j])
                            
                            error.remove('blue')
                            break
        print('error2',error)
        for j in error:
            for i in self.edgemask:
                if(i[0]=='unknow'):
                    i[0]=j
                    # i[2]=(0,0)
                    newlist.append(i)
                    break
        
        uselist=[0,0,0,0,0]
        for i in newlist:
            if(i[0]=='red'):
                uselist[0]=i
            if(i[0]=='yellow'):
                uselist[1]=i
            if(i[0]=='green'):
                uselist[2]=i
            if(i[0]=='blue'):
                uselist[3]=i
            if(i[0]=='black'):
                uselist[4]=i
        self.edgemask = uselist
        for i in range(len(self.edgemask)):
            for i in range(5):
                if(self.edgemask[i][2]==(0,0)):
                    self.finalmask[i]=self.mask[i]
                elif(m==1):
                    
                    self.finalmask[i] = cv2.bitwise_and(self.mask[5],self.edgemask[i][4])
                    self.finalmask[i] = cv2.bitwise_or(self.mask[i],self.edgemask[i][4])
                elif(m==0):
                    
                    self.finalmask[i] =self.edgemask[i][4]
    def contrast(self,image,brightness,contrast):
        img = image  # mandrill reference image from USC SIPI
        s = self.shape[0]
        brightness = brightness-128

        def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

            if brightness != 0:
                if brightness > 0:
                    shadow = brightness
                    highlight = 255
                else:
                    shadow = 0
                    highlight = 255 + brightness
                alpha_b = (highlight - shadow)/255
                gamma_b = shadow

                buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
            else:
                buf = input_img.copy()

            if contrast != 0:
                f = 131*(contrast + 127)/(127*(131-contrast))
                alpha_c = f
                gamma_c = 127*(1-f)

                buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

            return buf
        blist = [brightness] # list of brightness values
        clist = [contrast] # list of contrast values

        out = np.zeros((self.shape[1],self.shape[0],self.shape[2]), dtype = np.uint8)

        for i, b in enumerate(blist):
            c = clist[i]
            row = s*int(i/3)
            col = s*(i%3)
            out[row:row+s, col:col+s] = apply_brightness_contrast(img, b, c)
        return out
    def drawgrid(self,img,width,height):
        for j in range(0,self.shape[1],width):
            for i in range(0,self.shape[0],height):
                cv2.line(img, (i, 0), (i, self.shape[1]), (120, 120, 120),1)
                cv2.line(img, (0, j), (self.shape[0], j), (120, 120, 120),1)
def m(image,imagesnap,mask=[],maskmode=0,placemode='centroid',debug=False):
    img = Cards(image,(601,601,3),0,imagesnap,debug=debug)
    img.incard_mask=mask
    img.sethsv()
    img._setwhite()
    img.edgedetectranking(imagesnap)
    img.hsvsegment()
    if (maskmode != 3):
        img.doublecheck(maskmode)
    else:
        img.finalmask[0]=img.mask[0]
        img.finalmask[1]=img.mask[1]
        img.finalmask[2]=img.mask[2]
        img.finalmask[3]=img.mask[3]
        img.finalmask[4]=img.mask[4]
    color_pos = img.drawbox(0,6,placemode=placemode)
    time = 1500
    if(debug):
        cv2.imshow('red',img.finalmask[0])
        cv2.imshow('yellow',img.finalmask[1])
        cv2.imshow('green',img.finalmask[2])
        cv2.imshow('blue',img.finalmask[3])
        cv2.imshow('black_wb',img.finalmask[4])
        cv2.imshow('blackcolor',img.mask[4])
        cv2.imshow('white',img.mask[5])
        time=0
    cv2.imshow('image_box',img.imgsnap)
    cv2.waitKey(time)
    cv2.destroyAllWindows()
    return color_pos