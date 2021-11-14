import cv2
import numpy as np
from . import cards
from . import camera
import time
import imutils
from . import colorsegment as segment
from . import calibrate
class Vision:
    @staticmethod
    def calcardpos(Color_pos):
        new_color_pos = Color_pos.copy()
        scale = 250/601
        for i in new_color_pos:
            new_color_pos[i][0]=int(new_color_pos[i][0]*scale*10)
            new_color_pos[i][1]=int(new_color_pos[i][1]*scale*10)
        return new_color_pos
    @staticmethod
    def calibrate(camport = 0):
        calibrate.maincalib(camport)
    
    @staticmethod
    def run(camport = 0,flip = 0,colormode = 0,maskmode=0,centermode=0,placemode='centroid',debug=False):
        cam = camera.Camera(1280,720,True) # (width,height,calibated)
        video=cam.open(camport) #(camera chanel)
        cam.createcardspacetrackbar()
        video_status=1
        setper=1 # Set 0 to select edge of card space for Perspective View, Set 1 to load card space edge form file
        state = 0
        time_start = 0
        print("Press Q to quit\n")
        while(video_status==1):
            ret,frame = video.read()
            # frame = cv2.flip(frame,0)
            if(setper == 0):
                x1,x2,x3,x4,y1,y2,y3,y4=cam.readcardspacetrackerbar()
            else:
                file1 = open("./imagesnap/per_point.txt","r")
                text=file1.readlines()
                text =text[0].split(",")
                
                x1,x2,x3,x4,y1,y2,y3,y4=int(text[0]),int(text[1]),int(text[2]),int(text[3]),int(text[4]),int(text[5]),int(text[6]),int(text[7])
                state = 1
                file1.close()

                cv2.destroyWindow("track")
                # print(x1,x2,x3,x4,y1,y2,y3,y4)
            frame=cam._applyCalibate(frame)
            per_frame = cam.appplyPerspective(frame,pts=((x1, y1),(x2, y2),(x3, y3),(x4, y4)),size=(601,601)) # edit size for cardspace ratio
            pre_proc = cards.preprocess_image(per_frame)
            cnts_sort, cnt_is_card = cards.find_cards(pre_proc)
            sq = np.zeros(shape=[601,601, 3], dtype=np.uint8)
            crop_image=0
            if (len(cnts_sort) != 0) and state ==1:
                card = []
                k = 0
                for i in range(len(cnts_sort)):
                    if (cnt_is_card[i] == 1):

                        card.append(cards.preprocess_card(cnts_sort[i],per_frame))
                        crop_image = per_frame[card[0].pts[1]:card[0].pts[1]+card[0].pts[3],card[0].pts[0]:card[0].pts[0]+card[0].pts[2]]
                        if(flip == 0):
                            sq=card[0].warp
                        else:
                        # cv2.imshow('c',sq)
                            try:
                                sq=cards.squareframe(card[0].pts[0],card[0].pts[1],card[0].pts[2],card[0].pts[3],card[0],crop_image)
                                # cv2.imshow('c',sq)
                            except:
                                pass
                        ksq=sq.copy()
                        per_frame = cards.draw_results(per_frame, card[k])
                        k = k + 1
                        pre_proc_sq = cards.preprocess_image(sq,1)
                        cnts_sort_sq, cnt_is_card_sq = cards.findscale_cards(pre_proc_sq)
                        if len(cnts_sort_sq) != 0:
                            card_sq=[]
                            k_sq=0
                            for i_sq in range(len(cnts_sort_sq)):
                                if (cnt_is_card_sq[i_sq] == 1):
                                    card_sq.append(cards.preprocess_card(cnts_sort_sq[i_sq],sq,1))
                                    clean_sq = sq
                                    if(time_start==0):
                                        start=time.time()
                                        now = 0
                                        time_start=1
                                    else:
                                        now=time.time()
                                    if(now-start>1):
                                        card_sq.append(cards.preprocess_card(cnts_sort_sq[i_sq],sq,1))
                                        ksq = cards.draw_results(ksq, card_sq[k_sq],1)
                                        snap_card=sq
                                        card_prop=card_sq[k_sq]
                                        video_status=0
                                        break
                                    k_sq = k_sq + 1
                                    cv2.imshow('thresh_sq',pre_proc_sq)
                            if (len(card_sq) != 0):
                                temp_cnts_sq = []
                                for i in range(len(card)):
                                    temp_cnts_sq.append(card_sq[i].contour)
                                cv2.drawContours(ksq,temp_cnts_sq, -1, (255,0,0), 2)
                        cv2.imshow('thresh_sq',pre_proc_sq)
                if (len(card) != 0):
                    temp_cnts = []
                    for i in range(len(card)):
                        temp_cnts.append(card[i].contour)
                    cv2.drawContours(per_frame,temp_cnts, -1, (255,0,0), 2)
            cam.show('frame',frame)
            cam.show('perspective_frame',per_frame)
            cv2.imshow('Mask',pre_proc)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                snap_card = sq
                video_status=0
            elif key == ord('s'):

                file1 = open("./imagesnap/per_point.txt","w")
                msg="{},{},{},{},{},{},{},{}".format(x1,x2,x3,x4,y1,y2,y3,y4)

                file1.write(str(msg))
                file1.close()
                print("save position\n",str(msg))
                cam.per_point=(x1,x2,x3,x4,y1,y2,y3,y4)
                state = 1
        video.release()
        cv2.destroyAllWindows()
        cv2.imshow('snap',snap_card)
        cv2.imwrite('./imagesnap/clean_sq1.jpg',snap_card)
        result = cv2.fastNlMeansDenoisingColored(snap_card,None,5,10,7,10)
        kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
        mask_snap=cards.preprocess_image(result)
        cnts = cv2.findContours(mask_snap.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(mask_snap.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        cv2.imshow('mas',mask)
        imageROI = cv2.bitwise_and(result, result,mask=mask)
        cv2.imshow('sanapa',imageROI)
        cv2.imwrite('./imagesnap/clean_sq.jpg',imageROI)
        color_pos = segment.m(imageROI,snap_card,mask,maskmode=maskmode,placemode=placemode,debug=debug)
        print(color_pos)
        world_color_pos = Vision.calcardpos(color_pos)
        print(world_color_pos)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        return world_color_pos