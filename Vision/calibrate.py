import cv2
import numpy as np
from . import cards
from . import camera
import time
import imutils
from colorama import init
from termcolor import cprint 
import sys
init(strip=not sys.stdout.isatty())

calib_state = 1
def perspective(cammode=0):
    global calib_state
    cam = camera.Camera(1280,720,True)
    video=cam.open(cammode)
    cam.createcardspacetrackbar()
    while(calib_state == 2):
        ret,frame=video.read()
        frame=cam._applyCalibate(frame)
        x1,x2,x3,x4,y1,y2,y3,y4=cam.readcardspacetrackerbar()
        per_frame = cam.appplyPerspective(frame,pts=((x1, y1),(x2, y2),(x3, y3),(x4, y4)),size=(625,625))
        cv2.imshow('frame',frame)
        cv2.imshow('perspective frame',per_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file1 = open("./imagesnap/per_point.txt","w")
            msg="{},{},{},{},{},{},{},{}".format(x1,x2,x3,x4,y1,y2,y3,y4)
            file1.write(str(msg))
            file1.close()
            print("save position\n",str(msg))
            calib_state = 2
        elif key == ord('m'):
            calib_state = 1
            break
        elif key == ord('e'):
            calib_state = 0
            break
    video.release()
    cv2.destroyAllWindows()
def backgroundcalib(cammode=0):
    print("ddd")
    global calib_state
    cam = camera.Camera(1280,720,True)
    video=cam.open(cammode)
    def nothing(x):
            pass
    cv2.namedWindow("background")
    cv2.namedWindow("background_U")
    # blank_image = np.zeros(shape=[601,601, 3], dtype=np.uint8)
    print("eher")
    f = open("./imagesnap/"+"background"+".txt","r")
    msg = list(map(int,f.readline().split(',')))
    print(msg)
    f.close()
    u = open("./imagesnap/"+"background_U"+".txt","r")
    msgu = list(map(int,u.readline().split(',')))
    print(msg)
    u.close()
    cv2.createTrackbar('lh_min', "background",msg[0],180,nothing)
    cv2.createTrackbar('lh_max', "background",msg[1],180,nothing)
    cv2.createTrackbar('ls_min', "background",msg[2],256,nothing)
    cv2.createTrackbar('ls_max', "background",msg[3],256,nothing)
    cv2.createTrackbar('lv_min', "background",msg[4],256,nothing)
    cv2.createTrackbar('lv_max', "background",msg[5],256,nothing)
    cv2.createTrackbar('uh_min', "background_U",msgu[0],180,nothing)
    cv2.createTrackbar('uh_max', "background_U",msgu[1],180,nothing)
    cv2.createTrackbar('us_min', "background_U",msgu[2],256,nothing)
    cv2.createTrackbar('us_max', "background_U",msgu[3],256,nothing)
    cv2.createTrackbar('uv_min', "background_U",msgu[4],256,nothing)
    cv2.createTrackbar('uv_max', "background_U",msgu[5],256,nothing)
    while(calib_state == 5):
        ret,frame=video.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lhmin = cv2.getTrackbarPos('lh_min',"background")
        lhmax = cv2.getTrackbarPos('lh_max',"background")
        lsmin = cv2.getTrackbarPos('ls_min',"background")
        lsmax = cv2.getTrackbarPos('ls_max',"background")
        lvmin = cv2.getTrackbarPos('lv_min',"background")
        lvmax = cv2.getTrackbarPos('lv_max',"background")
        uhmin = cv2.getTrackbarPos('uh_min',"background_U")
        uhmax = cv2.getTrackbarPos('uh_max',"background_U")
        usmin = cv2.getTrackbarPos('us_min',"background_U")
        usmax = cv2.getTrackbarPos('us_max',"background_U")
        uvmin = cv2.getTrackbarPos('uv_min',"background_U")
        uvmax = cv2.getTrackbarPos('uv_max',"background_U")
        lower_blue = np.array([lhmin,lsmin,lvmin])
        upper_blue = np.array([lhmax,lsmax,lvmax])

        maskl = cv2.inRange(hsv,lower_blue, upper_blue)
        lower_blue = np.array([uhmin,usmin,uvmin])
        upper_blue = np.array([uhmax,usmax,uvmax])
        masku = cv2.inRange(hsv,lower_blue, upper_blue)
        mask = cv2.bitwise_or(maskl,masku)
        result = cv2.bitwise_and(frame,frame,mask = mask)
        cv2.imshow('frame',result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file1 = open("./imagesnap/"+"background"+".txt","w")
            msg="{},{},{},{},{},{}".format(lhmin,lhmax,lsmin,lsmax,lvmin,lvmax)
            file1.write(str(msg))
            file1.close()
            print("save "+"background"+"range\n",str(msg))
            file2 = open("./imagesnap/"+"background_U"+".txt","w")
            msg="{},{},{},{},{},{}".format(uhmin,uhmax,usmin,usmax,uvmin,uvmax)
            file2.write(str(msg))
            file2.close()
            print("save "+"background_U"+"range\n",str(msg))
            calib_state = 5
        # elif key == ord('r'):
        #     file1 = open("./imagesnap/"+mode+".txt","a")
        #     msg="{},{},{},{},{},{}".format(hmin,hmax,smin,smax,vmin,vmax)
        #     file1.write(str(msg))
        #     file1.close()
        #     print("save "+mode+" range\n",str(msg))
        #     calib_state = 3
        elif key == ord('m'):
            calib_state = 1
            break
        elif key == ord('e'):
            calib_state = 0
            break
        
    video.release()
    cv2.destroyAllWindows()
def colorrange(mode,cammode=0):
    global calib_state
    cam = camera.Camera(1280,720,True)
    video=cam.open(cammode)
    def nothing(x):
            pass
    cv2.namedWindow(mode)
    # blank_image = np.zeros(shape=[601,601, 3], dtype=np.uint8)
    f = open("./imagesnap/"+mode+".txt","r")
    msg = list(map(int,f.readline().split(',')))
    print(msg)
    f.close()
    cv2.createTrackbar('h_min', mode,msg[0],180,nothing)
    cv2.createTrackbar('h_max', mode,msg[1],180,nothing)
    cv2.createTrackbar('s_min', mode,msg[2],256,nothing)
    cv2.createTrackbar('s_max', mode,msg[3],256,nothing)
    cv2.createTrackbar('v_min', mode,msg[4],256,nothing)
    cv2.createTrackbar('v_max', mode,msg[5],256,nothing)
    
    while(calib_state == 3):
        ret,frame=video.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hmin = cv2.getTrackbarPos('h_min',mode)
        hmax = cv2.getTrackbarPos('h_max',mode)
        smin = cv2.getTrackbarPos('s_min',mode)
        smax = cv2.getTrackbarPos('s_max',mode)
        vmin = cv2.getTrackbarPos('v_min',mode)
        vmax = cv2.getTrackbarPos('v_max',mode)
        lower_blue = np.array([hmin,smin,vmin])
        upper_blue = np.array([hmax,smax,vmax])

        mask = cv2.inRange(hsv,lower_blue, upper_blue)
        result = cv2.bitwise_and(frame,frame,mask = mask)
        cv2.imshow('frame'+mode,result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file1 = open("./imagesnap/"+mode+".txt","w")
            msg="{},{},{},{},{},{}".format(hmin,hmax,smin,smax,vmin,vmax)
            file1.write(str(msg))
            file1.close()
            print("save "+mode+"range\n",str(msg))
            calib_state = 3
        elif key == ord('r'):
            file1 = open("./imagesnap/"+mode+".txt","a")
            msg="{},{},{},{},{},{}".format(hmin,hmax,smin,smax,vmin,vmax)
            file1.write(str(msg))
            file1.close()
            print("save "+mode+" range\n",str(msg))
            calib_state = 3
        elif key == ord('m'):
            calib_state = 1
            break
        elif key == ord('e'):
            calib_state = 0
            break
        
    video.release()
    cv2.destroyAllWindows()
def contrast(image,brightness,contrast):
    # print('h')
    brightness=brightness-128
    img = image  # mandrill reference image from USC SIPI
    s = img.shape[0]
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


    # blist = [0, -127, 127,   0,  0, 64] # list of brightness values
    # clist = [0,    0,   0, -64, 64, 64] # list of contrast values
    blist = [brightness] # list of brightness values
    clist = [contrast] # list of contrast values

    out = np.zeros((img.shape[0],img.shape[1],img.shape[2]), dtype = np.uint8)

    for i, b in enumerate(blist):
        c = clist[i]

        row = img.shape[1] *int(i/3)
        col = img.shape[0]*(i%3)

        out[row:row+img.shape[1], col:col+img.shape[0]] = apply_brightness_contrast(img, b, c)

    # cv2.imshow('out',out)
    # cv2.waitKey()
    return out
def snap(camport = 0,flip = 0,colormode = 0):
    cam = camera.Camera(1280,720,True) # (width,height,calibated)
    video=cam.open(camport) #(camera chanel)
    cam.createcardspacetrackbar()
    video_status=1
    setper=1 # Set 0 to select edge of card space for Perspective View, Set 1 to load card space edge form file
    state = 0
    time_start = 0

    # print("Press Q to quit \nPress S to set cardspace edge")
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

        frame=cam._applyCalibate(frame)
        per_frame = cam.appplyPerspective(frame,pts=((x1, y1),(x2, y2),(x3, y3),(x4, y4)),size=(625,625)) # edit size for cardspace ratio
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
        # cam.show('frame',frame)
        # cam.show('perspective_frame',per_frame)
        # cv2.imshow('Mask',pre_proc)
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
    # cv2.destroyAllWindows()
    # cv2.imshow('snap',snap_card)
    return snap_card
def contrastcalib(cammode=0):
    imgsnap = snap(cammode)
    conimgsnap=imgsnap
    cv2.destroyAllWindows()
    global calib_state
    cv2.namedWindow('contrast background')
    cv2.namedWindow('contrast background2')
    bg = open("./imagesnap/"+"con_background"+".txt","r")
    msg_bg = list(map(int,bg.readline().split(',')))
    bg.close()
    b=msg_bg[0]
    c = msg_bg[1]

    def nothing(x):

        pass
    
    
    cv2.createTrackbar('brightness', 'contrast background',msg_bg[0],180,nothing)
    cv2.createTrackbar('contrast', 'contrast background',msg_bg[1],131,nothing)
    cv2.createTrackbar('h_min', 'contrast background',msg_bg[2],180,nothing)
    cv2.createTrackbar('h_max', 'contrast background',msg_bg[3],180,nothing)
    cv2.createTrackbar('s_min', 'contrast background',msg_bg[4],255,nothing)
    cv2.createTrackbar('s_max', 'contrast background',msg_bg[5],255,nothing)
    cv2.createTrackbar('v_min', 'contrast background',msg_bg[6],255,nothing)
    cv2.createTrackbar('v_max', 'contrast background',msg_bg[7],255,nothing)
    # print(calib_state)

    bold =0
    cold = 0
    while(1):
        hsv = cv2.cvtColor(conimgsnap,cv2.COLOR_BGR2HSV)
        b = cv2.getTrackbarPos('brightness','contrast background')
        c = cv2.getTrackbarPos('contrast','contrast background')
        hmin = cv2.getTrackbarPos('h_min','contrast background')
        hmax = cv2.getTrackbarPos('h_max','contrast background')
        smin = cv2.getTrackbarPos('s_min','contrast background')
        smax = cv2.getTrackbarPos('s_max','contrast background')
        vmin = cv2.getTrackbarPos('v_min','contrast background')
        vmax = cv2.getTrackbarPos('v_max','contrast background')
        lower_blue = np.array([hmin,smin,vmin])
        upper_blue = np.array([hmax,smax,vmax])

        mask = cv2.inRange(hsv,lower_blue, upper_blue)
        result = cv2.bitwise_and(conimgsnap,conimgsnap,mask = mask)
        if(b==bold and c==cold ):
            pass
        else:
            conimgsnap = contrast(imgsnap,b,c)
            bold = b
            cold = c
        cv2.imshow('frame'+'contrast background',conimgsnap)
        cv2.imshow('mask',result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file1 = open("./imagesnap/"+"con_background"+".txt","w")
            msg="{},{},{},{},{},{},{},{}".format(b,c,hmin,hmax,smin,smax,vmin,vmax)
            file1.write(str(msg))
            file1.close()
            print("save "+'contrast background'+" range\n",str(msg))
            calib_state = 4
        
        elif key == ord('m'):
            calib_state = 1
            break
        elif key == ord('e'):
            calib_state = 0
            break
        
    # video.release()
    cv2.destroyAllWindows()
def edgebackground(cammode=0):
    
    global calib_state
    cam = camera.Camera(1280,720,True)
    video=cam.open(cammode)
    def nothing(x):
            pass
    cv2.namedWindow("edge")
    cv2.namedWindow("background_U")
    # blank_image = np.zeros(shape=[601,601, 3], dtype=np.uint8)
    f = open("./imagesnap/"+"background"+".txt","r")
    msg = list(map(int,f.readline().split(',')))
    print(msg)
    f.close()
    u = open("./imagesnap/"+"background_U"+".txt","r")
    msgu = list(map(int,u.readline().split(',')))
    print(msg)
    u.close()
    cv2.createTrackbar('lh_min', "background",msg[0],180,nothing)
    cv2.createTrackbar('lh_max', "background",msg[1],180,nothing)
    cv2.createTrackbar('ls_min', "background",msg[2],256,nothing)
    cv2.createTrackbar('ls_max', "background",msg[3],256,nothing)
    cv2.createTrackbar('lv_min', "background",msg[4],256,nothing)
    cv2.createTrackbar('lv_max', "background",msg[5],256,nothing)
    cv2.createTrackbar('uh_min', "background_U",msgu[0],180,nothing)
    cv2.createTrackbar('uh_max', "background_U",msgu[1],180,nothing)
    cv2.createTrackbar('us_min', "background_U",msgu[2],256,nothing)
    cv2.createTrackbar('us_max', "background_U",msgu[3],256,nothing)
    cv2.createTrackbar('uv_min', "background_U",msgu[4],256,nothing)
    cv2.createTrackbar('uv_max', "background_U",msgu[5],256,nothing)
    while(calib_state == 5):
        ret,frame=video.read()
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lhmin = cv2.getTrackbarPos('lh_min',"background")
        lhmax = cv2.getTrackbarPos('lh_max',"background")
        lsmin = cv2.getTrackbarPos('ls_min',"background")
        lsmax = cv2.getTrackbarPos('ls_max',"background")
        lvmin = cv2.getTrackbarPos('lv_min',"background")
        lvmax = cv2.getTrackbarPos('lv_max',"background")
        uhmin = cv2.getTrackbarPos('uh_min',"background_U")
        uhmax = cv2.getTrackbarPos('uh_max',"background_U")
        usmin = cv2.getTrackbarPos('us_min',"background_U")
        usmax = cv2.getTrackbarPos('us_max',"background_U")
        uvmin = cv2.getTrackbarPos('uv_min',"background_U")
        uvmax = cv2.getTrackbarPos('uv_max',"background_U")
        lower_blue = np.array([lhmin,lsmin,lvmin])
        upper_blue = np.array([lhmax,lsmax,lvmax])

        maskl = cv2.inRange(hsv,lower_blue, upper_blue)
        lower_blue = np.array([uhmin,usmin,uvmin])
        upper_blue = np.array([uhmax,usmax,uvmax])
        masku = cv2.inRange(hsv,lower_blue, upper_blue)
        mask = cv2.bitwise_or(maskl,masku)
        result = cv2.bitwise_and(frame,frame,mask = mask)
        cv2.imshow('frame',result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file1 = open("./imagesnap/"+"background"+".txt","w")
            msg="{},{},{},{},{},{}".format(lhmin,lhmax,lsmin,lsmax,lvmin,lvmax)
            file1.write(str(msg))
            file1.close()
            print("save "+"background"+"range\n",str(msg))
            file2 = open("./imagesnap/"+"background_U"+".txt","w")
            msg="{},{},{},{},{},{}".format(uhmin,uhmax,usmin,usmax,uvmin,uvmax)
            file2.write(str(msg))
            file2.close()
            print("save "+"background_U"+"range\n",str(msg))
            calib_state = 5
        # elif key == ord('r'):
        #     file1 = open("./imagesnap/"+mode+".txt","a")
        #     msg="{},{},{},{},{},{}".format(hmin,hmax,smin,smax,vmin,vmax)
        #     file1.write(str(msg))
        #     file1.close()
        #     print("save "+mode+" range\n",str(msg))
        #     calib_state = 3
        elif key == ord('m'):
            calib_state = 1
            break
        elif key == ord('e'):
            calib_state = 0
            break
        
    video.release()
    cv2.destroyAllWindows()
def maincalib(camport = 0):
    global calib_state
    calib_state = 1
    while(calib_state > 0):
        m = input("1: Set Card Space\n2: Set Color Range\n3: Set Contrast \n4: Background Check\n5: Back\n>>> ")
        if(m =='1'):
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('S: Save this configuration', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('M: Back to calibration menu', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('E: Back to main menu', 'yellow')
            calib_state = 2
            perspective(cammode = camport)
        elif(m =='2'):
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('S: Save this configuration', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('M: Back to calibration menu', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('E: Back to main menu', 'yellow')
            c_m = input("Pick a color below\n\tbackground\n\tred_L\n\tred_U\n\tyellow\n\tgreen\n\tblue\n\tblack\n>>> ")
            calib_state = 3
            colorrange(c_m, cammode = camport)
        elif(m =='3'):
            # print(m)
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('S: Save this configuration', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('M: Back to calibration menu', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('E: Back to main menu', 'yellow')
            
            calib_state = 4
            contrastcalib( cammode = camport)
        elif(m =='4'):
            # print(m)
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('S: Save this configuration', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('M: Back to calibration menu', 'yellow')
            cprint (' ', 'white', 'on_yellow',end = ' ')
            cprint ('E: Back to main menu', 'yellow')
            
            calib_state = 5
            backgroundcalib( cammode = camport)
        elif(m =='5'):
            calib_state =0
        elif(m =='6'):
            calib_state =6
