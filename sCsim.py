from sheldonOS import util, Uart
import time
import threading
import sys



v_pic = Uart('vdspic33', 'COM1', 115200)
v_nuc = Uart('vnucleo', 'COM2', 115200)



def ret():
        v_pic.write_line(bytes([ord('1'), ord('2'), 10]))
        time.sleep(0.01)
        v_pic.write_line(bytes([ord('2'), ord('2'), 10]))
        time.sleep(0.01)
        v_nuc.write_line(bytes([ord('1'), ord('2'), 10]))
        time.sleep(0.01)
        v_nuc.write_line(bytes([ord('2'), ord('2'), 10]))
        time.sleep(0.01)
        threading.Timer(0.2, ret).start()
        


ret()
prev_a = 0
prev_b = 0
while 1:
        if prev_a == [254, 255, 0, 0, 0, 0 ,0]:
                exit()
                
        a = v_pic.get_raw()     
        if prev_a != a:
                print ('pic : ', end = ' ')
                for d in a:
                        print (str(d), end = '\t')
                print ('\n')

        b = v_nuc.get_raw()
        if prev_b != b:
                print ('nuc : ', end = ' ')
                for d in b:
                        print (str(d), end = '\t')
                print ('\n')

        
        
        prev_a = a
        prev_b = b
        