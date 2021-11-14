from .com import Uart
from .utils import util as ut
from colorama import init
from termcolor import cprint 
import time, subprocess
import signal
import sys, math
import playsound
import pprint
import os

init(strip=not sys.stdout.isatty())
ut.sprint ("Reading config file....", 'normal')
try:
    config_file = open('./sheldonOS/sheldon.config', 'r')
    exec(config_file.read())
    pprint.pprint(config)
    print ('\n')
    # pprint.pprint(cmdLib)
    # print ('\n')
    config_file.close()

    help_file = open('./sheldonOS/sheldon.help', 'r')
    _help = help_file.read()
    help_file.close()
    # print (_help+'\n')
except:
    ut.sprint ('Something wrong with the config files, terminating...', 'warn')
    sys.exit(1)

class Sheldon:
    def __init__(self):
        self.default_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.__exit)
        self.config = config

        self.sheldonCore = Uart(device = "dspic33", port = self.config["xy_axis"], baudrate = 115200)
        self.zheldon = Uart(device = "f303k8", port = self.config["zg_axis"], baudrate = 115200)
        if not self.sheldonCore.status or not self.zheldon.status:
            ut.sprint ('Terminating...', 'warn')
            sys.exit(1)
        self.sheldonCore.reset()
        self.vias = 0
        self.__timeout__ = 0
        self.now = [0, 0, 0]
        self.__ps__ = 0

    def setTimeout(self, num):
        self.__timeout__ = num
    
    def psOn(self):
        ut.sprint ('Turning on powersupply', 'normal')
        self.__sendFrame__(cmdLib['psOn'])
        self.__ps__ = 1

    def psOff(self):
        ut.sprint ('Turning off powersupply', 'normal')
        self.__sendFrame__(cmdLib['psOff'])
        self.__ps__ = 0

    def __sendFrame__(self, cmd):
        [x, y, z, g, statexy, statez] = cmd
        for i in range(2):
            self.sheldonCore.write_line(ut.get_packet(x, y, statexy))
            if i <= 0:
                self.zheldon.write_line(ut.get_packet(z, g, statez))
            time.sleep(0.03)

    def home(self, axis = 0):
        if not axis:
            ut.sprint ('Homing all axis','normal')
            self.__sendFrame__(cmdLib['home'])
            self.sheldonClear()
            self.now = [0, 0, 0]
        elif axis == 1:
            ut.sprint ('Homing xy plane','normal')
            self.__sendFrame__(cmdLib['home_plane'])
            self.sheldonClear(axis)
            self.now[0] = 0
            self.now[1] = 0 
        elif axis == 2:
            ut.sprint ('Homing z axis','normal')
            self.__sendFrame__(cmdLib['home_z'])
            self.sheldonClear(axis)
            self.now[2] = 0
        time.sleep(0.5)

    def moveTo(self, xd = 0, yd = 0, zd = 0):
        dist = ut.eucDist(xd-self.now[0], yd-self.now[1])
        ang = math.atan2(xd-self.now[0], yd-self.now[1])
        ut.sprint ("R: {0:.2f}\tTh: {0:.2f}".format(dist, ang), 'cyan')
        # if dist <= 360 and dist >0:
        #     self.__sendFrame__([xd, yd, zd, 0, 2, 3])
        #     time.sleep(0.05)
        #     if dist > 180:
        #         time.sleep(0.03)
        # else:
        self.__sendFrame__([xd, yd, zd, 0, 3, 3])
        self.sheldonClear()
        self.now = [xd, yd, zd]
        ut.sprint ('Success                                 ', 'success')
        
    def grip(self):
        self.__sendFrame__(cmdLib['grip'])
        time.sleep(0.3)
        ut.sprint ('Success                                 ', 'success')

    def release(self):
        self.__sendFrame__(cmdLib['release'])
        time.sleep(0.035)
        ut.sprint ('Success                                 ', 'success')
    
    def manOp(self):
        ut.sprint ('Input 300 to exit this mode', 'normal')
        while 1:
            try:
                xd = int(input("xd>> "))
                if xd == 300: break 
                yd = int(input("yd>> "))
                if yd == 300: break 
                stxy = int(input("state_xy>> "))
                if stxy == 300: break 
                zd = int(input("zd>> "))
                if zd == 300: break 
                gd = int(input("gd>> "))
                if gd == 300: break 
                stzg = int(input("state_zg>> "))
                if stzg == 300: break 
                self.__sendFrame__([xd, yd, zd, gd, stxy, stzg])
            except:
                ut.sprint('There is a ploblem with your input', 'warn')
            

    def __reset(self, signum, frame):
        signal.signal(signal.SIGINT, self.default_sigint)
        self.psOff()
        
    def __exit(self, signum, frame):
        signal.signal(signal.SIGINT, self.default_sigint)
        print ('')
        ut.sprint ("Emergency Stopped. Returning to main menu...",'warn')
        self.psOff()
        self.sheldonCore.close()
        self.zheldon.close()
        subprocess.call('python main.py', shell = True)
        # sys.exit(1)
        
    def insertViaPoint(self, ls):
        vp = [0]
        for v in ls:
            vp.append(v)
        self.vias = vp
        
    def run(self):
        if self.vias not in ['', 0, None, []]:
            for vp in self.vias:
                if vp:
                    if len(vp) >= 3:
                        ut.sprint (vp, 'normal')
                        self.moveTo(vp[0], vp[1], vp[2])
                    elif vp == 'g':
                        ut.sprint ('Gripping', 'normal')
                        self.grip()
                    elif vp == 'r':
                        ut.sprint ('Releasing', 'normal')
                        self.release()
            ut.sprint('Finished all process', 'success')
        else: ut.sprint('No via points added', 'warn')

    def sing(self, track): 
        playsound.playsound('./sheldonOS/tracks/{}.mp3'.format(track), False)

    def sheldonClear(self, axis = 0):
        ts = time.time()
        if not axis:
            st_rdy = 0
            f1 = 0
            f2 = 0
            while st_rdy < 1:
                self.sheldonCore.flush()
                self.zheldon.flush()
                time.sleep(0.1)
                ret = self.sheldonCore.get_line()[0:2]
                retzg = self.zheldon.get_line()
                retz = self.zheldon.get_line()[0:2]
                a = 'xy plane {}\tz axis {} zdiag {}\r'.format(ret, retz, retzg)
                sys.stdout.write (a)
                # print(a)
                sys.stdout.flush()
                if time.time() - ts > self.__timeout__ and self.__timeout__ > 0:
                    ut.sprint ('Timeout. Terminating...                          ', 'warn')
                    self.psOff()
                    sys.exit(1)
                if ret in ['12', '22', '32'] and f1 == 0:
                    f1 = 1
                    st_rdy += 0.5
                if retz in ['12', '22', '32'] and f2 == 0:
                    f2 = 1
                    st_rdy += 0.5
            self.sheldonCore.flush_out()
            self.zheldon.flush_out()
        elif axis == 1:
            st_rdy = 0
            while not st_rdy:
                self.sheldonCore.flush()
                time.sleep(0.1)
                ret = self.sheldonCore.get_line()[0:2]
                a = 'xy plane {}\r'.format(ret)
                sys.stdout.write (a)
                sys.stdout.flush()
                if time.time() - ts > self.__timeout__ and self.__timeout__ > 0:
                    ut.sprint ('Timeout. Terminating...                          ', 'warn')
                    self.psOff()
                    sys.exit(1)                
                if ret in ['12', '22', '32']:
                    st_rdy = 1
            self.sheldonCore.flush_out()
        elif axis == 2:
            st_rdy = 0
            while not st_rdy:
                self.zheldon.flush()
                time.sleep(0.1)
                ret = self.zheldon.get_line()[0:2]
                a = 'z axis {}\r'.format(ret)
                sys.stdout.write (a)
                sys.stdout.flush()
                if time.time() - ts > self.__timeout__ and self.__timeout__ > 0:
                    ut.sprint ('Timeout. Terminating...                          ', 'warn')
                    self.psOff()
                    sys.exit(1)                
                if ret in ['12', '22', '32']:
                    st_rdy = 1
            self.zheldon.flush_out()
            
