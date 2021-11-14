from sheldonOS import Sheldon, util as ut
import time, sys, signal, subprocess, os, random
from democoordinate import democo
from colorama import init
from termcolor import cprint 
init(strip=not sys.stdout.isatty())
from datetime import date
from Vision.Vision import Vision
today = date.today()

sheldon1 = Sheldon()

visout = 0

cups = [[75, 65, 420, 'red'], 
        [620, 65, 420, 'yellow'], 
        [1160, 65, 420, 'green'], 
        [1710, 65, 420, 'blue'], 
        [2270, 65, 420, 'black']]

square = [[250, 2750, 2750, 250, 250],
        [600, 600, 3000, 3000, 600],
        [3000, 3000, 3000, 3000, 3000]]

def getVis(n):
        from Vision.Vision import Vision
        return Vision.run(camport = 0,flip = 0, colormode = 0, maskmode = n, placemode = 'centroid')

def reset():
        sheldon1.psOn()
        sheldon1.release()
        sheldon1.home(2)
        sheldon1.home(1)

def move():
        sheldon1.setTimeout(20)
        sheldon1.insertViaPoint(path)
        sheldon1.psOn()
        sheldon1.home(2)
        t_on = time.time()
        sheldon1.release()
        sheldon1.run()
        t_done = time.time()
        juke()
        sheldon1.__sendFrame__([0, 0, 400, 0, 1, 3])
        sheldon1.psOff()
        print ('Time taken: '+str(t_done-t_on)[:6] + 'seconds')

def juke():
        songs = []
        for song in os.scandir('./sheldonOS/tracks/'):
                songs.append(song.name.split('.')[0])
        sheldon1.sing(random.choice(songs))
        # sheldon1.sing('moan')

def move_back():
        sheldon1.setTimeout(20)
        sheldon1.insertViaPoint(_path)
        sheldon1.release()
        sheldon1.run()
        sheldon1.__sendFrame__([0, 0, 400, 0, 1, 3])
        sheldon1.psOff()

if __name__ == "__main__":
        print ('Welcome to Sheldon Console! {}'.format(today))
        camport = 1
        while 1:
                cmd = int(input('0: Home and Release\n1: Calibration Mode\n2: Run\n3: Run Demo\n4: Manual Operation\n5: Test Vision\n6: Retrive Cups\n7: Exit\n>>> '))
                if cmd == 0:
                        reset()
                elif cmd == 1:
                        try:
                                Vision.calibrate(camport = camport)
                        except:
                                ut.sprint('There is a ploblem with calibration unit. Returning to homepage...', 'warn')
                elif len(str(cmd)) == 2:
                        if str(cmd)[0] == '2':
                                mode = int(str(cmd)[1])
                                suc = 0
                                for _ in range(3):
                                        if suc == 1:break
                                        try:
                                                visout = Vision.run(camport = camport,flip = 0, colormode = 0, maskmode = mode, placemode = 'center', debug = False) 
                                                visout_prev = visout.copy()
                                                path = ut.gpfcc(visout, cups) 
                                                ut.sprint('Coordinate is verified', 'success')
                                                move()
                                                suc = 1
                                        except:
                                                suc = 0
                                                ut.sprint('There is a ploblem with vision unit', 'warn')
                        elif str(cmd)[0] == '5':
                                mode = int(str(cmd)[1])
                                visout = Vision.run(camport = camport,flip = 0, colormode = 0, maskmode = mode, placemode = 'center', debug = True) 
                                ut.sprint('Coordinate is verified', 'success')
                        else:
                                pass
                elif cmd == 2:
                        suc = 0
                        for _ in range(3):
                                if suc == 1:break
                                try:
                                        visout = Vision.run(camport = camport,flip = 0, colormode = 0, maskmode = 0, placemode = 'center', debug = False) 
                                        visout_prev = visout.copy()
                                        path = ut.gpfcc(visout, cups) 
                                        ut.sprint('Coordinate is verified', 'success')
                                        move()
                                        suc = 1
                                except:
                                        suc = 0
                                        ut.sprint('There is a ploblem with vision unit', 'warn')
                elif cmd == 4:
                        sheldon1.psOn()
                        sheldon1.manOp()
                elif cmd == 3:
                        reset()
                        visout = democo()
                        visout_prev = visout.copy()
                        path = ut.gpfcc(visout, cups)
                        move()
                elif cmd == 5:
                        visout = Vision.run(camport = camport,flip = 0, colormode = 0, maskmode = 0, placemode = 'center', debug = True) 
                        ut.sprint('Coordinate is verified', 'success')
                elif cmd == 6:
                        # reset()
                        sheldon1.home(2)
                        _path = ut._gpfcc(visout_prev, cups)
                        move_back()
                elif cmd == 7:
                        try:
                                ut.sprint('Quitting...', 'warn')
                                sheldon1.psOff()
                                sys.exit(1)
                        except:
                                sys.exit(1)
                else:
                        ut.sprint ('You entered a wrong command', 'warn')
