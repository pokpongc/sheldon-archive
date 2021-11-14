from sheldonOS import Sheldon, util
import time
from democoordinate import democo
from Vision.Vision import Vision
t_on = time.time()

star =  [[0, 700, 0],
        [0, 700, 2000],
        [1500, 2800, 2000], 
        [2800, 700, 2000], 
        [20, 1800, 2000], 
        [2800, 1800, 2000], 
        [20, 700, 2000],
        [20, 700, 0]]


# visout = Vision.run()


cups = [[100, 95, 425, 'red'], 
        [620, 95, 425, 'yellow'], 
        [1160, 95, 425, 'green'], 
        [1710, 95, 425, 'blue'], 
        [2270, 95, 425, 'black']]
square = [[250, 2750, 2750, 250, 250],
        [500, 500, 3000, 3000, 500],
        [3000, 3000, 3000, 3000, 3000]]


# path = util.gpfcc(visout, cups)         
def main():
        
        sheldon1 = Sheldon()
        sheldon1.setTimeout(20)
        sheldon1.insertViaPoint(path)
        sheldon1.psOn()
        time.sleep(1)
        # sheldon1.manOp()
        sheldon1.home(2)
        sheldon1.home(1)
        sheldon1.run()
        t_done = time.time()
        sheldon1.sing('FF7')
        sheldon1.home(2)
        sheldon1.home(1)
        sheldon1.psOff()
        print ('Time taken: '+str(t_done-t_on)[:6] + 'seconds')

if __name__ == "__main__":
        # util.showPath(path,square)
        # Vision.run()
        # Vision.run(camport = 0,flip = 0, colormode = 0, maskmode = 0, placemode = 'center', debug = True) 
        visout = Vision.run(camport = 0,flip = 0, colormode = 0, maskmode = 0, placemode = 'center', debug = True) 
        
        # Vision.calibrate(camport=0)
        # # main()
        
