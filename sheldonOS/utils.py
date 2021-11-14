
import time
import threading
import pprint
from math import atan2, pi, sqrt
from colorama import init
from termcolor import cprint 
import sys
init(strip=not sys.stdout.isatty())

class util:
    @staticmethod
    def int2bit(data):
        return [data&0b000011111111, data>>8]
        
    @staticmethod
    def get_packet(x, y, state):
        return bytes([254, 255]+util.int2bit(x)+util.int2bit(y)+[state])

    @staticmethod
    def imagellc(coors):
        count = 0
        for i in coors:
            print(coors[i][0])
            coors[i] = ([round(coors[i][0]*10),round((coors[i][1]*10),2)])
            count += 1
        return coors
    
    @staticmethod
    def eucDist(x, y):
        return sqrt((x*x)+(y*y))
    
    @staticmethod
    def showPath(path, square = None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        vp = [0]
        for v in path:
            vp.append(v)
        fig = plt.figure()
        ax = Axes3D(fig)
        xs, ys, zs = [0], [0], [0]
        ax.set(xlim=(0, 3000), ylim=(0, 3000), zlim = (0, 3000))
        prev = [0, 0, 0]
        ax.scatter(0, 0, 0, color = 'black', marker = '$O$')
        for point in vp[1:]:
            if len(point) >= 3:
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
                if len(point) == 3:
                    ax.scatter(point[0], point[1], point[2], color = 'black', marker = 'o', s = 30)
                else:
                    ax.scatter(point[0], point[1], point[2], color = 'gold' if point[3] is 'yellow' else point[3], marker = 'o', s = 30)
                prev = point
            elif point == 'g':
                ax.scatter(prev[0], prev[1], prev[2], color = 'white' , marker = '+', s = 20)
            elif point == 'r':
                ax.scatter(prev[0], prev[1], prev[2], color = 'white', marker = 'x', s = 20)
        ax.view_init(-150, -90)
        if square:ax.plot(xs = square[0], ys = square[1], zs = square[2], color = 'black', linestyle = '--', linewidth=1)
        ax.plot(xs = xs, ys = ys, zs = zs, color = 'lightcoral', linestyle = '--', linewidth=0.5)
        util.sprint ('Press <spacebar> to continue', 'normal')
        def quit_figure(event):
            if event.key == ' ':
                plt.close(event.canvas.figure)
        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        plt.show()
        plt.close(fig)
    
    @staticmethod
    def plane(coor):
        ret = []
        c = 0
        for ax in coor:
            if c != 2: ret.append(ax)
            else: ret.append(0)
            c+=1
        return ret 
    
    @staticmethod
    def sprint(text, msgType = 'normal'):
        if msgType == 'normal':
            cl = 'yellow'
        elif msgType == 'warn':
            cl = 'red'
        elif msgType == 'success':
            cl = 'green'
        else:
            cl = msgType
        cprint(' ', 'white', 'on_{}'.format(cl), end = ' ')
        cprint(text, cl)
    
    @staticmethod
    def gpfcc(dict_coors, list_cups, offset = [250, 600, 2750], pull_dist_from_home = 300, y_obs = 950, z_obs = 2500, ang_tol = 50, ang_getaway = 400):
        type_ = 0
        offset = offset
        for color in dict_coors:
            dict_coors[color] = [dict_coors[color][0]+offset[0], dict_coors[color][1]+offset[1], offset[2], color]
        pprint.pprint (dict_coors)
        print ('\n')
        path = []
        for num, color in enumerate(['red', 'yellow', 'green', 'blue', 'black']):
            ang = atan2(dict_coors[color][1]-list_cups[num][1], dict_coors[color][0]-list_cups[num][0])*(360/(2*pi))
            if dict_coors[color][1] < 940:
                
                sub_path = [util.plane(list_cups[num]), 
                        list_cups[num], 
                        'g', 
                        [list_cups[num][0], list_cups[num][1], pull_dist_from_home , list_cups[num][3]],
                        [dict_coors[color][0], y_obs, pull_dist_from_home , dict_coors[color][3]], 
                        [dict_coors[color][0], y_obs, z_obs , dict_coors[color][3]],
                        [dict_coors[color][0], dict_coors[color][1], z_obs , dict_coors[color][3]], 
                        dict_coors[color], 
                        'r', 
                        [dict_coors[color][0], dict_coors[color][1], z_obs , dict_coors[color][3]], 
                        [dict_coors[color][0], y_obs, z_obs , dict_coors[color][3]], 
                        [dict_coors[color][0], y_obs, pull_dist_from_home , dict_coors[color][3]]]
            else: 
                
                sub_path = [util.plane(list_cups[num]), 
                        list_cups[num], 
                        'g', 
                        [list_cups[num][0], list_cups[num][1], pull_dist_from_home , list_cups[num][3]],
                        [dict_coors[color][0], dict_coors[color][1], pull_dist_from_home , dict_coors[color][3]], 
                        dict_coors[color], 
                        'r', 
                        util.plane(dict_coors[color])]
            if 0 <= ang <= ang_tol or 180 >= ang >= 180-ang_tol: 
                # sub_path.insert(3, [list_cups[num][0], list_cups[num][1], pull_dist_from_home , list_cups[num][3]]) 
                sub_path.insert(4, [list_cups[num][0], ang_getaway, pull_dist_from_home , list_cups[num][3]])
                
            path += sub_path
        return path

    @staticmethod
    def _gpfcc(dict_coors, list_cups, offset = [250, 600, 2800], pull_dist_from_home = 300, y_obs = 950, z_obs = 2500, ang_tol = 45, ang_getaway = 400):
        type_ = 0
        offset = offset
        for color in dict_coors:
            dict_coors[color] = [dict_coors[color][0]+offset[0], dict_coors[color][1]+offset[1], offset[2], color]
        pprint.pprint (dict_coors)
        print ('\n')
        path = []
        for num, color in enumerate(['red', 'yellow', 'green', 'blue', 'black']):
            ang = atan2(dict_coors[color][1]-list_cups[num][1], dict_coors[color][0]-list_cups[num][0])*(360/(2*pi))
            if dict_coors[color][1] < 940:
                
                sub_path =[[dict_coors[color][0], y_obs, 0 , dict_coors[color][3]], 
                        [dict_coors[color][0], y_obs, z_obs , dict_coors[color][3]],
                        [dict_coors[color][0], dict_coors[color][1], z_obs , dict_coors[color][3]], 
                        dict_coors[color], 
                        'g', 
                        [dict_coors[color][0], dict_coors[color][1], z_obs , dict_coors[color][3]], 
                        [dict_coors[color][0], y_obs, z_obs , dict_coors[color][3]], 
                        [dict_coors[color][0], y_obs, 0 , dict_coors[color][3]],
                        [list_cups[num][0], list_cups[num][1], 0 , list_cups[num][3]], 
                        list_cups[num], 
                        'r', 
                        util.plane(list_cups[num])]
            else: 
                sub_path = [util.plane(dict_coors[color]), 
                        dict_coors[color], 
                        'g', 
                        util.plane(dict_coors[color]),
                        [list_cups[num][0], list_cups[num][1], 0 , list_cups[num][3]], 
                        list_cups[num], 
                        'r', 
                        util.plane(list_cups[num])]
            path += sub_path
        return path