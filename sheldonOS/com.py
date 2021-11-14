import serial
import time
from colorama import init
from termcolor import cprint 
import sys
init(strip=not sys.stdout.isatty())

class Uart:
    def __init__(self, device, port, baudrate):
        try:
            self.device = device
            self.ser = serial.Serial(port = port, baudrate = baudrate)
            a = ("{} is connected via {}".format(self.device, self.ser.portstr))
            cprint (' ','white','on_green',end = ' ')
            cprint (a,'green')
            self.status = 1
        except:
            cprint (' ','white','on_red',end = ' ')
            a = ("{} is not connected".format(self.device))
            cprint (a,'red')
            self.status = 0

    def __str__(self):
        return self.device

    def flush(self):
        self.ser.reset_input_buffer()

    def flush_out(self):
        self.ser.reset_output_buffer()

    def enable(self):
        self.ser.rts = 0

    def disable(self):
        self.ser.rts = 1

    def reset(self):
        self.disable()
        time.sleep(0.15)
        self.enable()
        time.sleep(0.15)

    def close(self):
        self.ser.close()

    def get_line(self):
        data = self.ser.readline()
        processed_data = data.decode()[:len(data)-1]
        return processed_data

    def get_raw(self):
        data = self.ser.read(7)
        if data[0] == 254:
            if data[1] == 255:
                return list(data)
        

    def write_line(self, data):    
        self.ser.write(data)
        return 1





if __name__ == "__main__":
    pass


