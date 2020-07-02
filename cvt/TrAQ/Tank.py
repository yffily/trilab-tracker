import sys
import os
import pickle
import numpy as np
import cv2
from cvt.utils import *


class Tank:

    def __init__(self):
        self.x_px    = 0
        self.y_px    = 0
        self.r_px    = 0
        self.points  = np.zeros((3,2))    
        self.n_point = 0
        self.found   = False
        self.frame   = None
        self.wname   = None
        self.cap     = None


    def release_capture(self):
        try:
            self.cap.release()
        except:
            pass
        self.cap = None
    

    def save(self, fname):
        keys = [ 'x_px', 'y_px', 'r_px', 'points' ]
        save_pik( fname, { k:self.__dict__[k] for k in keys } )
            

    def load(self, fname):
        try:
            self.__dict__.update(load_pik(fname))
            return True
        except:
            return False
    
    
    #########################
    # Tank locator GUI
    #########################
    
    
    def add_circle_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x,y)


    def add_point(self, x, y):
        if self.n_point > 2:
            if self.n_point == 3: 
                print("    Note: Only green points are used for calculation.")
            x_tmp, y_tmp = self.points[self.n_point%3][0], self.points[self.n_point%3][1]
            cv2.circle(self.frame, (int(x_tmp), int(y_tmp)), 3, (0,0,255), -1)
            cv2.imshow(self.wname,self.frame)
        self.points[self.n_point % 3] = [x, y]
        self.n_point += 1
        cv2.circle(self.frame, (int(x),int(y)), 3, (0,255,0), -1)
        cv2.imshow(self.wname,self.frame)
        if self.n_point > 2:
            self.calculate_circle()
            print("    Locating tank edges... ")


    def select_circle(self, event, x, y, flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if np.sqrt(pow(x-self.x_px,2)+pow(y-self.y_px,2)) > self.r_px:
                self.found = False
                cv2.circle(self.frame, (int(self.x_px),int(self.y_px)), int(self.r_px), (0,0,255), -1)
                cv2.imshow(self.wname,self.frame)
                k = wait_on_named_window(self.wname)
            else:
                cv2.circle(self.frame, (int(self.x_px),int(self.y_px)), int(self.r_px), (0,255,0), -1)
                cv2.imshow(self.wname,self.frame)
                k = wait_on_named_window(self.wname)


    def calculate_circle(self):
        midpoint = []
        m = []
        b = []
        for i in range(2):
            midpoint.append([(self.points[i+1][0]+self.points[i][0])/2,
                             (self.points[i+1][1]+self.points[i][1])/2])
            slope = ((self.points[i+1][1]-self.points[i][1])/
                     (self.points[i+1][0]-self.points[i][0]))
            m.append(-1./slope)
            b.append(midpoint[i][1]-m[i]*midpoint[i][0])
 
        self.x_px = (b[1]-b[0])/(m[0]-m[1])
        self.y_px = m[0]*self.x_px + b[0]
        self.r_px = np.sqrt( pow(self.x_px-self.points[0][0],2) +
                             pow(self.y_px-self.points[0][1],2) )
        self.found = True

    
    def interrupt(self,msg=None):
        cv2.destroyAllWindows()
        self.release_capture()
        if msg!=None:
            sys.stdout.write(f'\n       {msg}.\n')
            sys.stdout.flush()
        return False
    
    
    def locate(self,fvideo,i_frame=None):
        
        self.cap = cv2.VideoCapture(fvideo)
        
        if not self.cap.isOpened():
            sys.exit("  Video cannot be opened! Ensure proper video file specified.")
        
        n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if i_frame == None:
            i_frame = n_frames//2
        elif i_frame < 0:
            i_frame = n_frames - 1
        
        # open frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
        ret, self.frame = self.cap.read()
        if ret == True:
            # try locating the tank
            self.wname = create_named_window('Locate the tank (click on three distinct points on the boundary)')
#            cv2.setWindowProperty(self.wname,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(self.wname, self.add_circle_point)
            cv2.imshow(self.wname, self.frame)
            if wait_on_named_window(self.wname) == -2:
                return self.interrupt('Tank detection interrupted.')
            # show results and allow user to choose if it looks right
            cv2.circle(self.frame, (int(self.x_px),int(self.y_px)), int(self.r_px), (0,255,0), 2)
            cv2.circle(self.frame, (int(self.x_px),int(self.y_px)), 3, (0,255,0), -1)
            cv2.setMouseCallback(self.wname, self.select_circle)
            cv2.imshow(self.wname, self.frame)
            if wait_on_named_window(self.wname) == -2:
                return self.interrupt('Tank detection interrupted.')
            # if the user decides the tank location is good, then exit loop
        
        self.frame = None
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
      
        if self.found:
            sys.stdout.write("\n       Tank detection complete.\n")
            sys.stdout.flush()
            return True
        else:
            sys.stdout.write("\n       Tank detection failed.\n")
            sys.stdout.flush()
            return False
    
    
    def load_or_locate_and_save(self,tank_file,video_file,i_frame=None):
        if not self.load(tank_file):
            if self.locate(video_file,i_frame):
                self.save(tank_file)
    
    
