import sys
import numpy as np
import cv2
import pickle
import os
from cvt.utils import *


class Tank:

    def __init__(self, r_cm = 111./2):
        self.points  = np.zeros((3,2))    
        self.n_point = 0
        self.row_c   = 0
        self.col_c   = 0
        self.r       = 0
        self.r_cm    = r_cm
        self.found   = False
        self.frame   = None
        self.wname   = None


    def print_info(self):
        print("")
        print("        Tank information (pixels)")
        print("            row: %4.2e " % self.row_c )
        print("            col: %4.2e " % self.col_c )
        print("              R: %4.2e " % self.r     )
        print("")


    def save(self, fname):
        f = open(fname, 'wb')
        pickle.dump(self.__dict__, f, protocol = 3)
        sys.stdout.write("\n        Tank object saved as %s \n" % fname)
        sys.stdout.flush()
        f.close()


    def load(self, fname):
        try:
            f = open(fname, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict) 
            sys.stdout.write("\n        Tank object loaded from %s \n" % fname)
            sys.stdout.flush()
            return True
        except:
            sys.stdout.write("\n        Tank not found %s \n" % fname)
            sys.stdout.flush()
            return False


    def load_txt(self, fname_txt = None):
        try:
            f = open(os.path.abspath(fname_txt),'r')
            f.readline()
            vals = f.readline().split(' ')
            vals = np.array(vals, dtype = float)
            self.row_c = vals[0]
            self.col_c = vals[1]
            self.r     = vals[2]
        except:
            print("    Cannot locate %s!" % os.path.abspath(fname_txt))
            exit()



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
            cv2.circle(self.frame, (int(x_tmp), int(y_tmp)), 4, (0, 0, 255), -1)
            cv2.imshow(self.wname,self.frame)
            cv2.waitKey(cv2.EVENT_LBUTTONUP)
        self.points[self.n_point % 3] = [x, y]
        self.n_point += 1
        cv2.circle(self.frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.imshow(self.wname,self.frame)
        cv2.waitKey(cv2.EVENT_LBUTTONUP)
        if self.n_point > 2:
            self.calculate_circle()
            print("    Locating tank edges... ")


    def select_circle(self, event, x, y, flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            px,py = x,y
            if np.sqrt(pow(px-self.row_c,2)+pow(py-self.col_c,2)) > self.r:
                self.found = False
                cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), int(self.r), (0, 0, 255), -1)
                cv2.imshow(self.wname,self.frame)
                cv2.waitKey(0)
            else:
                cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), int(self.r), (0, 255, 0), -1)
                cv2.imshow(self.wname,self.frame)
                cv2.waitKey(0)    


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
 
        self.row_c = (b[1]-b[0])/(m[0]-m[1])
        self.col_c = m[0]*self.row_c + b[0]
        self.r = np.sqrt(pow(self.row_c-self.points[0][0],2) + pow(self.col_c-self.points[0][1],2))
        self.found = True


    def locate(self,fvideo,i_frame=None):
        
        cap = cv2.VideoCapture(fvideo)
        
        if cap.isOpened() == False:
            sys.exit("  Video cannot be opened! Ensure proper video file specified.")
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if i_frame == None:
            i_frame = n_frames//2
        elif i_frame < 0:
            i_frame = n_frames - 1
        
        # open frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
        ret, self.frame = cap.read()
        if ret == True:
            # try locating the tank
            self.wname = create_named_window('Locate the tank (click on three distinct points on the boundary)')
            cv2.setMouseCallback(self.wname, self.add_circle_point)
            cv2.imshow(self.wname, self.frame)
            cv2.waitKey(0)
            # show results and allow user to choose if it looks right
            cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), int(self.r), (0, 255, 0), 4)
            cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), 5, (0, 255, 0), -1)
            cv2.setMouseCallback(self.wname, self.select_circle)
            cv2.imshow(self.wname, self.frame)
            cv2.waitKey(0)
            # if the user decides the tank location is good, then exit loop
        
        self.frame = None
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
      
        if self.found:
            sys.stdout.write("\n       Tank detection complete.\n")
        else:
            sys.stdout.write("\n       Tank detection failed.\n")
        sys.stdout.flush()
    
    def load_or_locate_and_save(self,tank_file,video_file,i_frame=None):
        if not self.load(tank_file):
            self.locate(video_file,i_frame)
            self.save(tank_file)
    
    
