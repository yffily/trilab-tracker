import sys
import os
import logging
import numpy as np
import cv2
from tracker.utils import *

point_radius = 8

class Tank:

    def __init__(self):
        self.x_px    = 0
        self.y_px    = 0
        self.r_px    = 0
        self.points  = []
        self.found   = False
        self.raw_frame = None
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
            logging.info(parindent+f'Tank loaded from {fname}')
            return True
        except:
            return False


    def load_or_locate_and_save(self,tank_file,video_file,i_frame=None):
        if not self.load(tank_file):
            if self.locate(video_file,i_frame):
                self.save(tank_file)


    #########################
    # Tank locator GUI
    #########################


    def locate(self,fvideo,i_frame=None):
        
        self.cap = cv2.VideoCapture(fvideo)
        
        if not self.cap.isOpened():
            self.interrupt(f'Could not open {fvideo}.')
        
        # Pick frame.
        n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if i_frame == None:
            i_frame = n_frames//2
        elif i_frame < 0:
            i_frame = n_frames - i_frame
        
        # Open frame.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
        ret, self.raw_frame = self.cap.read()
        self.frame = self.raw_frame.copy()
        if not ret:
            self.interrupt(f'Could not open frame {i_frame}.')
        
        # Wait for user to click on the edge three times.
        self.wname = create_named_window(f'[{fvideo}] Click three points on the boundary. Drag as needed. Press space to accept, esc to cancel.')
        self.point_dragged = None
        cv2.setMouseCallback(self.wname, self.add_point)
        cv2.imshow(self.wname,self.frame)
        while True:
            k = wait_on_named_window(self.wname)
            if k == -2:
                return self.interrupt('Tank detection interrupted.')
            elif k == space_key and len(self.points)==3: # accept the result
                return self.interrupt('Tank detection complete.', True)

    
    def redraw_points(self):
        self.frame = self.raw_frame.copy()
        for p in self.points:
            cv2.circle(self.frame, (int(p[0]), int(p[1])), point_radius, (0,255,0), -1)
        if len(self.points)==3:
            self.calculate_circle()
        cv2.imshow(self.wname,self.frame)
    
    
    def add_point(self, event, x, y, flags, param):
    
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if we grabbed an existing point.
            for i,p in enumerate(self.points):
                if np.sqrt((x-p[0])**2+(y-p[1])**2)<point_radius:
                    self.point_dragged = i
                    return
            # Create a new point.
            if len(self.points)<3:
                self.points.append((x,y))
                self.redraw_points()
                return

        if event == cv2.EVENT_MOUSEMOVE and self.point_dragged!=None:
            self.points[self.point_dragged] = (x,y)
            self.redraw_points()
            return
        
        if event == cv2.EVENT_LBUTTONUP and self.point_dragged!=None:
            self.points[self.point_dragged] = (x,y)
            self.redraw_points()
            self.point_dragged = None
            return


    def calculate_circle(self,draw=True):
        points    = np.array(self.points)
        midpoints = (points[1:]+points[:-1])/2
        vectors   = points[1:]-points[:-1]
        m         = -vectors[:,0]/vectors[:,1] # -1/slopes
        b         = midpoints[:,1]-m*midpoints[:,0] # intercepts
        self.x_px = (b[1]-b[0])/(m[0]-m[1])
        self.y_px = m[0]*self.x_px + b[0]
        self.r_px = np.sqrt( (self.x_px-points[0,0])**2 + 
                             (self.y_px-points[0,1])**2 )
        if draw:
            cv2.circle(self.frame, (int(self.x_px),int(self.y_px)), int(self.r_px), (0,255,0), 1)


    def interrupt(self,msg=None,retval=False):
        cv2.destroyAllWindows()
        self.release_capture()
        if msg!=None:
            logging.info(parindent+f'{msg}')
        return retval


