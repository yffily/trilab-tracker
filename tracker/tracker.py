import sys
import os
import logging
import cv2
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import skew
from tracker.tank import Tank
from tracker.utils import *
import datetime


class Tracker:
    
    def __init__(self, input_video, output_dir, n_ind, 
                 t_start = 0, t_end = -1,  
                 n_pixel_blur = 3, block_size = 15, threshold_offset = 13, 
                 min_area = 20, max_area = 400, ideal_area = None, 
                 max_aspect = 10, ideal_aspect = None, area_penalty = 1, 
                 reversal_threshold = None, significant_displacement = None, 
                 adaptiveMethod = 'gaussian', threshType = 'inv', 
                 bkgSub_options = dict( n_training_frames = 500, 
                                        t_start = 0, t_end = -1,
                                        contrast_factor = 4 ), 
                 RGB = True, live_preview = False, GPU = False, **args ):

        self.input_video    = input_video
        self.output_dir     = output_dir
        self.n_ind          = n_ind  # number of individuals to track
        self.trial_file     = os.path.join(output_dir,'trial.pik')
        self.settings_file  = os.path.join(output_dir,'settings.txt')
        
        self.colors         = color_list     
        self.GPU            = GPU

        # Video input.
        self.t_start        = t_start
        self.t_end          = t_end
        
        # Video output.
        self.RGB            = RGB
        self.live_preview   = live_preview
        self.preview_window = 'live tracking preview'
#        ext,self.codec      = 'avi','DIVX'
        ext,self.codec      = 'mp4','mp4v'
        self.output_video   = os.path.join(output_dir,'tracked.'+ext)
        
        # Background subtraction.
        self.background     = None
        self.bkgSub_options = bkgSub_options

        # Tracking parameters.
        self.n_pix_avg      = n_pixel_blur
        self.thresh         = []
        self.block_size     = block_size
        self.offset         = threshold_offset
        self.threshMax      = 100
        self.adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptiveMethod=='gaussian' \
                              else cv2.ADAPTIVE_THRESH_MEAN_C
        self.threshType     = cv2.THRESH_BINARY_INV if threshType=='inv' \
                              else cv2.THRESH_BINARY
        self.min_area       = max(min_area,1)
        self.max_area       = max_area
        self.ideal_area     = ideal_area if ideal_area!=None else (min_area+max_area)/2
        self.max_aspect     = max_aspect
        self.ideal_aspect   = ideal_aspect if ideal_aspect!=None else max_aspect/2
        self.area_penalty   = area_penalty
        
        body_size_estimate  = np.sqrt(self.max_area)
        self.reversal_threshold = reversal_threshold if reversal_threshold!=None \
                                  else body_size_estimate*0.05
        self.significant_displacement = significant_displacement if significant_displacement!=None \
                                        else body_size_estimate*0.2
    

    ############################
    # cv2.VideoCapture functions
    ############################
    
    def init_directory(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


    def init_tank(self):
        self.tank_file = os.path.join(self.output_dir,'tank.pik')
        self.tank = Tank()
        self.tank.load_or_locate_and_save(self.tank_file,self.input_video)


    def init_background(self):
        self.bkg_file       = os.path.join(self.output_dir,'background.npz')
        self.bkg_img_file   = os.path.join(self.output_dir,'background.png')
        if not self.load_background(self.bkg_file):
            self.compute_background()
            self.save_background(self.bkg_file)
        cv2.imwrite(self.bkg_img_file,self.background)


    def init_tracking_data_structure(self):
        self.contours       = []
        # The output of the tracking is stored in a numpy array.
        # Axes: 0=frame, 1=fish, 2=quantity.
        # Quantities: x_px, y_px, angle, area.
        # x_px and y_px are the cv2 pixel coordinates
        # (x_px=horizontal, y_px=vertical upside down).
        self.data = np.empty((0,self.n_ind,4), dtype=np.float)
        self.frame_list = np.array([],dtype=np.int)
        self.last_known = np.zeros((self.n_ind,4), dtype=int)


    def init_video_input(self):
        self.cap = cv2.VideoCapture(self.input_video)
        if self.cap.isOpened() == False:
            sys.exit(f'Cannot read video file {self.input_video}')
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps      = int(self.cap.get(cv2.CAP_PROP_FPS))
#        self.fourcc   = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.width    = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frame_start    = max(0, int(self.t_start*self.fps))
        self.frame_end      = int(self.t_end*self.fps) if self.t_end>0 else self.n_frames-1
        self.frame_end      = min(self.frame_end-1, self.frame_end)
        self.set_frame(self.frame_start-1)
        
        
    def init_video_output(self):
        # Video writer class to output video with contour and centroid of tracked
        # object(s) make sure the frame size matches size of array 'final'
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.out = cv2.VideoWriter( filename = self.output_video, 
                                    frameSize = (self.width,self.height), 
                                    fourcc = fourcc, fps = self.fps, isColor = self.RGB )


    def init_all(self):
        self.init_directory()
        self.init_tank()
        self.init_video_input()
        self.init_video_output()
        self.init_background()
        self.init_tracking_data_structure()


    def release(self):
        self.cap.release()
        self.out.release()
#        self.cap,self.out = None, None # This should help make the Tracker object picklable.
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        logging.info(f'{parindent}Video capture released')


    def tracked_frames(self):
        return self.frame_num - self.frame_start

    
    def set_frame(self, i):
        self.frame_num = i
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)


    def get_next_frame(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret == True:
                self.frame_num += 1
                if self.GPU:
                    self.frame = cv2.UMat(self.frame)
                return True
        return False

    
    def get_frame(self,i):
        self.set_frame(i-1)
        return self.get_next_frame()
        
    
    def load_background(self, bkg_file):
        ext = os.path.splitext(bkg_file)[1]
        if os.path.exists(bkg_file):
            if ext=='.npy':
                self.background = np.load(bkg_file)
                return True
            if ext=='.npz':
                # Assume there's only one variable in the file.
                self.background = next(iter(np.load(bkg_file).values()))
                return True
        return False


    def save_background(self, bkg_file):
        ext = os.path.splitext(bkg_file)[1]
        if ext=='.npy':
            np.save(bkg_file,self.background)
            return True
        if ext=='.npz':
            np.savez_compressed(bkg_file,background=self.background)
            return True
        return False


    def compute_background(self):
        logging.info(parindent+'Computing background') 
        t_start    = self.bkgSub_options['t_start']
        t_end      = self.bkgSub_options['t_end']
        n_training = self.bkgSub_options['n_training_frames']
        i_start    = max( 0, int(t_start*self.fps) )
        i_end      = int(t_end*self.fps) if t_end>0 else self.n_frames-1
        i_end      = min( self.n_frames-1, i_end )
        training_frames = np.linspace(i_start, i_end, n_training, dtype=int)
        self.get_next_frame()
        self.background = np.zeros(self.frame.shape)
        count = 0
        for i in training_frames:
            self.set_frame(i)
            if self.get_next_frame():
                self.background += self.frame
                count           += 1
        self.background /= count


    def subtract_background(self):
        if type(self.background) != type(None):
            self.frame  = self.frame-self.background
            self.frame *= self.bkgSub_options['contrast_factor']
            self.frame  = np.absolute(self.frame)
            self.frame  = (255 - np.minimum(255, self.frame)).astype(np.uint8)
            
    
    def init_live_preview(self):
        if (self.live_preview):
            create_named_window(self.preview_window)
    

    def write_frame(self):
        if self.GPU:
            self.frame = cv2.UMat.get(self.frame)
        return self.out.write(self.frame)

        
    def post_frame(self,delay=None):
        if delay==None or delay<1:
            delay = int(1000/self.fps)
        if self.live_preview:
            wname = self.preview_window
            cv2.imshow(wname,self.frame)
            k = wait_on_named_window(wname,delay)
            if k==-2:
                return 0
            if k==space_key:
                while True:
                    k2 = wait_on_named_window(wname,delay)
                    if k2==-2:
                        return 0
                    if k2==space_key:
                        break
        return 1


    def get_current_timestamp(self):
        s = self.frame_num/self.fps
        m,s = divmod(s,60)
        h,m = divmod(m,60)
        return f'{h:02.0f}:{m:02.0f}:{s:05.2f}'


    def get_percent_complete(self):
        return 100*(self.frame_num-self.frame_start)/(self.frame_end-self.frame_start)
        

    ############################
    # Contour functions
    ############################


    def threshold_detect(self, hist = False):
        # blur and current image for smoother contours
        blur = cv2.GaussianBlur(self.frame, (self.n_pix_avg, self.n_pix_avg), 0)
       
        # convert to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        if hist:
            plt.title("Raw image grayscale histogram")
            plt.hist(self.frame.ravel()[self.frame.ravel() > 0],256)
            plt.show()
        
            plt.title("Blurred grayscale histogram")
            plt.hist(blur.ravel()[blur.ravel() > 0],256)
            plt.show()
        
            plt.title("Grayscale histogram")
            plt.hist(gray.ravel()[gray.ravel() > 0],256)
            plt.show()
    
        # calculate adaptive threshold using cv2
        #   more info:
        #       https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        self.thresh = cv2.adaptiveThreshold( gray, 
                                             maxValue = self.threshMax, 
                                             adaptiveMethod = self.adaptiveMethod,
                                             thresholdType = self.threshType,
                                             blockSize = self.block_size, 
                                             C = self.offset )
    
    
    def detect_contours(self):
        self.threshold_detect()
        # The [-2:] makes this work with openCV versions 3 and 4.
        self.contours, hierarchy = cv2.findContours( self.thresh, 
                                                     cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE )[-2:]

        self.new = []
        if self.tracked_frames()==1:
            self.cimg = np.zeros(self.frame.shape[:2])
        for i,c in enumerate(self.contours):
            M = cv2.moments(c)
            # If area is valid, proceed with contour analysis.
            area = M['m00']
            if self.min_area<=area<=self.max_area:
                x     = M['m10'] / area
                y     = M['m01'] / area
                theta = 0.5 * np.arctan2(2*M['mu11'], M['mu20']-M['mu02'])
                mu    = np.array([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])/area
                eVal,eVec = la.eigh(mu)
#                theta = np.arctan2(eVec[1,1],eVec[1,0])
                aspect = np.sqrt(eVal[1]/eVal[0])
                # At the beginning, use skewness to guess front/back.
                if self.tracked_frames()==1:
                    cv2.drawContours(self.cimg, [c], 0, color=i, thickness=-1)
                    Y,X = np.nonzero(self.cimg==i)
                    # TODO: Try using width of the fish in the front half vs rear half instead
                    # of the skew along the long axis. Find out which one is more robust.
                    U = (X-x)*np.cos(theta)+(Y-y)*np.sin(theta)
                    if skew(U)>0:
                        theta += np.pi
                if aspect<=self.max_aspect:
                    M['valid'] = True
                    self.new.append([x,y,theta,area])
                    continue
            self.contours[i] = None
        for i in range(len(self.new),self.n_ind):
            self.new.append([np.nan]*4)
        self.new = np.array(self.new,dtype=np.float)
        self.contours = [ c for c in self.contours if type(c)!=type(None) ]
        

    #############################
    # Frame-to-frame functions
    #############################

    
    # Predict fish coordinates in current frame based on past frames.
    def predict_next(self):
        # Grab last coordinates.
        last = self.data[-1].copy()
        if len(self.data)==1:
            return last
        # If available, use before-last coordinates to refine guess.
        penul      = self.data[-2].copy()
        penul[:,3] = last[:,3] # Don't feed area noise into the prediction.
        pred       = 2*last - penul
        # If last or penul is nan, use the other.
        # TODO: If last and penul are both nan, look at older coordinates.
        I          = np.isnan(penul)
        pred[I]    = last[I]
        I          = np.isnan(last)
        pred[I]    = penul[I]
        # For the angle, use the last known value no matter how old.
        I          = np.isnan(pred[:,2])
        pred[I,2]  = self.data[self.last_known[I,2],I,2]
        return pred

    
    # Use information from past frames to identify the contour corresponding
    # to each fish in the current frame.
    def connect_frames(self):
        
        # If there is no previous frame, use the n_ind best contours.
        # If there aren't enough contours, fill with NaN.
        if len(self.data) == 0:
            if len(self.new)>self.n_ind:
                # Use the distance to the ideal area and aspect ratio to measure
                # contour quality. I don't save the moment-obtained aspect ratio 
                # so using ellipse fitting to save time/space. Also giving a little 
                # more weight to the area distance than the aspect one.
                d1 = np.absolute(self.new[:,3]/self.ideal_area-1)
                ellipses = [ cv2.fitEllipse(c) for c in self.contours ]
                aspects  = np.array([ e[1][1]/e[1][0] for e in ellipses ])
                d2 = np.absolute(aspects/self.ideal_aspect-1)
                d  = d1+d2
                # TODO: Take another look at the cluster algorithm approach used 
                # in cvtracer. What situation is it meant to address? Is that idea 
                # that some fish may be broken into multiple contours?
                self.new = self.new[np.argsort(d)][:self.n_ind]
        else:
            # If there is a valid previous frame, compute the generalized distance between
            # every object in the frame and every object in the previous frame, then solve
            # the assignment problem with scipy.
            predicted        = self.predict_next()
            # Include contour areas in the generalized distance to penalize excessive 
            # changes of area between two frames.
            coord_pred       = predicted[:,[0,1,3]]
            coord_new        = self.new[:,[0,1,3]]
            coord_pred[:,2] *= self.area_penalty
            coord_new[:,2]  *= self.area_penalty
            d                = cdist(coord_pred,coord_new)
            # Use a two-tiered penalty system for missing fish.
            # If a fish has no predicted coordinates (NaN), use its last known position
            # but increase the distance a by a factor 1e4 so recently seen fish get
            # matched first.
            # If a fish has no last known position set the distance to 1e8 so those get
            # matched last.
            I                = np.nonzero(np.isnan(coord_pred))[0]
            coord_pred[I]    = self.last_known[I][:,[0,1,3]]
            coord_pred[I,2] *= self.area_penalty
            d[I]             = 1e4*cdist(coord_pred[I],coord_new)
            d[np.isnan(d)]   = 1e8
            Io,In = linear_sum_assignment(d)
            self.new = self.new[In]
            
            # TODO: When a fish is missing use its last known position or some prediction 
            # based on it. Caveat: if the last known position is old, try to match recently
            # valid objects first.
            
            # TODO: Identify cases in which a disappeared fish likely
            # merged contours with another fish (e.g. detect likely occlusions).
            # This will involve looking for area changes and making sure it doesn't
            # involve an overly large frame-to-frame displacement.
        
            # Fix orientations.
            past    = self.data[-5:]
            # 1. Look for continuity with the predicted value. Use last known value rather than
            # predicted value to avoid reversal instability following a very quick turn (as 
            # happens e.g. when contours merge).
            dtheta  = self.new[:,2]-self.last_known[:,2]
            dtheta[np.isnan(dtheta)] = 0 # Avoid nan propagating from predicted to new.
            self.new[:,2] -= np.pi*np.rint(dtheta/np.pi)
            # 2. Reverse direction if recent motion as been against it.
            if len(past)>4:
                history = np.concatenate([past,self.new[None,:,:]])
                dxy     = history[1:,:,:2]-history[:-1,:,:2]
                dot     = np.cos(history[:-1,:,2])*dxy[:,:,0] + np.sin(history[:-1,:,2])*dxy[:,:,1]
                dot[np.isnan(dot)] = 0
                reverse = np.max(dot,axis=0)<-self.reversal_threshold
                past[:,reverse,2]   += np.pi # Go back in history to fix orientation.
                self.new[reverse,2] += np.pi        
        
        self.data = np.append(self.data, [self.new], axis=0)
        self.frame_list = np.append(self.frame_list,self.frame_num)
        I = ~np.isnan(self.new)
        self.last_known[I] = self.new[I]
    
        
    ############################
    # Masking functions
    ############################
    
        
    def mask_tank(self):
        x = int(self.tank.x_px)
        y = int(self.tank.y_px)
        R = int(self.tank.r_px) + 1 # ??
        if self.threshType==cv2.THRESH_BINARY_INV:
            tank_mask = 255*np.ones_like(self.frame)
            if self.GPU:
                tank_mask = cv2.UMat(tank_mask)
            cv2.circle(tank_mask, (x,y), R, (0,0,0), thickness=-1)
            self.frame = cv2.bitwise_or(self.frame, tank_mask)
        else:
            tank_mask = np.zeros_like(self.frame)
            if self.GPU:
                tank_mask = cv2.UMat(tank_mask)
            cv2.circle(tank_mask, (x,y), R, (255,255,255), thickness=-1)
            self.frame = cv2.bitwise_and(self.frame, tank_mask)


#    def mask_contours(self):
#        self.contour_masks = []
#        # add contour areas to mask
#        for i in range(len(self.contours)):
#            mask = np.zeros_like(self.frame, dtype=bool)
#            cv2.drawContours(mask, self.contours, i, 1, -1)
#            self.contour_masks.append(mask)


    ############################
    # Drawing functions
    ############################


    # Show the current frame. Press any key or click "close" button to exit.
    def show_current_frame(self):
        window_name = create_named_window('current frame')
        cv2.imshow(window_name,self.frame)
        wait_on_named_window(window_name)
        cv2.destroyAllWindows()
        return 1


    def draw(self, tank=True, contours=False, 
             contour_color=(0,0,0), contour_thickness=1,
             points=True, directors=True, timestamp=True):
        if tank:
            self.draw_tank(self.tank)
        if contours:
            self.draw_contours(contour_color, contour_thickness)
        if points:
            self.draw_points()
        if directors:
            self.draw_directors()
        if timestamp:
            self.draw_tstamp()

    
    def draw_tstamp(self):
        color = (0,0,0) if self.RGB else 0
        font  = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        t_str = self.get_current_timestamp()
        cv2.putText(self.frame, t_str, (5,30), font, 1, color, 2)

    
    def draw_tank(self, tank):
        color = (0,0,0) if self.RGB else 0
        cv2.circle(self.frame, (int(tank.x_px), int(tank.y_px)), int(tank.r_px),
                   color, thickness=1)

    
    # Draw every contour.
    def draw_contours(self,contour_color=(0,0,255), contour_thickness=1):
        color = contour_color if self.RGB else 0
        cv2.drawContours(self.frame, self.contours, -1, color, int(contour_thickness))


    # Draw center of each individual as a dot.
    def draw_points(self):
        XY = self.new[:,:2]
        for i,(x,y) in enumerate(XY):
            if not (np.isnan(x) or np.isnan(y)):
                x,y = int(x),int(y)
                color = self.colors[i%len(self.colors)] if self.RGB else 0
                cv2.circle(self.frame, (x,y), 3, color, -1, cv2.LINE_AA)


    # Draw orientation of each individual as an arrow.
    def draw_directors(self, arrow_size=20):
        XY = self.new[:,:2]
        Th = self.new[:,2]
        U  = arrow_size/2*np.array([np.cos(Th),np.sin(Th)]).T
        for i in range(self.n_ind):
            if not np.isnan(XY[i]).any():
                color = self.colors[i%len(self.colors)] if self.RGB else 255
                (x1,y1),(x2,y2) = (XY[i]-U[i]).astype(int),(XY[i]+U[i]).astype(int)
                cv2.arrowedLine(self.frame, (x1,y1), (x2,y2), color=color, 
                                thickness=2, tipLength=0.3)


    ############################
    # Save/Load
    ############################
    
    def init_output_dir(self, output_dir=None):
        if output_dir==None:
            output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)        
    

    def save_settings(self, fname = None):
        keys = [ 't_start', 't_end', 'n_frames', 'frame_start', 'frame_end', 'frame_num', 
                 'bkgSub_options', 'n_pix_avg', 'block_size', 
                 'offset', 'threshMax', 'adaptiveMethod', 'threshType', 
                 'min_area', 'max_area', 'ideal_area', 'reversal_threshold', 
                 'significant_displacement' ]
        if fname == None:
            fname = self.settings_file
        save_txt( fname, { k:self.__dict__[k] for k in keys if k in self.__dict__.keys() } )
            

    def save_trial(self, fname = None):
        keys = [ 'input_video', 'output_dir', 'n_ind', 'fps', 'tank', 'data', 'frame_list' ]
        if fname == None:
            fname = self.trial_file
        save_pik( fname, { k:self.__dict__[k] for k in keys } )


# Remaining variables (not in settings or trial):
# keys = [ 'trial_file', 'colors', 'GPU', 'cap', 'width', 'height', 'RGB', 'live_preview', 'preview_window', 'codec', 'output_video', 'out', 'tank_file', 'background', 'bkg_file', 'bkg_img_file', 'thresh', 'contours', 'moments', 'frame', 'new' ]
