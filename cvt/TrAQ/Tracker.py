import sys
import os
import pickle
import cv2
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from cvt.TrAQ.Tank import Tank
from cvt.utils import *


class Tracker:
    
    def __init__(self, input_video, output_dir, n_ind, 
                 t_start = 0, t_end = -1,  
                 n_pixel_blur = 3, block_size = 15, 
                 threshold_offset = 13, min_area = 20, max_area = 400, 
                 max_aspect = 10, area_penalty = 1, 
                 max_frame2frame_distance = None, significant_displacement = None, 
                 adaptiveMethod = 'gaussian', threshType = 'inv', 
                 bkgSub_options = dict( n_training_frames = 500, 
                                        t_start = 0, t_end = -1,
                                        contrast_factor = 4 ), 
                 RGB = True, live_preview = False, GPU = False ):

        self.input_video    = input_video
        self.output_dir     = output_dir
        self.n_ind          = n_ind  # number of individuals to track
        self.trial_file     = os.path.join(output_dir,'trial.pik')
        self.settings_file  = os.path.join(output_dir,'settings.txt')
        
        self.colors         = color_list     
        self.GPU            = GPU

        # Video input.
        self.init_video_input()
        self.t_start        = t_start
        self.t_end          = t_end
        self.frame_start    = max(0, int(t_start*self.fps))
        self.frame_end      = int(t_end*self.fps) if t_end>0 else self.n_frames-1
        self.frame_end      = min(self.frame_end-1, self.frame_end)
        self.set_frame(self.frame_start-1)
        
        # Video output.
        self.RGB            = RGB
        self.live_preview   = live_preview
        self.preview_window = 'live tracking preview'
        ext                 = 'mp4' # 'avi'
        self.codec          = {'mp4':'mp4v','avi':'DIVX'}[ext]
        self.output_video   = os.path.join(output_dir,'tracked.'+ext)
        self.init_video_output()
        
        # Tank.
        self.tank_file      = os.path.join(output_dir,'tank.pik')
        self.tank           = Tank()
        self.tank.load_or_locate_and_save(self.tank_file,self.input_video)

        # Background subtraction.
        self.background     = None
        self.bkgSub_options = bkgSub_options
        self.bkg_file       = os.path.join(output_dir,'background.npz')
        self.bkg_img_file   = os.path.join(output_dir,'background.png')
        if not self.load_background(self.bkg_file):
            self.compute_background()
            self.save_background(self.bkg_file)
        cv2.imwrite(self.bkg_img_file,self.background)

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
        self.ideal_area     = (min_area+max_area)/2
        self.max_aspect     = max_aspect
        self.area_penalty   = area_penalty
        
        self.max_frame2frame_distance = max_frame2frame_distance
        if self.max_frame2frame_distance == None:
            # Maximum distance a fish can reasonably travel over a single frame.
            # sqrt(max_area) works as a rough estimate for the body length.
            self.max_frame2frame_distance = np.sqrt(self.max_area)*2
        
        self.significant_displacement = significant_displacement
        if significant_displacement == None:
            self.significant_displacement = np.sqrt(self.max_area)*0.2
    
        # Tracking data structures.
        self.contours       = []
        self.moments        = []
        # The tracked fish data is stored as a dataframe with multiindex columns:
        # level 1=fish id, level 2=quantity (x,y,angle,...).
        # x_px and y_px are the cv2 pixel coordinates
        # (x_px=horizontal, y_px=vertical upside down).
        columns = pd.MultiIndex.from_product([ range(self.n_ind),
                                               ['x_px','y_px','ang','area'] ])
        self.df = pd.DataFrame(columns=columns, index=pd.Index([],name='frame'))


    ############################
    # cv2.VideoCapture functions
    ############################


    def init_video_input(self):
        self.cap = cv2.VideoCapture(self.input_video)
        if self.cap.isOpened() == False:
            sys.exit(f'Cannot read video file {self.input_video}\n')
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps      = int(self.cap.get(cv2.CAP_PROP_FPS))
#        self.fourcc   = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.width    = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
    def init_video_output(self):
        # Video writer class to output video with contour and centroid of tracked
        # object(s) make sure the frame size matches size of array 'final'
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.out = cv2.VideoWriter( filename = self.output_video, 
                                    frameSize = (self.width,self.height), 
                                    fourcc = fourcc, fps = self.fps, isColor = self.RGB )
        

    def release(self):
        # release the capture
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        sys.stdout.write("\n")
        sys.stdout.write("       Video capture released.\n")
        sys.stdout.flush()


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
        
        sys.stdout.write("       Computing background...") 
        sys.stdout.flush()
        
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
        if ( self.live_preview ):
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
 

    def print_current_frame(self):
        t_csc = int(self.frame_num/self.fps * 100)
        t_sec = int(t_csc/100) 
        t_min = t_sec/60
        t_hor = t_min/60
        sys.stdout.write("       Current tracking time: %02i:%02i:%02i:%02i \r" 
                         % (t_hor, t_min % 60, t_sec % 60, t_csc % 100) )
        sys.stdout.flush()
        

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

#        # Dismiss contours with out-of-bounds area.
#        self.contours = [ c for c in self.contours if 
#                          self.min_area<=cv2.contourArea(c)<=self.max_area ]
        
        # Compute contour moments.
        self.moments  = [ cv2.moments(c) for c in self.contours ]
        
        for M in self.moments:
            mu = np.array([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])/M['m00']
            eVal,eVec = la.eigh(mu)
            M['aspect'] = eVal[0]/eVal[1]
            
        # Dismiss contours with out-of-bounds area or aspect ratio.
        self.contours, self.moments = zip(* 
                          [ (c,M) for c,M in zip(self.contours,self.moments) if 
                          self.min_area<=M['m00']<=self.max_area 
                          and M['aspect']<=self.max_aspect ]
                          )
        
        # Compute tentative new coordinates.
        # If there aren't enough contours, fill with NaN.
        n = max(len(self.contours),self.n_ind)
        self.new = np.empty((n,4),dtype=float)
        self.new.fill(np.nan)
        for i,M in enumerate(self.moments):
            x     = M['m10'] / M['m00']
            y     = M['m01'] / M['m00']
            area  = M['m00']
            theta = 0.5 * np.arctan2(2*M['mu11'], M['mu20']-M['mu02'])
            
            # This is a bit more costly but provides the semi-axes as well.
#            mu = np.array([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])/M['m00']
#            eVal,eVec = la.eigh(mu)
#            theta = np.arctan2(eVec[1,1],eVec[1,0])
            
            self.new[i] = [x, y, theta, area]


    #############################
    # Frame-to-frame functions
    #############################


    def predict_next(self):
        # Grab last coordinates.
        last = self.df.values[-1].reshape((self.n_ind,4))
        if len(self.df)==1:
            return last
        # If available, use before-last coordinates to refine guess.
        penul = self.df.values[-2].reshape((self.n_ind,4))
        penul[:,3] = last[:,3] # Don't feed area noise into the prediction.
        pred  = 2*last - penul
        # If last or penul is nan, use the other.
        # TODO: If last and penul are both nan, look at older coordinates.
        I     = np.isnan(penul)
        pred[I] = last[I]
        I     = np.isnan(last)
        pred[I] = penul[I]
        return pred


    # this is the main algorithm the tracer follows when trying 
    # to associate individuals identified in this frame with 
    # those identified in the previous frame/s
    def connect_frames(self):
        
        # If there is no previous frame, use the n_ind best contours.
        # If there aren't enough contours, fill with NaN.
        if len(self.df) == 0:
            if len(self.new)>self.n_ind:
                # TODO: Take another look at the cluster algorithm approach used 
                # in cvtracer. What situation is it meant to address? Is that idea 
                # that some fish may be broken into multiple contours?
                d = np.absolute(self.new[:,3]-self.ideal_area) # distance ideal contour area
                self.new = self.new[np.argsort(d)][:self.n_ind]
        else:
            # If there is a valid previous frame, first set missing positions aside,
            # then solve assignment problem on remaining positions.
            old     = self.predict_next()
            # coord contains the positions and area weighted with self.area_change_penalty.
            coord_old  = old[:,[0,1,3]]
            coord_new  = self.new[:,[0,1,3]]
            coord_old *= self.area_penalty
            coord_new *= self.area_penalty
            d          = cdist(coord_old,coord_new)
            # Putting a high cost on connections involving a missing position (NaN)
            # effectively makes the algorithm match all valid contours first.
            # This likely causes problems when multiple fish are lost.
            # Using the last known position would probably be better.
            d[np.isnan(d)] = 1e8
            Io,In = linear_sum_assignment(d)
            self.new = self.new[In]
            
            # TODO: When a fish is missing look for its last known position. 
            # Even better: use last few known positions to predict current position.
            
            # TODO: Identify cases in which a disappeared fish likely
            # merged contours with another fish (e.g. detect likely occlusions).
            # This will involve looking for area changes and making sure it doesn't
            # involve a jump longer than max_frame2frame_distance.
        
        
            # Fix spurious orientation reversals:
            # 1. Look for continuity with previous frame.
            dtheta  = self.new[:,2]-old[:,2]
            I       = ~np.isnan(dtheta)
            self.new[I,2] -= np.pi*np.rint(dtheta[I]/np.pi)
            # 2. Align direction with motion unless it's too slow.
            dxy     = self.new[:,:2]-old[:,:2]
            dot     = dxy[:,0]*np.cos(self.new[:,2])+dxy[:,1]*np.sin(self.new[:,2])
            I       = (dot<-self.significant_displacement)
            self.new[I,2] += np.pi
            # TODO: Measure the orientation directly from the contour, probably by 
            # looking at the sign of the third moment along the unsigned orientation.
        
        
        self.df.loc[self.frame_num] = self.new.flatten()
    
        
    ############################
    # Masking functions
    ############################
    
        
    def mask_tank(self):
        x = int(self.tank.x_px)
        y = int(self.tank.y_px)
        R = int(self.tank.r_px) + 1
        if self.GPU:
            tank_mask = cv2.UMat(np.zeros_like(self.frame))
        else:
            tank_mask = np.zeros_like(self.frame)
        
        cv2.circle(tank_mask, (x,y), R, (255,255,255), thickness=-1)
        self.frame = cv2.bitwise_and(self.frame, tank_mask)


    def mask_contours(self):
        self.contour_masks = []
        # add contour areas to mask
        for i in range(len(self.contours)):
            mask = np.zeros_like(self.frame, dtype=bool)
            cv2.drawContours(mask, self.contours, i, 1, -1)
            self.contour_masks.append(mask)


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
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        t_str = "%02i:%02i:%02i.%02i" % ( 
                int(self.frame_num / self.fps / 3600),
                int(self.frame_num / self.fps / 60 % 60),
                int(self.frame_num / self.fps % 60),
                int(self.frame_num % self.fps * 100 / self.fps) )
        cv2.putText(self.frame, t_str, (5,30), font, 1, color, 2)

    
    def draw_tank(self, tank):
        color = (0,0,0) if self.RGB else 0
        cv2.circle(self.frame, (int(tank.x_px), int(tank.y_px)), int(tank.r_px),
                   color, thickness=4)

    
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


#    def save(self, fname = None, trial=True, tracking=True, misc=False):
#        if fname == None:
#            fname = self.trial_file
##        try:
#        self.release()
#        self.cap = None
#        self.out = None
#        with open(fname,'wb') as f:
#            pickle.dump(self.__dict__, f, protocol = 3)
#        sys.stdout.write(f'       Tracker object saved as {fname}\n')
#        sys.stdout.flush()
#        return True
##        except:
##            return False

    def save_settings(self, fname = None):
        keys = [ 't_start', 't_end', 'n_frames', 'frame_start', 'frame_end', 'frame_num', 
                 'bkgSub_options', 'n_pix_avg', 'block_size', 
                 'offset', 'threshMax', 'adaptiveMethod', 'threshType', 
                 'min_area', 'max_area', 'ideal_area', 'max_frame2frame_distance', 
                 'significant_displacement' ]
        txt = '\n'.join(f'{k} = {self.__dict__[k]}' for k in keys)
        if fname == None:
            fname = self.settings_file
        with open(fname,'w') as f:
            f.write(txt)
            
        

    def save_trial(self, fname = None):
        keys = [ 'input_video', 'output_dir', 'n_ind', 'fps', 'tank', 'df' ]
        if fname == None:
            fname = self.trial_file
        with open(fname,'wb') as f:
            pickle.dump({k:self.__dict__[k] for k in keys}, f, protocol = 3)


# Remaining variables (not in settings or trial):
# keys = [ 'trial_file', 'colors', 'GPU', 'cap', 'width', 'height', 'RGB', 'live_preview', 'preview_window', 'codec', 'output_video', 'out', 'tank_file', 'background', 'bkg_file', 'bkg_img_file', 'thresh', 'contours', 'moments', 'frame', 'new' ]


#    def load(self, fname = None):
#        if fname == None:
#            fname = self.trial_file
##        try:
#        f = open(fname, 'rb')
#        tmp_dict = pickle.load(f)
#        f.close()
#        self.__dict__.update(tmp_dict)
#        
#        # This should help use trial files created on a different machine.
#        self.output_dir = os.path.split(fname)[0]
#        for v in 'trial_file','tank_file','traced_file':
#            p = os.path.basename(self.__dict__[v])
#            self.__dict__[v] = os.path.join(self.output_dir,p)
#        
#        sys.stdout.write("\n        Trial loaded from %s \n" % fname)
#        sys.stdout.flush()
#        return True
##        except:
##            sys.stdout.write("\n        Unable to load Trial from %s \n" % fname)
##            sys.stdout.flush()
##            return False

