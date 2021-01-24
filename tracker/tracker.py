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
from .tank import Tank
from .utils import *
from .frame import FrameAnalyzer
import datetime
import enum
    
# Message to include in the pickled tracking output. 
info = \
'''- n_ind is the number of individuals in the trial.
- n_extra is the number of extra dark/light regions the tracker keeps track of on top of the n_ind one would normally track. Sometimes the tracker mistakes part of the background for an individual, and one of the actual individuals to be tracked gets discarded. Keeping track of additional contours allows to go back and recover the mistakenly discarded data after the fact.
- data is a numpy array with size (n_frames,n_ind+n_extra,5). The 5 quantities saved for each contour in each frame are: x coordinate, y coordinate, angle with x axis, area, and aspect ratio.'''


class Tracker:
    
    def __init__(self, input_video, output_dir, n_ind, 
                 t_start = 0, t_end = -1,  
                 n_blur = 3, block_size = 15, threshold_offset = 13, 
                 min_area = 20, max_area = 400, ideal_area = None, 
                 max_aspect = 10, ideal_aspect = None, area_penalty = 1, 
                 n_extra=1, morph_transform = [], 
                 reversal_threshold = None, significant_displacement = None, 
                 bkgSub_options = dict( n_training_frames = 500, 
                                        t_start = 0, t_end = -1,
                                        contrast_factor = 4,
                                        secondary_subtraction = True,
                                        secondary_factor = 1 ), 
                video_output_options = dict( tank=True, points=False, directors=True, 
                                        extra_points=True, timestamp=True, 
                                        contours=True, contour_color=(100,255,0), 
                                        contour_thickness=1 ), 
                 save_video = True, **args ):

        self.input_video    = input_video
        self.output_dir     = output_dir
        self.n_ind          = n_ind     # number of individuals to track
        self.n_extra        = n_extra   # number of extra contours to keep track of
        self.n_track        = self.n_ind+self.n_extra # total contours to keep track of
        self.trial_file     = os.path.join(output_dir,'trial.pik')
        self.settings_file  = os.path.join(output_dir,'settings.txt')
        
        self.colors         = color_list     

        # Video input.
        self.t_start        = t_start
        self.t_end          = t_end
        
        # Video output.
        self.save_video     = save_video
#        ext,self.codec      = 'avi','DIVX'
        ext,self.codec      = 'mp4','mp4v' # 'FMP4' # 
        self.output_video   = os.path.join(output_dir,'tracked.'+ext)
        self.video_output_options = video_output_options
        
        # Background subtraction.
        self.bkgSub_options = bkgSub_options

        # Tracking parameters.
        self.n_blur         = n_blur
        self.thresh         = []
        self.block_size     = block_size
        self.offset         = threshold_offset
        self.min_area       = max(min_area,1)
        self.max_area       = max_area
        self.ideal_area     = ideal_area if ideal_area!=None else (min_area+max_area)/2
        self.max_aspect     = max_aspect
        self.ideal_aspect   = ideal_aspect if ideal_aspect!=None else max_aspect/2
        self.area_penalty   = area_penalty
        
        # Parameters for morphological operations on contour candidates.
        # The input parameter should be list of (cv2.MorphType,pixel_radius) pairs.
        # Here we convert each radius to a kernel.
        kernel              = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
        self.morph          = [ (t,kernel(r)) for t,r in morph_transform ]
        
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


    def init_tank(self,img=None,**args):
        self.tank_file = os.path.join(self.output_dir,'tank.pik')
        self.tank_img_file = os.path.join(self.output_dir,'tank.png')
        self.tank = Tank()
        if not self.tank.load(self.tank_file):
            for k,v in args.items():
                self.tank.__dict__[k] = v
            b = (self.frame.bkg is None)
            if ( b and self.tank.locate_from_video(self.input_video) ) \
                or ( (not b) and self.tank.locate(self.frame.bkg.astype(np.uint8), \
                                                  self.input_video) ):
                self.tank.save(self.tank_file)
                self.tank.save_img(self.tank_img_file)


    def init_background(self):
        self.bkg_file     = os.path.join(self.output_dir,'background.npz')
        self.bkg_img_file = os.path.join(self.output_dir,'background.png')
        self.frame.bkg    = self.load_background(self.bkg_file)
        if self.frame.bkg is None:
            self.compute_background()
            self.save_background(self.bkg_file,self.frame.bkg)
            cv2.imwrite(self.bkg_img_file,self.frame.bkg)


    def init_secondary_background(self):
        self.bkg_file2     = os.path.join(self.output_dir,'background2.npz')
        self.bkg_img_file2 = os.path.join(self.output_dir,'background2.png')
        self.frame.bkg2    = self.load_background(self.bkg_file2)
        if self.frame.bkg2 is None:
            self.compute_secondary_background()
            self.save_background(self.bkg_file2, self.frame.bkg2)
        cv2.imwrite(self.bkg_img_file2, 255-self.frame.bkg2)
        self.frame.bkg2 *= self.bkgSub_options['secondary_factor'] * self.bkgSub_options['contrast_factor']


    def init_tracking_data_structure(self):
        # The output of the tracking is stored in a numpy array.
        # Axes: 0=frame, 1=fish, 2=quantity.
        # Quantities: x_px, y_px, angle, area.
        # x_px and y_px are the cv2 pixel coordinates
        # (x_px=horizontal, y_px=vertical upside down).
        self.data = np.empty((0,self.n_track,5), dtype=np.float)
        self.frame_list = np.array([],dtype=np.int)
        self.last_known = np.zeros((self.n_track,), dtype=int) # frame index of last known position


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
        self.frame_end      = int(self.t_end*self.fps)
        if self.t_end<=0 or self.frame_end>self.n_frames:
            self.frame_end = self.n_frames
        self.set_frame(self.frame_start-1)
        
        self.bgr   = np.empty( shape=(self.height,self.width), dtype=np.uint8 )
        self.frame = FrameAnalyzer( shape=(self.height,self.width) ) 
        self.frame.contrast_factor = self.bkgSub_options['contrast_factor']
        
        
    def init_video_output(self):
        if self.save_video:
            # Video writer class to output video with contour and centroid of tracked
            # object(s) make sure the frame size matches size of array 'final'
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.out = cv2.VideoWriter( filename = self.output_video, 
                                        frameSize = (self.width,self.height), 
                                        fourcc = fourcc, fps = self.fps, isColor = True )


    def init_tank_mask(self):
        self.frame.mask = self.tank.create_mask((self.height,self.width))


    def init_all(self):
        self.init_directory()
        self.init_video_input()
        if self.save_video:
            self.init_video_output()
        self.init_background()
        if self.bkgSub_options['secondary_subtraction']:
            self.init_secondary_background()
        self.init_tank()
        self.init_tank_mask()
        self.init_tracking_data_structure()


    def release(self):
        if hasattr(self,'cap'):
            self.cap.release()
            del self.cap
        if hasattr(self,'out'):
            self.out.release()
            del self.out
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
            ret,self.bgr = self.cap.read()
            if ret == True:
                self.frame.from_bgr(self.bgr)
                self.frame_num += 1
                return True
        return False

    
    def get_frame(self,i):
        self.set_frame(i-1)
        return self.get_next_frame()
        
    
    def load_background(self, bkg_file):
        ext = os.path.splitext(bkg_file)[1]
        if os.path.exists(bkg_file):
            if ext=='.npy':
                return np.load(bkg_file)
            if ext=='.npz':
                # Assume there's only one variable in the file.
                return next(iter(np.load(bkg_file).values()))
        return None


    def save_background(self, bkg_file, bkg):
        ext = os.path.splitext(bkg_file)[1]
        if ext=='.npy':
            np.save(bkg_file,bkg)
            return True
        if ext=='.npz':
            np.savez_compressed(bkg_file,background=bkg)
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
        count      = 0
        self.frame.bkg = np.zeros( (self.height,self.width), dtype=np.float32 )
        training_frames = np.linspace(i_start, i_end, n_training, dtype=int)
        for i in training_frames:
            self.set_frame(i)
            if self.get_next_frame():
                np.add(self.frame.bkg, self.frame.i8, out=self.frame.bkg)
                count += 1
        self.frame.bkg /= count


    def compute_secondary_background(self):
        logging.info(parindent+'Computing secondary background')
        t_start    = self.bkgSub_options['t_start']
        t_end      = self.bkgSub_options['t_end']
        n_training = self.bkgSub_options['n_training_frames']
        i_start    = max( 0, int(t_start*self.fps) )
        i_end      = int(t_end*self.fps) if t_end>0 else self.n_frames-1
        i_end      = min( self.n_frames-1, i_end )
        training_frames = np.linspace(i_start, i_end, n_training, dtype=int)
        self.frame.bkg2 = np.zeros( (self.height,self.width), dtype=np.float32 )
        contrast_factor = self.frame.contrast_factor
        self.frame.contrast_factor = 1
        count      = 0
        for i in training_frames:
            self.set_frame(i)
            if self.get_next_frame():
                self.frame.subtract_background(primary=True, secondary=False)
                self.frame.bkg2 += self.frame.f32
                count           += 1
        self.frame.bkg2 /= count
        self.frame.contrast_factor = contrast_factor


    def write_frame(self):
        if self.save_video:
            return self.out.write(self.bgr)


    def get_current_timestamp(self):
        s = self.frame_num/self.fps
        m,s = divmod(s,60)
        h,m = divmod(m,60)
        return f'{h:02.0f}:{m:02.0f}:{s:05.2f}'


    def get_percent_complete(self):
        return 100*(self.frame_num-self.frame_start)/(self.frame_end-self.frame_start)


    def track_next_frame(self,save_frames=False):
        if not self.get_next_frame():
            return False
        b = save_frames if isinstance(save_frames,bool) else save_frames(self.frame_num)
        frames_dir  = os.path.join(self.output_dir,'frames')
        frames_path = lambda fn: os.path.join(frames_dir,f'{self.frame_num}-'+fn)
        if b:
            if not os.path.exists(frames_dir):
                os.mkdir(frames_dir)
            cv2.imwrite(frames_path('1_raw.png'),self.frame.i8)
        self.frame.subtract_background(secondary=False)
        if b:
            cv2.imwrite(frames_path('2a_bkg-subtracted.png'),self.frame.i8)
        self.frame.subtract_background(primary=False)
        if b:
            cv2.imwrite(frames_path('2b_bkg2-subtracted.png'),self.frame.i8)
        self.frame.apply_mask()
        self.frame.blur(self.n_blur)
        self.frame.threshold(self.block_size, self.offset)
        if b:
            cv2.imwrite(frames_path('3_thresholded.png'),self.frame.i8)
        for mtype,mval in self.morph:
            self.frame.apply_morphological_transform(mtype,mval)
        if b and len(self.morph)>0:
            cv2.imwrite(frames_path('3b_morph-transformed.png'),self.frame.i8)
        self.frame.detect_contours()
        self.frame.analyze_contours( self.n_track, self.min_area, self.max_area, 
                                     self.max_aspect, guess_front=self.tracked_frames()==1)
        self.connect_frames()
        if b:
            opt = self.video_output_options.copy()
            opt.update(points=False, extra_points=False, directors=False, contours=True)
            self.draw(**opt)
            cv2.imwrite(frames_path('4_contours.png'),self.bgr)
        if b or self.save_video:
            self.draw(**self.video_output_options)
        if b:
            cv2.imwrite(frames_path('5_directors.png'),self.bgr)
        self.write_frame()
        return True


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
        penul       = self.data[-2].copy()
        penul[:,3:] = last[:,3:] # Don't feed area or aspect ratio noise into the prediction.
        pred        = 2*last - penul
        # If last or penul is nan, use the other.
        # TODO: If last and penul are both nan, look at older coordinates.
        I           = np.isnan(penul)
        pred[I]     = last[I]
        I           = np.isnan(last)
        pred[I]     = penul[I]
        # For the angle, use the last known value no matter how old.
        I           = np.isnan(pred[:,2])
        pred[I,2]   = self.data[self.last_known[I],I,2]
        return pred


    # Rank potential individuals from their current (x,y,area,aspect).
    # Typically applied to all or part of self.new.
    def rank_matches(self,matches):
        # Use the distances to the ideal area and to the ideal aspect ratio 
        # to quantify contour quality. Give a little more weight to the area.
        d1 = np.absolute(matches[:,3]/self.ideal_area-1)
        d2 = np.absolute(matches[:,4]/self.ideal_aspect-1)
        d  = d1+0.5*d2
        # TODO: Take another look at the cluster algorithm approach used 
        # in cvtracer. What situation is it meant to address? Is that idea 
        # that some fish may be broken into multiple contours?
        return np.argsort(d)

    
    # Use information from past frames to identify the contour corresponding
    # to each fish in the current frame.
    def connect_frames(self):
        self.new = self.frame.coord
        # Rank contours from most likely to least likely to be a match.
#        if len(self.data) == 0:
        I = self.rank_matches(self.new)
        self.new = self.new[I]
        self.frame.contours = [ self.frame.contours[i] for i in \
                                I if i<len(self.frame.contours) ]
        # If there is no previous frame, use the n_track best contours.
        # If there aren't enough contours, fill with NaN.
        if len(self.data) == 0:
            self.new = self.new[:self.n_track]
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
            # but add a distance penalty of 1e4 so recently seen fish get matched first.
            # If a fish has no last known position set the distance to 1e8 so those get
            # matched last.
            I                = np.nonzero(np.any(np.isnan(coord_pred),axis=1))[0]
            coord_pred[I]    = self.data[self.last_known[I],I][:,[0,1,3]]
            coord_pred[I,2] *= self.area_penalty
            d[I]             = 1e4 + cdist(coord_pred[I],coord_new)
            d[np.isnan(d)]   = 1e8
            # First look for matches for the previous frame's fish, then look for matches
            # for the previous frame's extra contours.
            # This is tricky to get right. Lost fish can get replaced with spurious 
            # contours which then get prioritized when the actual fish reappears.
            Io1,In1  = linear_sum_assignment(d[:self.n_ind,:])
            Ie       = np.array([ i for i in range(self.n_track) if i not in In1 ])
            Io2,In2  = linear_sum_assignment(d[self.n_ind:,Ie])
            In       = np.concatenate([In1,Ie[In2]])
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
            dtheta  = self.new[:,2]-self.data[self.last_known,range(self.n_track),2]
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
        I = ~np.any(np.isnan(self.new),axis=1)
        self.last_known[I] = len(self.data)-1
    
        
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
             points=True, directors=True, extra_points=False, 
             timestamp=True):
        if tank:
            self.draw_tank(self.tank)
        if contours:
            self.draw_contours(contour_color, contour_thickness)
        if points:
            self.draw_points()
        if directors:
            self.draw_directors()
        if extra_points:
            self.draw_extra_points()
        if timestamp:
            self.draw_tstamp()

    
    def draw_tstamp(self):
        color = (0,0,0)
        font  = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        t_str = self.get_current_timestamp()
        cv2.putText(self.bgr, t_str, (5,30), font, 1, color, 2)

    
    def draw_tank(self, tank):
        self.tank.draw_outline(self.bgr, color=(0,0,0), thickness=1)

    
    # Draw every contour.
    def draw_contours(self,contour_color=(0,0,255), contour_thickness=1):
        color = contour_color
        cv2.drawContours(self.bgr, self.frame.contours, -1, color, int(contour_thickness))


    # Draw center of each individual as a dot.
    def draw_points(self):
        XY = self.new[:self.n_ind,:2]
        for i,(x,y) in enumerate(XY):
            if not (np.isnan(x) or np.isnan(y)):
                x,y = int(x),int(y)
                color = self.colors[i%len(self.colors)]
                cv2.circle(self.bgr, (x,y), 3, color, -1, cv2.LINE_AA)


    # Draw orientation of each individual as an arrow.
    def draw_directors(self, arrow_size=20):
        XY = self.new[:,:2]
        Th = self.new[:,2]
        U  = arrow_size/2*np.array([np.cos(Th),np.sin(Th)]).T
        for i in range(self.n_ind):
            if not np.isnan(XY[i]).any():
                color = self.colors[i%len(self.colors)]
                (x1,y1),(x2,y2) = (XY[i]-U[i]).astype(int),(XY[i]+U[i]).astype(int)
                cv2.arrowedLine(self.bgr, (x1,y1), (x2,y2), color=color, 
                                thickness=2, tipLength=0.3)


    # Draw the extra (n_track-n_ind) contours we keep track of just in case.
    def draw_extra_points(self,size=5):
        XY = self.new[self.n_ind:,:2]
        for i,(x,y) in enumerate(XY):
            if not (np.isnan(x) or np.isnan(y)):
                x,y = int(x),int(y)
                color = self.colors[(i+self.n_ind)%len(self.colors)]
                cv2.line(self.bgr, (x-size,y-size), (x+size,y+size) , color, 2)
                cv2.line(self.bgr, (x-size,y+size), (x+size,y-size) , color, 2)
        
        
    ############################
    # Save/Load
    ############################
    
    def init_output_dir(self, output_dir=None):
        if output_dir==None:
            output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)        
    

    def save_settings(self, fname = None):
        keys = [ # 'input_video', 'output_dir', 
                 'n_ind', 'n_extra', 
                 't_start', 't_end', 
                 'n_frames', 'frame_start', 'frame_end', 'frame_num', 
                 'n_blur', 'block_size', 'offset', 
                 'min_area', 'max_area', 'ideal_area', 
                 'max_aspect', 'ideal_aspect', 'area_penalty', 
                 'reversal_threshold', 'significant_displacement', 
                 'bkgSub_options' ]
        if fname == None:
            fname = self.settings_file
        save_txt( fname, { k:self.__dict__[k] for k in keys if k in self.__dict__.keys() } )
    

    def save_trial(self, fname = None):
        D = { k:self.__dict__[k] for k in [ 'input_video', 'output_dir', 'n_ind', 'n_extra', 'fps', 'data', 'frame_list' ] }
        D['tank'] = self.tank.to_dict()
        D['info'] = info
        if fname == None:
            fname = self.trial_file
        save_pik( fname, D )


# Remaining variables (not in settings or trial):
# keys = [ 'trial_file', 'colors', 'cap', 'width', 'height', 'RGB', 'preview_window', 'codec', 'output_video', 'out', 'tank_file', 'background', 'bkg_file', 'bkg_img_file', 'thresh', 'contours', 'moments', 'frame', 'new' ]
