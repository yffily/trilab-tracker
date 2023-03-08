import datetime
import enum
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import skew
from .tank import Tank
from .frame import Frame
from .utils import *
from .gui_main import start_gui


# Message to include in the pickled tracking output. 
info = \
'''n_ind: Number of individuals in the trial.
n_extra: Number of candidate individuals that were tracked in addition to the n_ind we know are there.
data: Numpy array with size (n_frames,n_ind+n_extra,5). The 5 quantities saved for each contour in each frame are the x coordinate, y coordinate, angle with x axis, area, and aspect ratio.'''

arg_info = \
'''
input_video: Path to the video to track.
output_dir: Directory where output files will be saved.
n_ind: Number of individuals in the trial.
t_start: Timestamp at which to start tracking (in seconds).
t_stop: Timestamp at which to stop tracking (in seconds).
n_extra: Number of candidate individuals to track in addition to the n_ind we know are there. When part of the background is mistaken for an individual, the actual individual is lost. Tracking extra contours helps fix that after the fact without retracking from scratch.
n_report: Number of times to print progress when tracking a video.
n_blur: Radius of Gaussian blur.
block_size: Radius or region used in adaptive thresholding.
threshold_offset: Offset threshold used in thresholding (difference of grayscale value between background and foreground).
min_area: Minimal area for a contour to qualify as an individual (in pixels).
max_area: Maximal area for a contour to qualify as an individual (in pixels).
ideal_area: Approximate area of an individual (in pixels). Used to identify the n_ind best candidate individuals in the first frame.
max_aspect: Maximal aspect ratio for a contour to qualify as an individual.
ideal_aspect: Approximate aspect ratio of an individual. Used to identify the n_ind best candidate individuals in the first frame.
area_penalty: Weight of area changed compared to distance traveled when connecting the candidate individuals found in a frame to the ones found in the previous frame.
morph_transform: List of morphological transformation to apply after thresholding. Each transformation is given as a pair (cv2 transform, radius). The radius is used to generate a circular kernel.
reversal_threshold: Minimal distance traveled by an individual against its own orientation over the last 5 frames to reverse said orientation.
bkg.n_training_frames: Number of frames to average to compute the video's background.
bkg.t_start: Timestamp at which to start computing the video's background.
bkg.t_stop: Timestamp at which to stop computing the video's background.
bkg.contrast_factor: Number by which to multiply the difference between a frame and the background.
bkg.secondary_subtraction: Whether to perform secondary subtraction, i.e., de-emphasize pixels whose difference with the background is consistently large.
bkg.secondary_factor: How much to de-emphasize pixels during secondary background subtraction. 
'''

default_args = dict( t_start = 0, t_end = -1, n_extra = 1, n_report = 100, 
                     n_blur = 3, block_size = 15, threshold_offset = 13, 
                     min_area = 20, max_area = 800, ideal_area = None, 
                     max_aspect = 15, ideal_aspect = None, area_penalty = 1, 
                     morph_transform = [], reversal_threshold = None, 
                     # significant_displacement = None, 
                     bkg = dict( n_training_frames = 100, t_start = 0, t_end = -1,
                                 contrast_factor = 5, secondary_subtraction = True,
                                 secondary_factor = 1 )
                    ) 


class Tracker:
    
    def __init__(self, input_video, output_dir, n_ind, **args):
        
        self.__dict__.update(input_video=input_video, output_dir=output_dir, n_ind=n_ind)        
        if any( ('.' in k) for k in args.keys() ):
            args = flatten_dict.unflatten(args, 'dot')
        self.__dict__.update(default_args)
        self.__dict__.update({ k:v for k,v in args.items() if k in default_args.keys() })
        self.bkg['secondary_subtraction'] = bool(self.bkg['secondary_subtraction'])
        
        self.n_track       = self.n_ind+self.n_extra # Number of contours to track = number of individuals + n_extra.
        self.n_report      = 100 # Number of times to report progress when tracking full video.
        self.trial_file    = osp.join(output_dir,'trial.pik')
        self.settings_file = osp.join(output_dir,'settings.txt')
        self.colors        = color_list     
        self.min_area      = max(self.min_area,1)
        if self.ideal_area is None:
            self.ideal_area   = (self.min_area+self.max_area)/2
        if self.ideal_aspect is None:
            self.ideal_aspect = self.max_aspect/2 
        body_size_estimate = np.sqrt(self.max_area)
        if self.reversal_threshold is None:
            self.reversal_threshold = body_size_estimate*0.05
#        if significant_displacement is None:
#            self.significant_displacement = body_size_estimate*0.2
        
        # Parameters for morphological operations on contour candidates.
        # The input parameter should be list of (cv2.MorphType,pixel_radius) pairs.
        # Here we convert each radius to a kernel.
        kernel             = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
        self.morph         = [ (t,kernel(r)) for t,r in self.morph_transform ]
        

    def init_directory(self):
        if not osp.exists(self.output_dir):
            os.mkdir(self.output_dir)


    def init_tank(self,img=None,**args):
        logging.info(parindent+'Detecting tank')
        self.tank_file = osp.join(self.output_dir,'tank.pik')
#        self.tank_img_file = osp.join(self.output_dir,'tank.png')
        self.tank = Tank()
        if not self.tank.load(self.tank_file):
            for k,v in args.items():
                self.tank.__dict__[k] = v
            b = (self.frame.bkg is None)
            if ( b and self.tank.locate_from_video(self.input_video) ) \
                or ( (not b) and self.tank.locate(self.frame.bkg.astype(np.uint8), \
                                                  self.input_video) ):
                self.tank.save(self.tank_file)
#                self.tank.save_img(self.tank_img_file)


    def init_background(self):
        self.bkg_file     = osp.join(self.output_dir,'background.npz')
#        self.bkg_img_file = osp.join(self.output_dir,'background.png')
        self.frame.bkg    = self.load_background(self.bkg_file)
        if self.frame.bkg is None:
            self.compute_background()
            self.save_background(self.bkg_file,self.frame.bkg)
#            cv2.imwrite(self.bkg_img_file,self.frame.bkg)


    def init_secondary_background(self):
        self.bkg_file2     = osp.join(self.output_dir,'background2.npz')
#        self.bkg_img_file2 = osp.join(self.output_dir,'background2.png')
        self.frame.bkg2    = self.load_background(self.bkg_file2)
        if self.frame.bkg2 is None:
            self.compute_secondary_background()
            self.save_background(self.bkg_file2, self.frame.bkg2)
#        cv2.imwrite(self.bkg_img_file2, 255-self.frame.bkg2)
        self.frame.bkg2 *= self.bkg['secondary_factor'] * self.bkg['contrast_factor']


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
        self.frame = Frame( shape=(self.height,self.width) ) 
        self.frame.contrast_factor = self.bkg['contrast_factor']
        
        
    def init_tank_mask(self):
        self.frame.mask = self.tank.create_mask((self.height,self.width))


    def init_video_link(self):
        ext = osp.splitext(self.input_video)[1]
        # Create a symbolic link to the input video and a text file containing
        # the path the the video, which serves as a backup link. The symbolic 
        # link gives direct access to the video, but doesn't work in windows. 
        # The text file ensure the viewer can locate the video either way.
        relative_input = osp.relpath(self.input_video, self.output_dir)
        input_link = osp.join(self.output_dir, 'raw.txt')
        if not osp.exists(input_link):
            with open(input_link, 'w') as fh:
                fh.write(relative_input)
        try:
            input_link = osp.join(self.output_dir,'raw'+ext)
            if not osp.exists(input_link):
                input_link = osp.join(self.output_dir, 'raw'+ext)
                os.symlink(relative_input, input_link)
        except:
            pass


    def init_all(self):
        self.init_directory()
        self.init_video_link()
        self.init_video_input()
        self.init_background()
        if self.bkg['secondary_subtraction']:
            self.init_secondary_background()
        self.init_tank()
        self.init_tank_mask()
        self.init_tracking_data_structure()


    ############################
    # cv2.VideoCapture functions
    ############################
    
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
        ext = osp.splitext(bkg_file)[1]
        if osp.exists(bkg_file):
            if ext=='.npy':
                return np.load(bkg_file)
            if ext=='.npz':
                # Assume there's only one variable in the file.
                return next(iter(np.load(bkg_file).values()))
        return None


    def save_background(self, bkg_file, bkg):
        ext = osp.splitext(bkg_file)[1]
        if ext=='.npy':
            np.save(bkg_file,bkg)
            return True
        if ext=='.npz':
            np.savez_compressed(bkg_file,background=bkg)
            return True
        return False


    def compute_background(self):
        logging.info(parindent+'Computing background')
        t_start    = self.bkg['t_start']
        t_end      = self.bkg['t_end']
        n_training = self.bkg['n_training_frames']
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
        t_start    = self.bkg['t_start']
        t_end      = self.bkg['t_end']
        n_training = self.bkg['n_training_frames']
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


    def get_current_timestamp(self):
        s = self.frame_num/self.fps
        m,s = divmod(s,60)
        h,m = divmod(m,60)
        return f'{h:02.0f}:{m:02.0f}:{s:05.2f}'


    def get_percent_complete(self):
        return 100*(self.frame_num-self.frame_start)/(self.frame_end-self.frame_start)


    def track_next_frame(self):
        if not self.get_next_frame():
            return False
        self.frame.subtract_background(secondary=False)
        self.frame.subtract_background(primary=False)
        self.frame.apply_mask()
        self.frame.blur(self.n_blur)
        self.frame.threshold(self.block_size, self.threshold_offset)
        for mtype,mval in self.morph:
            self.frame.apply_morphological_transform(mtype,mval)
        self.frame.detect_contours()
        self.frame.analyze_contours( self.n_track, self.min_area, self.max_area, 
                                     self.max_aspect, guess_front=self.tracked_frames()==1)
        self.connect_frames()
        return True


    def track_video(self):
        add_log_file(osp.join(self.output_dir,'log.txt'))
        logging.info(parindent+'Initializing')
        self.init_all()
        self.save_settings()
        try:
            logging.info(parindent+'Starting to track')
            self.set_frame(self.frame_start)
            i_report = max(1,int((self.frame_end-self.frame_start)/self.n_report))
            for i_frame in range(self.frame_start, self.frame_end):
                if (i_frame-self.frame_start)%i_report==0:
                    self.save_trial()
                    percent = (i_frame-self.frame_start)/(self.frame_end-self.frame_start)
                    t = str(datetime.datetime.now()).split('.')
                    logging.info(parindent + f'Tracking: {self.get_current_timestamp()}, ' +
                                 f'{self.get_percent_complete():4.1f}% complete, {t[0]}.{t[1][:2]}' )
                if not self.track_next_frame():
                    raise Exception(f'Could not read frame {i_frame}.')
            self.release()
            logging.info(parindent+'Saving')
            self.save_trial()
            logging.info(parindent+'Done')
            reset_logging()
            return 0
        except:
            self.release()
            logging.info('Failed')
            logging.exception('')
            return 1
    

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
    # Save/Load
    ############################
    
    def init_output_dir(self, output_dir=None):
        if output_dir==None:
            output_dir = self.output_dir
        if not osp.exists(output_dir):
            os.mkdir(output_dir)        
    

    def save_settings(self, fname = None):
        keys = [ # 'input_video', 'output_dir', 
                 'n_ind', 'n_extra', 't_start', 't_end', 
                 'n_frames', 'frame_start', 'frame_end', 'frame_num', 
                 'n_blur', 'block_size', 'threshold_offset', 
                 'min_area', 'max_area', 'ideal_area', 
                 'max_aspect', 'ideal_aspect', 'area_penalty', 
                 'reversal_threshold', #'significant_displacement', 
                 'bkg' ]
        if fname == None:
            fname = self.settings_file
        D = { k:v for k,v in self.__dict__.items() if k in keys }
        save_txt( fname, flatten_dict.flatten(D, 'dot') )
    

    def save_trial(self, fname = None):
        D = { k:self.__dict__[k] for k in [ 'input_video', 'output_dir', 'n_ind', 'n_extra', 'fps', 'data', 'frame_list' ] }
        D['tank'] = self.tank.to_dict()
        D['info'] = info
        if fname == None:
            fname = self.trial_file
        save_pik( fname, D )


# Remaining variables (not in settings or trial):
# keys = [ 'trial_file', 'colors', 'cap', 'width', 'height', 'RGB', 'preview_window', 'codec', 'output_video', 'out', 'tank_file', 'background', 'bkg_file', 'bkg_img_file', 'thresh', 'contours', 'moments', 'frame', 'new' ]
