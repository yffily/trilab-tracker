import os
import sys
import pickle
import numpy as np
import pandas as pd
import copy
import matplotlib.cm as mpl_cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cvt.TrAQ.Tank import Tank


class Trial:

    def __init__(self):
        self.result = {}
        self.issue  = {}
        self.cut_stats = {}
        self.plot_options = dict(dpi=150)
        
    
    def init(self, video_file, output_dir = None, n_ind = 0, 
             fps = 30, tank_radius = 1., t_start = 0, t_end = -1 ):
        
        self.video_file = video_file
        self.output_dir = output_dir
        
        if output_dir == None:
            print('Warning: output files will be written in the directory of the input video.')
            self.output_dir = os.path.split(video_file)[0]
        
        self.trial_file  = os.path.join(output_dir,'trial.pik')
        self.traced_file = os.path.join(output_dir,'traced.mp4')
        self.tank_file   = os.path.join(output_dir,'tank.pik')
        
        self.tank        = Tank(tank_radius)
        self.tank.load_or_locate_and_save(self.tank_file,self.video_file)
        self.n_ind       = int(n_ind)
        self.fps         = fps
        self.frame_start = int(t_start*fps)
        self.frame_end   = int(t_end*fps) if t_end>0 else -1
                
        # Store fish data as a dataframe with multiindex columns:
        # level 1=fish id, level 2=quantity (x,y,angle,...).
        # x_px and y_px are the cv2 pixel coordinates
        # (x_px=horizontal, y_px=vertical upside down).
        columns = pd.MultiIndex.from_product([ range(self.n_ind),
                                               ['x_px','y_px','ang','area'] ])
        self.df = pd.DataFrame(columns=columns, index=pd.Index([],name='frame'))
        

    def print_info(self):
        print("       trial: %s" % ( self.trial_file ) )
        print("       input: %s" % ( self.fvideo_raw ) )
        if self.issue:
            print("       Known issues: " )
            for key in self.issue:
                print("           %s: %s" % (key, self.issue[key]))
    
    
    def save(self, fname = None):
        if fname == None:
            fname = self.trial_file
        try:
            f = open(fname, 'wb')
            pickle.dump(self.__dict__, f, protocol = 3)
            sys.stdout.write("\n       Trial object saved as %s \n" % fname)
            sys.stdout.flush()
            f.close()
            return True
        except:
            return False


    def load(self, fname = None):
        if fname == None:
            fname = self.trial_file
#        try:
        f = open(fname, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        
        # This should help use trial files created on a different machine.
        self.output_dir = os.path.split(fname)[0]
        for v in 'trial_file','tank_file','traced_file':
            p = os.path.basename(self.__dict__[v])
            self.__dict__[v] = os.path.join(self.output_dir,p)
        
        sys.stdout.write("\n        Trial loaded from %s \n" % fname)
        sys.stdout.flush()
        return True
#        except:
#            sys.stdout.write("\n        Unable to load Trial from %s \n" % fname)
#            sys.stdout.flush()
#            return False


    def convert_pixels_to_cm(self):
        a = self.tank.r_cm/self.tank.r
        for i in range(self.n_ind):
            self.df[i,'x'] =  a*(self.df[i,'x_px']-self.tank.col_c)
            self.df[i,'y'] = -a*(self.df[i,'y_px']-self.tank.row_c)

    
