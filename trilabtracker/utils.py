import sys
import os
import os.path as osp
import logging
import traceback
import pickle
import re
import cv2
import numpy as np
import pandas as pd
from fnmatch import fnmatch
import screeninfo
import flatten_dict
import platform
import openpyxl
from . import fixer

#========================================================
# Logging.

parindent = '' # ' '*5

#DEFAULT_LOGGING_CONFIG = {
#    'version': 1,
#    'disable_existing_loggers': True,
#    'loggers': { 
#         '': { 'level': 'INFO', 'handlers': ['stdout'] } 
#         },
#    'handlers': {
#        'stdout': { 'level': 'INFO', 
#                    'class': 'logging.StreamHandler', 
#                    'stream':'ext://sys.stdout' }
#        }
#}


#def add_log_file(filename,level=logging.INFO):
#    config = DEFAULT_LOGGING.copy()
#    config['loggers']['']['handlers'] = ['stdout','file']
#    config['handlers']['file'] = {
#            'level': 'INFO',
#            'formatter': 'standard',
#            'class': 'logging.FileHandler',
#            'filename': filename,
#            'mode': 'w'
#            }
#    logging.dictConfig(config)


def add_log_file(filename, level=logging.INFO):
    handler = logging.FileHandler(filename,mode='w')
    handler.setLevel(level)
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

def add_log_stream(stream=sys.stdout, level=logging.INFO, overwrite=False):
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    if overwrite:
        handler.terminator = ''
        handler.setFormatter(logging.Formatter('\r'+' '*200+'\r     %(message)s'))        
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
        
def reset_logging():
    logging.root.handlers.clear()
    logging.root.addHandler(logging.NullHandler())
#    for h in logging.root.handlers:
#        logging.root.removeHandler(h)

def set_log_level(lvl):
    for h in logging.root.handlers:
        h.setLevel(lvl)

def overprint(msg):
    print('\r'+' '*200+f'\r{msg}', end='')

#========================================================
# openCV window interaction.

# Key codes for cv2.waitKey.
return_key,esc_key,space_key,backspace_key = 13,27,32,8
plus_key,minus_key,zero_key = 61,45,48
lbrack_key,rbrack_key = 91,93


#default_window_size = 800,800
screen = screeninfo.get_monitors()[0]


def create_named_window(name='preview window'):
#    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
#    cv2.resizeWindow(name,default_window_size[0],default_window_size[1])
#    cv2.resizeWindow(name,screen.width,screen.height)
    cv2.resizeWindow(name,int(0.9*screen.width),int(0.9*screen.height))
    cv2.moveWindow(name,screen.x,screen.y)
    return name

#def default_resize_named_window(name):
#    cv2.resizeWindow(name,screen.width,screen.height)
#    cv2.moveWindow(name,screen.x,screen.y)
#    return name

# Wait for a set duration or until a key is pressed.
# Return the keycode or -2 if the window needs to be closed.
def wait_on_named_window(name,delay=-1):
    delay = int(delay)
    if delay < 1:
        delay = 1000000 # wait "forever" ~20 minutes
    for t in range(delay):
        k = cv2.waitKey(1)
#        if cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)!=1:
#            return -2
        if k==esc_key:
            return -2
        if k!=-1:
            return k
    return -1

#========================================================
# Save and load.

# Save a dictionary as a text file.
def save_txt(filename, data_dict):
    txt = '\n'.join( f'{key} = {value}' for key,value in data_dict.items() )
    with open(filename,'w') as f:
        f.write(txt)

# Load a dictionary from a text file.
def load_txt(filename):
    with open(filename,'r') as f:
        return dict( line.strip().split(' = ') for line in f )

# Save a dictionary as a pickle file.
def save_pik(filename, data_dict):
    with open(filename,'wb') as f:
        pickle.dump(data_dict, f, protocol = 3)

# Load a dictionary from a pickle file.
def load_pik(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

# Load a settings file into a dictionary.
# Can be a txt file (e.g. output from tracker) or an excel file.
def load_settings(settings_file):
    ext = osp.splitext(settings_file)[1]
    if ext=='.txt':
        settings = load_txt(settings_file)
        for k,v in settings.items():
            settings[k] = eval(v)
    elif ext=='.xlsx':
        df = pd.read_excel(settings_file, index_col='parameter name', dtype=str)
        settings = df['parameter value'].apply(eval).to_dict()
    else:
        settings = {}
    return settings

# Load an excel settings file with a "trial filter" column specifying
# which trials each setting applies to in the form of a glob pattern.
# The result is a dictionary of settings dictionaries. 
# The keys of the outer dictionary are the glob patterns.
def load_filtered_settings(settings_file):
    df = pd.read_excel(settings_file, dtype=str)
    df['parameter value'] = df['parameter value'].apply(eval)
    fsettings = {}
    for pattern,df_ in df.groupby('trial filter'):
        fsettings[pattern] = dict(df_[['parameter name','parameter value']].values)
    return fsettings

# Create a trial's settings dictionary from a filtered settings dictionary.
def apply_filtered_settings(filtered_settings, trial_name):
    settings = {}
    for pattern,settings_ in filtered_settings.items():
        if pattern=='' or fnmatch(trial_name.lower(),pattern.lower()):
            settings.update(settings_)
    return settings

# Locate a video's frame timestamps.
def locate_timestamps(path):
    # Assume the path is either the trial's directory, or a file in it.
    trial_dir = path if osp.isdir(path) else osp.dirname(path)
    fn = open(osp.join(trial_dir, 'raw.txt')).read() # Raw video file.
    fn = osp.splitext(fn)[0]+'.txt' # Timestamp file.
    # Make fn relative to current dir rather than trial_dir.
    fn = osp.join(osp.relpath(trial_dir, os.getcwd()), fn)
    fn = osp.normpath(fn)
    return fn if os.path.exists(fn) else None
    
# Load a video's frame timestamps.
# Returns a numpy array where row index represents frame index starting at 0.
# Beware, frame indices in tracker.frame_list start from 1 (openCV convention).
def load_timestamps_(ts_file):
    try:
        df   = pd.read_csv(ts_file, index_col=0)
        time = np.empty(df.index[-1]+1)
        time.fill(np.nan)
        time[df.index] = df.iloc[:,0]
        I = np.nonzero(np.isfinite(time))[0]
        J = np.nonzero(time[I][1:]-time[I][:-1]<0)[0]
        for j in J:
            time[I[j]+1:] += 128
        time[I] -= time[I[0]]
        return time
    except:
        return None

# Load a trial file created by trilab-tracker.
def load_trial(trial_file, load_fixes=True, load_timestamps=True):
    trial_dir  = osp.dirname(trial_file)
    trial      = { 'trial_dir':trial_dir }
    with open(trial_file,'rb') as f:
        trial.update(pickle.load(f))
    if len(trial['tank']['points'])==1:
        # If arbitrary tank contour, fit a circle to it.
        ellipse = cv2.fitEllipse(trial['tank']['contour'])
        trial['tank']['xc'],trial['tank']['yc'] = np.array(ellipse[0])
#        trial['center'] = np.array(ellipse[0])
        trial['tank']['R'] = np.mean(ellipse[1])/2
    fixes_file = osp.join(trial_dir, 'gui_fixes.pik')
    if load_fixes and osp.exists(fixes_file):
        fixes = load_pik(osp.join(trial_dir,'gui_fixes.pik'))['fixes']
        for fix in fixes:
            fixer.fix(trial['data'], *fix)
    trial['data'] = trial['data'][:,:trial['n_ind'],:]
    try:
        assert(load_timestamps)
        ts_file = locate_timestamps(trial_file)
        time = load_timestamps_(ts_file)
        time = time[trial['frame_list']-1] # -1 because frame_list indices start at 1
        trial['timestamps_file'] = ts_file
        trial['time'] = time
    except:
        logging.info(f'No timestamps found. Using fps instead.')
        trial['timestamps_file'] = None
        trial['time'] = (trial['frame_list']-1)/trial['fps']
    return trial

# Load a trial file created by ethovision.
def load_trial_ethovision(trial_file):
    # Load fish data from ethovision-generated spreadsheet.
    data  = pd.read_excel( trial_file, sheet_name=None, 
                           na_values='-', index_col=0, 
                           skiprows=list(range(35))+[36], 
                           usecols=['Recording time','X center','Y center','Area'] )
    n_ind = len(data.keys())
    # Merge the sheets corresponding to each fish into a single dataframe.
    df    = pd.concat([ df.rename(columns=lambda x:(i,x)) for i,df in \
                        enumerate(data.values()) ], axis=1)
    # Convert to numpy array.
    time  = df.index.values
    pos   = df.values.reshape((-1,n_ind,3))
    vel   = pos[1:,:,:2]-pos[:-1,:,:2]
    ang   = np.arctan2(vel[:,:,1],vel[:,:,0])
    for i in range(n_ind):
        I        = ~np.isnan(ang[:,i])
        ang[I,i] = np.unwrap(ang[I,i])
    
    data  = np.dstack([pos[:-1,:,:2],ang,pos[:-1,:,2]])
    fps   = 1/np.min(time[1:]-time[:-1])
    info  = '- n_ind is the number of individuals.\n- time contains the timestamp of each frame.\n- data is an array of size (n_frames,n_ind,4). The 4 quantities saved for each fish in each frame are: x coordinate, y coordinate, angle with x axis (from velocity), area.'
    
    trial = { k:v for k,v in locals().items() if k in 
              ['trial_file', 'n_ind', 'time', 'data', 'fps', 'info'] }
    
    return trial

# Analyze tracking output to determine tracking status (including % complete).
def tracking_status(trial_dir):
    name  = osp.basename(trial_dir)
    if not osp.isdir(trial_dir):
        return dict(name=name, status='not found')
    in_trial_dir   = lambda x: osp.join(trial_dir,x)
    has_tank       = osp.exists(in_trial_dir('tank.pik'))
    has_background = osp.exists(in_trial_dir('background.npz'))
    has_trial      = osp.exists(in_trial_dir('trial.pik'))
    try:
        with open(in_trial_dir('settings.txt')) as f:
            settings = [ [x.strip() for x in line.split('=')] for line in f.readlines() ]
    except:
        settings = None
    try:
        with open(in_trial_dir('log.txt')) as f:
            log = [ line.strip() for line in f.readlines() ]
        if 'Failed' in log:
            status = 'failed'
        elif log[-1] == 'Done':
            status = 'complete'
        else:
            progress = [ line.strip() for line in log if 'Tracking' in line ]
            if len(progress)==0:
                status = 'tracking (0%)'
            else:
                percent = re.findall('[\d.]*%',progress[-1])[0]
                status  = f'tracking ({percent})'
    except:
        status = 'no log'
    return { k:v for k,v in locals().items() if not k in ['f','in_trial_dir','progress'] }


#========================================================
# Plotting.

#color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

#color_list = plt.cm.tab20.colors
#color_list = color_list[:14]+color_list[16:]
#colors = colors[::2]+colors[1::2]

#color_list = [tuple(int(255*x) for x in color) for color in color_list]

#color_list = [ c[::-1] for c in color_list ] # from RGB from openCV's BGR

color_list = [  (   0,   0, 255),  # red
                ( 255,   0,   0),  # blue
                (   0, 255,   0),  # green
                (   0, 127, 255),  # orange
                ( 255,   0, 255),  # purple
                (   0,   0, 170),  # dark red
                ( 255, 255,   0),  # cyan
                (   0, 155,   0),  # dark green
                (  50, 200, 250),  # yellow
                ( 155,   0, 155),  # dark purple
                ( 155,   0,   0),  # dark blue
                ( 255, 255, 255),  # white
                (   0,   0,   0) ] # black


# Five colors sets (blue, orange, teal, pink, green) with 4 shades of increasing darkness in each.
color_shades = [ [ '#AADFFC', '#689BB7', '#315C72', '#09232F' ],  
                 [ '#FC7940', '#BF5428', '#823413', '#491702' ],
                 [ '#3AFAF2', '#4FB1A8', '#406E67', '#22332E' ],
                 [ '#F890A2', '#C75778', '#8C264E', '#4C0225' ],
                 [ '#7bed98', '#5fb86e', '#206634', '#023b13' ] ]

# Five color maps in the same tones as the color shades above.
color_maps = [ 'Blues', 'Oranges', 'BuGn', 'RdPu', 'Greens' ]

#========================================================

