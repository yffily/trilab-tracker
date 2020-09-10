import os
import sys
import logging
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import screeninfo


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


def add_log_file(filename,level=logging.INFO):
    handler = logging.FileHandler(filename,mode='w')
    handler.setLevel(level)
    logging.root.addHandler(handler)
    logging.root.setLevel(level)

def add_log_stream(stream=sys.stdout,level=logging.INFO):
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
#    handler.terminator = '\r'
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
        
def reset_logging():
    logging.root.handlers.clear()
    logging.root.addHandler(logging.NullHandler())
#    for h in logging.root.handlers:
#        logging.root.removeHandler(h)

#========================================================
# openCV window interaction.

return_key,esc_key,space_key,backspace_key = 13,27,32,8 # Key codes for cv2.waitKey.

#default_window_size = 800,800
screen = screeninfo.get_monitors()[0]


def create_named_window(name='preview window'):
#    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
#    cv2.resizeWindow(name,default_window_size[0],default_window_size[1])
    cv2.resizeWindow(name,screen.width,screen.height)
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
        if cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)!=1 or k==esc_key:
            return -2
        if k!=-1:
            return k
    return -1

#========================================================
# Save and load.

def save_txt(filename, data_dict):
    txt = '\n'.join( f'{key} = {value}' for key,value in data_dict.items() )
    with open(filename,'w') as f:
        f.write(txt)

def load_txt(filename):
    with open(filename,'r') as f:
        return dict( line.strip().split(' = ') for line in f )

def save_pik(filename, data_dict):
    with open(filename,'wb') as f:
        pickle.dump(data_dict, f, protocol = 3)

def load_pik(filename):
    data_dict = {}
    with open(filename,'rb') as f:
        return pickle.load(f)

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


#========================================================

