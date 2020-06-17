import os
import sys
#import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import screeninfo


#========================================================
# openCV window interaction.

return_key,esc_key,space_key,backspace_key = 13,27,32,8 # Key codes for cv2.waitKey.

#default_window_size = 800,800
screen = screeninfo.get_monitors()[0]


def create_named_window(name='preview window'):
#    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow(name,screen.x,screen.y)
#    cv2.resizeWindow(name,default_window_size[0],default_window_size[1])
    cv2.resizeWindow(name,screen.width,screen.height)
    return name

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
# Plotting.

#color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

#color_list = plt.cm.tab20.colors
#color_list = color_list[:14]+color_list[16:]
#colors = colors[::2]+colors[1::2]

#color_list = [tuple(int(255*x) for x in color) for color in color_list]

#color_list = [ c[::-1] for c in color_list ] # from RGB from openCV's BGR

color_list = [  (   0,   0, 255),
                ( 255,   0,   0),
                (   0, 255,   0),
                (   0, 127, 255),
                ( 255,   0, 255),
                (   0,   0, 170),
                ( 255, 255,   0),
                (   0, 155,   0),
                (  50, 200, 250),
                ( 155,   0, 155),
                ( 155,   0,   0),
                ( 255, 255, 255),
                (   0,   0,   0) ]


#========================================================


