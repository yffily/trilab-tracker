import os
import sys
#import pickle
import cv2
import numpy as np

return_key,esc_key,space_key = 13,27,32 # Key codes for return, esc, space (for cv2.waitKey).

default_window_size = 800,800

def create_named_window(name='preview window'):
#    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.namedWindow(name,cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow(name,0,0)
    cv2.resizeWindow(name,default_window_size[0],default_window_size[1])
    return name

# Wait for a set duration then return false if the window needs closing, 
# true otherwise.
def wait_on_named_window(name,delay=-1):
    k = cv2.waitKey(delay)
    if cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)!=1 or k==esc_key:
        return -2
    return k

