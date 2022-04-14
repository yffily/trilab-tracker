import sys
import cv2
import numpy as np
import numpy.linalg as la
from scipy.stats import skew


class Frame:
    
    def __init__(self, shape):
        # shape is (height,width,channels)
        self.i8   = np.empty(shape=shape, dtype=np.uint8)
        self.f32  = np.empty(shape=shape, dtype=np.float32)
        self.bkg  = None
        self.bkg2 = None
        self.mask = None
        self.contrast_factor = 1
    
    def from_bgr(self,bgr):
        cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY, dst=self.i8)
    
    def apply_mask(self):
        self.i8 = cv2.bitwise_and(self.i8, self.mask)

    def subtract_background(self, primary=True, secondary=True):
        b1 = primary and (not self.bkg is None)
        b2 = secondary and (not self.bkg2 is None)
        if b1:
            np.subtract(self.i8, self.bkg, out=self.f32)
            np.multiply(self.f32, self.contrast_factor, out=self.f32)
            np.absolute(self.f32, out=self.f32)
            np.minimum(self.f32, 255, out=self.f32)
        elif b2:
            self.f32[...] = self.i8
        if b2:
            np.subtract(self.f32, self.bkg2, out=self.f32)
            np.maximum(0, self.f32, out=self.f32)
        if b1 or b2:
            self.i8[...] = self.f32
            return True
        else:
            return False

    def blur(self, n_blur):
        cv2.GaussianBlur(self.i8, (n_blur,n_blur), 0, dst=self.i8)
        
    # We assume the objects to detect are darker than the background.
    # If not invert the image right after reading it and converting to grayscale.
    def threshold(self, block_size, offset):
#        cv2.threshold( self.i8, 2*offset, 255, cv2.THRESH_BINARY, dst=self.i8 )
        cv2.adaptiveThreshold( self.i8, maxValue=255, 
                               adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               thresholdType=cv2.THRESH_BINARY,
                               blockSize=block_size, C=-offset, dst=self.i8 )
        
    def apply_morphological_transform(self, mtype, mval):
            cv2.morphologyEx(self.i8, mtype, mval, dst=self.i8)
        
    def detect_contours(self):
#        self.contours, hierarchy = cv2.findContours( self.i8, cv2.RETR_TREE, 
#                                                     cv2.CHAIN_APPROX_SIMPLE )[-2:]
        self.contours, hierarchy = cv2.findContours( self.i8, cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE )[-2:]
        # Restricting to outermost contours discards spots inside a fish, however
        # sometimes the edge of the image is detected as a contour and fish contours
        # get pushed to the second level in the hierarchy.
        
        # Note: findContours used to return a list of contours. The rest of the code
        # expects a list. In more recent versions of opencv, findContours return a 
        # tuple instead. The line below makes sure it's always a list.
        self.contours = list(self.contours)

    def analyze_contours(self, n_track, min_area, max_area, max_aspect, guess_front=False):
        self.coord = []
        if guess_front:
            contour_img = np.zeros_like(self.i8)
        for i,c in enumerate(self.contours):
            M = cv2.moments(c)
            # If area is valid, proceed with contour analysis.
            area = M['m00']
            if area>0 and min_area<=area<=max_area:
                x     = M['m10'] / area
                y     = M['m01'] / area
                theta = 0.5 * np.arctan2(2*M['mu11'], M['mu20']-M['mu02'])
                mu    = np.array([[M['mu20'],M['mu11']],[M['mu11'],M['mu02']]])/area
                eVal,eVec = la.eigh(mu)
                aspect = np.sqrt(eVal[1]/eVal[0])
                if guess_front:
                    cv2.drawContours(contour_img, [c], 0, color=i, thickness=-1)
                    Y,X = np.nonzero(contour_img==i)
                    # TODO: Try using width of the fish in the front half vs rear half instead
                    # of the skew along the long axis. Find out which one is more robust.
                    U = (X-x)*np.cos(theta)+(Y-y)*np.sin(theta)
                    if skew(U)>0:
                        theta += np.pi
                if aspect<=max_aspect:
                    M['valid'] = True
                    self.coord.append([x,y,theta,area,aspect])
                    continue
            self.contours[i] = None
        for i in range(len(self.coord),n_track):
            self.coord.append([np.nan]*5)
        self.coord = np.array(self.coord,dtype=np.float)
        self.contours = [ c for c in self.contours if not (c is None) ]

