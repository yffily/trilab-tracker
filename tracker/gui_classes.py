import sys
import os
import cv2
import numpy as np
from .utils import color_list, load_pik, load_txt
from .frame import FrameAnalyzer
from .tank import Tank
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent
import flatten_dict


marker_size   = 5
marker_lw     = 3
arrow_size    = 10
contour_width = 2
contour_color = (0,255,0) # bgr
select_radius = 15

#pyqtgraph.setConfigOption('leftButtonPan', False)

def get_color(i):
    return color_list[i%len(color_list)]


#----------------------------------------------------------

class Track:

    def __init__(self, input_dir):
        trial          = load_pik(os.path.join(input_dir,'trial.pik'))
        self.fps       = trial['fps']
        self.frames    = trial['frame_list']
        self.track     = trial['data']
        self.n_ind     = trial['n_ind']
        self.n_tracks  = self.track.shape[1]
        
        self.fixes     = []
        
        self.settings  = load_txt(os.path.join(input_dir,'settings.txt'))
        for k,v in self.settings.items():
            self.settings[k] = eval(v)
        self.settings  = flatten_dict.flatten(self.settings,'dot')
        
        input_video    = os.path.join(input_dir,'raw.avi')
        self.cap       = cv2.VideoCapture(input_video)
        self.n_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width          = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height         = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame     = FrameAnalyzer((height,width))
        self.frame.contrast_factor = self.settings['bkgSub_options.contrast_factor']
        self.bgr       = np.empty(shape=(height,width,3), dtype=np.uint8)
        self.overlay   = np.empty_like(self.bgr)
        self.read_frame()
        
        tank           = Tank()
        tank.load(os.path.join(input_dir,'tank.pik'))
        self.frame.mask = tank.create_mask((height,width))
        
        bkg_file = os.path.join(input_dir,'background.npz')
        self.frame.bkg  = next(iter(np.load(bkg_file).values()))
        bkg2_file = os.path.join(input_dir,'background2.npz')
        if os.path.exists(bkg2_file):
            self.frame.bkg2_ = next(iter(np.load(bkg2_file).values()))
            self.frame.bkg2  = self.frame.bkg2_.copy()
            self.frame.bkg2 *= self.settings['bkgSub_options.secondary_factor'] * \
                                  self.settings['bkgSub_options.contrast_factor']

    def current_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def read_frame(self, i=None):
        if isinstance(i,int):
            di = i-self.current_frame()
            if 1<di<20:
                for j in range(di-1):
                    self.cap.grab()
            elif di==1:
                pass
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,i-1)
        ret,self.bgr = self.cap.read()
        self.frame.from_bgr(self.bgr)
        return ret
    
    def draw(self, i, track_length, show_fish=True, show_extra=True, show_contours=False):
        has_overlay = False
        self.overlay.fill(0)
        if show_contours:
            cv2.drawContours(self.overlay, self.frame.contours, -1, contour_color, contour_width)
            has_overlay = True
        if not (track_length>0 or show_fish or show_extra):
            return has_overlay
        l = np.searchsorted(self.frames,i)
        if not (l<len(self.frames) and self.frames[l]==i):
            return has_overlay
        for m in range(self.n_tracks):
            color = get_color(m)
            if track_length>0 and m<self.n_ind:
                points = self.track[max(0,l-track_length):l+track_length,m,:2]
                points = points[~np.any(np.isnan(points),axis=1)]
                points = points.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(self.overlay, [points], False, color, 1)
                has_overlay = True
            if (show_fish or show_extra) and not np.any(np.isnan(self.track[l,m,:2])):
                x,y = self.track[l,m,:2].astype(np.int32)
                if show_fish and m<self.n_ind:
                    XY = self.track[l,m,:2]
                    Th = self.track[l,m,2]
                    U  = arrow_size*np.array([np.cos(Th),np.sin(Th)])
                    (x1,y1),(x2,y2) = (XY-U).astype(int),(XY+U).astype(int)
                    cv2.arrowedLine( self.overlay, (x1,y1), (x2,y2), color=color, 
                                     thickness=marker_lw, tipLength=0.5 )
                    has_overlay = True
                if show_extra and m>=self.n_ind:
                    xy = [(x+a*marker_size,y+b*marker_size) for a in [-1,1] for b in [-1,1]]
                    cv2.line(self.overlay, xy[0], xy[3], color, marker_lw)
                    cv2.line(self.overlay, xy[1], xy[2], color, marker_lw)
                    has_overlay = True
        # Return True if an overlay was created.
        return has_overlay

    # If xy is in vicinity of a fish in frame i, return that fish's id.
    def select_fish(self, xy, i=None):
        if i is None:
            i = self.current_frame()-1
        xy = np.array(xy)
        d  = np.hypot(*(self.track[i,:,:2]-xy[None,:]).T)
        j  = np.nanargmin(d)
#        return (j,self.track[i,j,:2]) if d[j]<select_radius else None
        return j if d[j]<select_radius else None

    # Swap fish j1 and j2 from frame i on.
    def swap(self, j1, j2, i=None):
        if i is None:
            i = self.current_frame()-1
        self.track[i:,[j1,j2]] = self.track[i:,[j2,j1]]
        self.fixes.append(('swap',(i,j1,j2)))
        return

    # Move fish j to position xy in frame i.
    def move(self, j, xy, i=None):
        if i is None:
            i = self.current_frame()-1
        self.track[i,j,:2] = xy
        self.fixes.append(('move',(i,j,xy)))
        return

#    def save_fixes(self):
#        with open(os.path.join(input_dir,'fixes.txt'),'w') as f:
#            for fix in self.fixes:
#                print(fix,file=f)
#        return

#----------------------------------------------------------

class Video(pyqtgraph.ImageView):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
#        self.getImageItem().mouseClickEvent = self.click
        
    # Convert from openCV image matrix to pyqtgraph image matrix then show on screen.
    def setImage_(self, img):
        img = np.swapaxes(img, 0, 1) # Swap rows and columns.
        img = np.flip(img, 2) # Flip color channel order.
        self.setImage(img, autoRange=False)
    
#    def click(self, event):
#        print('click event:', event.pos())

#----------------------------------------------------------

class SidePanel(QtWidgets.QTabWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create right panel tabs.
        for k in ['tune', 'view', 'fix']:
            tab = QtWidgets.QWidget()
            self.__dict__[k] = QtWidgets.QVBoxLayout(tab)
            self.addTab(tab, k.capitalize())

