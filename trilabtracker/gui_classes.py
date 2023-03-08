import sys
import os
import cv2
from glob import glob
import numpy as np
from . import utils
from .frame import Frame
from .tank import Tank
from . import fixer
from PyQt5 import QtCore, QtGui, QtWidgets
try:
    from PyQt5.QtGui import QDialog
except:
    from PyQt5.QtWidgets import QDialog
import pyqtgraph
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent
import flatten_dict


#pyqtgraph.setConfigOption('leftButtonPan', False)

def get_color(i):
    return utils.color_list[i%len(utils.color_list)]

#----------------------------------------------------------

class Config():
    
    def __init__(self):
        self.marker_size      = 8
        self.marker_lw        = 2
        self.arrow_tip_length = 0.5
        self.line_width       = 1
        self.select_radius    = 15
        # Colors in BGR space.
        self.contour_color    = (0,255,0)
        self.text_color       = (0,255,0)
        self.keys             = self.__dict__.copy().keys()

    def items(self):
        return { k:v for k,v in self.__dict__.items() if k in self.keys }

    def dialog(self):
        dialog  = QDialog()
        layout  = QtWidgets.QGridLayout()
        widgets = dict( marker_size   = QtWidgets.QSpinBox(), 
                        marker_lw     = QtWidgets.QSpinBox(), 
                        arrow_tip_length = QtWidgets.QDoubleSpinBox(), 
                        line_width    = QtWidgets.QSpinBox(), 
#                        contour_color = QtWidgets.QColorDialog(), 
#                        text_color    = QtWidgets.QColorDialog(), 
                        select_radius = QtWidgets.QSpinBox() )
        widgets['arrow_tip_length'].setSingleStep(0.1)
        for i,(k,w) in enumerate(widgets.items()):
            if hasattr(w, 'setValue'):
                w.setValue(self.__dict__[k])
                def changed(value, self=self, k=k):
                    self.__dict__[k] = value
                w.valueChanged.connect(changed)
            layout.addWidget(QtWidgets.QLabel(k), i, 0)
            layout.addWidget(w, i, 1)
        button = QtWidgets.QPushButton('Reset')
        def reset(_, self=self, widgets=widgets):
            self.__init__()
            for k,w in widgets.items():
                if hasattr(w, 'setValue'):
                    w.setValue(self.__dict__[k])
        button.clicked.connect(reset)
        layout.addWidget(button)
#        button_box = QDialogButtonBox(QtGui.QDialogButtonBox.Ok|QtGui.QDialogButtonBox.Cancel)
#        button_box.accepted.connect(dialog.accept)
#        button_box.rejected.connect(dialog.reject)
#        layout.addWidget(button_box)
        dialog.setLayout(layout)
        dialog.exec_()
#        if dialog.exec_():
#            for k,w in widgets.items():
#                if hasattr(w, 'setValue'):
#                    self.__dict__[k] = w.value()

#----------------------------------------------------------

class Track:

    def __init__(self, input_dir):
        self.__dict__.update(Config().items())
        self.join       = lambda path,d=input_dir: os.path.join(d,path)
        trial           = utils.load_pik(self.join('trial.pik'))
        self.load_tracks(trial)
        self.load_parameters()
    
    def load_tracks(self, trial=None):
        trial = utils.load_pik(self.join('trial.pik')) if trial is None else trial
        self.fps        = trial['fps']
        self.frames     = trial['frame_list']
        self.tracks     = trial['data']
        self.n_ind      = trial['n_ind']
        self.n_tracks   = self.tracks.shape[1]
        self.bad_frames = np.zeros(len(self.frames), dtype=np.int8)
    
    def locate_bad_frames(self):
        self.bad_frames[:] = 0
        X = self.tracks[:,:self.n_ind,:3]
        if not self.bad_displacement is None:
            dX = X[1:,:,:2]-X[:-1,:,:2]
            dX[np.isnan(dX)] = 0
            I = np.nonzero(np.any(np.sum(dX**2, axis=2)>self.bad_displacement**2, axis=1))[0]
            self.bad_frames[I+1] = 2
        I = np.nonzero(np.any(np.isnan(X), axis=(1,2)))
        self.bad_frames[I] = 1
    
    def load_parameters(self):
        self.settings = utils.load_settings(self.join('settings.txt'))
        self.settings = flatten_dict.flatten(self.settings, 'dot')
        
        # Load the video. Always use the txt link, which works in windows as well.
        input_video = self.join('raw.txt')
        with open(input_video) as fh:
            input_video = fh.readline().strip()
        input_video = self.join(input_video)
        
        self.cap       = cv2.VideoCapture(input_video)
        self.n_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height    = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame     = Frame((self.height,self.width))
        self.frame.contrast_factor = self.settings['bkg.contrast_factor']
        self.bgr       = np.empty(shape=(self.height,self.width,3), dtype=np.uint8)
        self.overlay   = np.empty_like(self.bgr)
        self.read_frame()
        
        self.tank      = Tank()
        self.tank.load(self.join('tank.pik'))
        self.frame.mask = self.tank.create_mask((self.height,self.width))
        
        bkg_file = self.join('background.npz')
        self.frame.bkg  = next(iter(np.load(bkg_file).values()))
        bkg2_file = self.join('background2.npz')
        if os.path.exists(bkg2_file):
            self.frame.bkg2_ = next(iter(np.load(bkg2_file).values()))
            self.frame.bkg2  = self.frame.bkg2_.copy()
            self.frame.bkg2 *= self.settings['bkg.secondary_factor'] * \
                                  self.settings['bkg.contrast_factor']

    def current_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def current_timestamp(self):
        s = (self.current_frame()-1)/self.fps
        m,s = divmod(s,60)
        return m,s
#        h,m = divmod(m,60)
#        return h,m,s # f'{h:02.0f}:{m:02.0f}:{s:05.2f}'

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

    def draw_fish(self, xy_th, color):
        x,y,th = xy_th
        if not (np.isnan(x) or np.isnan(y)):
            if np.isnan(th):
                cv2.circle(self.overlay, (int(x),int(y)), self.marker_size, color, self.marker_lw)
            else:
                ux,uy = self.marker_size*np.cos(th),self.marker_size*np.sin(th)
                x1,y1,x2,y2 = int(x-ux),int(y-uy),int(x+ux),int(y+uy)
                cv2.arrowedLine( self.overlay, (x1,y1), (x2,y2), color=color, 
                                 thickness=self.marker_lw, tipLength=self.arrow_tip_length )

    def draw_extra(self, xy, color):
        x,y = xy
        if not (np.isnan(x) or np.isnan(y)):
            x,y = int(x),int(y)
            xy = [(x+a*self.marker_size,y+b*self.marker_size) for a in [-1,1] for b in [-1,1]]
            cv2.line(self.overlay, xy[0], xy[3], color, self.marker_lw)
            cv2.line(self.overlay, xy[1], xy[2], color, self.marker_lw)

    def draw_scale_bar(self, length_px, pad=10):
        x,y,l,w = pad,self.height-pad,int(length_px),int(2*self.marker_lw)
        cv2.line(self.overlay, (x,y), (x+l,y), self.text_color, self.marker_lw)
        cv2.line(self.overlay, (x,y-w), (x,y+w), self.text_color, self.marker_lw)
        cv2.line(self.overlay, (x+l,y-w), (x+l,y+w), self.text_color, self.marker_lw)

    def draw_time(self, show_frame_number, show_timestamp):
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        y = 30
        if show_timestamp:
            m,s   = self.current_timestamp()
            t_str = f'{m:02.0f}:{s:05.2f}'
            cv2.putText(self.overlay, t_str, (5,y), font, 1, self.text_color, 2)
            y += 30
        if show_frame_number:
            t_str = f'{self.current_frame()}'
            cv2.putText(self.overlay, t_str, (5,y), font, 1, self.text_color, 2)

    def draw(self, i, track_length, show_fish=True, show_extra=True, show_tank=False, 
             show_contours=False, show_frame_number=False, show_timestamp=False, scale_bar=None):
        has_overlay = False
        self.overlay.fill(0)
        if show_tank:
            self.tank.draw_outline(self.overlay, color=self.contour_color, 
                                   thickness=self.line_width)
            has_overlay = True
        if show_contours:
            cv2.drawContours(self.overlay, self.frame.contours, -1, 
                             self.contour_color, self.line_width)
            has_overlay = True
        if not (track_length>0 or show_fish or show_extra):
            return has_overlay
        l = np.searchsorted(self.frames,i)
        if not (l<len(self.frames) and self.frames[l]==i):
            return has_overlay
        for m in range(self.n_tracks):
            color = get_color(m)
            if track_length>0 and m<self.n_ind:
                points = self.tracks[max(0,l-track_length):l+track_length,m,:2]
                points = points[~np.any(np.isnan(points),axis=1)]
                points = points.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(self.overlay, [points], False, color, self.line_width)
                has_overlay = True
            if show_fish and m<self.n_ind:
                self.draw_fish(self.tracks[l,m,:3], color)
                has_overlay = True
            if show_extra and m>=self.n_ind:
                self.draw_extra(self.tracks[l,m,:2], color)
                has_overlay = True
        if show_frame_number or show_timestamp:
            self.draw_time(show_frame_number, show_timestamp)
            has_overlay = True
        if not scale_bar is None:
            self.draw_scale_bar(scale_bar)
            has_overlay = True
        # Return True if an overlay was created.
        return has_overlay

    # If xy is in vicinity of a fish in frame i, return that fish's id.
    def select_fish(self, xy, i=None):
        if i is None:
            i = self.current_frame()-1
        xy = np.array(xy)
        d  = np.hypot(*(self.tracks[i,:,:2]-xy[None,:]).T)
        if np.all(np.isnan(d)):
            return None
        j  = np.nanargmin(d)
        return j if d[j]<self.select_radius else None


    def fix(self, *fix, history=None, recompute_bad_frames=True):
        i = self.current_frame()-1 if fix[1] is None else fix[1]
        fix = fix[:1] + (i,) + fix[2:]
        fixer.fix(self.tracks, *fix)
#        getattr(self,'fix_'+fix[0])(*fix[1:])
        if not history is None:
            history.add_fix(fix)
        if recompute_bad_frames:
            self.locate_bad_frames()

#----------------------------------------------------------

class History(QtWidgets.QListWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.fixes = []

    def fix2str(self, fix):
        return ', '.join(map(str,fix))

    def add_fix(self, fix):
        self.fixes.append(fix)
        self.addItem(QtWidgets.QListWidgetItem(self.fix2str(fix)))
        self.repaint()
    
    def remove_fix(self, i=None):
        i = self.count()-1 if i is None else i
        self.fixes.pop(i)
        self.takeItem(i)
        self.repaint()

    def sync(self):
        self.clear()
        for fix in self.fixes:
            self.addItem(QtWidgets.QListWidgetItem(self.fix2str(fix)))
        self.repaint()

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
        self.setImage(img, autoRange=False, autoLevels=False, levels=(0,255))
    
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

#----------------------------------------------------------

