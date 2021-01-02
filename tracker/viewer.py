#! /usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import utils as utils
from utils import create_named_window, wait_on_named_window
#from tank import Tank
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimediaWidgets
import pyqtgraph
from pyqtgraph.widgets.RawImageWidget import RawImageWidget


input_dir    = '../output_data/SF_Sat_14dpf_GroupA_n5_2020-06-13-120445-0000'
track_length = 200 # number of frames shown in past trajectory and in future trajectory
marker_size  = 5
marker_lw    = 2
arrow_size   = 10

class Track:

    def __init__(self, input_dir, track_length=track_length):
        background_file = os.path.join(input_dir,'background.npz')
        self.background = next(iter(np.load(background_file).values()))
        
        background_file2 = os.path.join(input_dir,'background2.npz')
        if os.path.exists(background_file2):
            self.background2 = next(iter(np.load(background_file2).values()))
        else:
            self.background2 = None
        
        trial         = utils.load_pik(os.path.join(input_dir,'trial.pik'))
        self.fps      = trial['fps']
        self.frames   = trial['frame_list']
        self.track    = trial['data']
        self.n_ind    = trial['n_ind']
        self.n_tracks = self.track.shape[1]
        self.track_length = track_length
        
        input_video   = os.path.join(input_dir,'raw.avi')
        self.cap      = cv2.VideoCapture(input_video)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width         = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height        = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame    = np.empty(shape=(height,width,3), dtype=np.uint8)
        self.frame_float = np.empty(shape=(height,width,3), dtype=float)
        self.read_frame()
    
    def current_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def read_frame(self,i=None):
        if isinstance(i,int):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,i-1)
        ret, self.frame = self.cap.read()
        return ret
    
    def subtract_background(self):
        self.frame_float = np.absolute(4*(self.frame-self.background))
        self.frame  = (255 - np.minimum(255, self.frame_float.astype(np.uint8)))
    
    def subtract_secondary_background(self):
        if not self.background2 is None:
            self.frame = np.minimum(255,self.frame+self.background2).astype(np.uint8)

    def color(self,i):
        return utils.color_list[i%len(utils.color_list)]

    def draw_track(self,i,show_fish=True,show_track=True,show_extra=True):
        if not (show_track or show_fish):
            return
        l = np.searchsorted(self.frames,i)
        if not (l<len(self.frames) and self.frames[l]==i):
            return
        for m in range(self.n_tracks):
            color = self.color(m)
            if show_track and m<self.n_ind:
                points = self.track[max(0,l-self.track_length):l+self.track_length,m,:2]
                points = points[~np.any(np.isnan(points),axis=1)]
                points = points.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(self.frame, [points], False, color, 1)
            if show_fish:
                if not np.any(np.isnan(self.track[l,m,:2])):
                    x,y = self.track[l,m,:2].astype(np.int32)
                    if m<self.n_ind:
#                        cv2.circle(self.frame, (x,y), marker_size, color, marker_lw, cv2.LINE_AA)
                        XY = self.track[l,m,:2]
                        Th = self.track[l,m,2]
                        U  = arrow_size*np.array([np.cos(Th),np.sin(Th)])
                        (x1,y1),(x2,y2) = (XY-U).astype(int),(XY+U).astype(int)
                        cv2.arrowedLine(self.frame, (x1,y1), (x2,y2), color=color, 
                                        thickness=2*marker_lw, tipLength=0.5)
                    elif show_extra:
                        xy = [ (x+a*marker_size,y+b*marker_size) for a in [-1,1] for b in [-1,1] ]
                        cv2.line(self.frame, xy[0], xy[3], color, marker_lw)
                        cv2.line(self.frame, xy[1], xy[2], color, marker_lw)


class MainWidget(QtWidgets.QWidget):
    
    def __init__(self, input_dir, parent=None):
        super().__init__(parent)
        
        self.input_dir = input_dir
        self.track = Track(self.input_dir)
        
        self.video = RawImageWidget(scaled=True)
        
        
        tabs = { 'View':QtWidgets.QVBoxLayout(), 'Fix':QtWidgets.QVBoxLayout() }
        for k in tabs.keys():
            tabs[k].addWidget(QtWidgets.QLabel('Legend:'))
            for i in range(self.track.n_tracks):
                row    = QtWidgets.QHBoxLayout()
                label  = QtWidgets.QLabel()
                pixmap = QtGui.QPixmap(10,10)
                pixmap.fill(QtGui.QColor(*self.track.color(i)[::-1]))
                label.setPixmap(pixmap)
                row.addWidget(label)
                text = f'fish {i+1}' if i<self.track.n_ind else f'object {i+1}'
                label  = QtWidgets.QLabel(text=text)
                row.addWidget(label)
                row.addStretch(1)
                tabs[k].addLayout(row)
        tabs['View'].addWidget(QtWidgets.QLabel(' '))
        tabs['View'].addWidget(QtWidgets.QLabel('Viewing Options:'))
        self.checkboxes = {}
        for k in ['Subtract Background', 'Subtract Secondary Background', 
                   'Show Fish', 'Show Track', 'Show Extra Objects']:
            self.checkboxes[k] = QtWidgets.QCheckBox(k)
            self.checkboxes[k].stateChanged.connect(self.redraw)
            tabs['View'].addWidget(self.checkboxes[k])
#        tabs['Fix'].addWidget(QtWidgets.QLabel(' '))
#        tabs['Fix'].addWidget(QtWidgets.QLabel('Active Fish:'))
#        self.active_fish = QtWidgets.QComboBox()
#        for i in range(self.track.n_ind):
#            self.active_fish.addItem(f'fish {i+1}')
#        tabs['Fix'].addWidget(self.active_fish)
        tabs['Fix'].addWidget(QtWidgets.QLabel(' '))
        tabs['Fix'].addWidget(QtWidgets.QLabel('Not implemented yet.'))
                
        tab_widget = QtWidgets.QTabWidget()
        for k in tabs.keys():
            tabs[k].addStretch(1)
            tab = QtWidgets.QWidget()
            tab.setLayout(tabs[k])
            tab_widget.addTab(tab,k)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(self.track.n_frames)
        self.frame_slider.valueChanged.connect(self.redraw)
        
        self.clock = QtWidgets.QLabel()
        
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.play_pause)
        self.reload_button = QtWidgets.QPushButton('Reload')
        self.reload_button.clicked.connect(self.reload)
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.clock)
        for button in [ self.play_button, self.reload_button ]:
            buttons_layout.addWidget(button)
        
        layout1 = QtWidgets.QHBoxLayout() # Video & right-side panel.
        layout1.addWidget(self.video,2)
        layout1.addWidget(tab_widget)
        layout = QtWidgets.QVBoxLayout() # Full layout.
        layout.addLayout(layout1)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.frame_slider)
        self.setLayout(layout)
        self.resize(800,600)
        
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.timeout)
        
        shortcuts  = { 'esc':quit, 'ctrl+q':quit, 'q':quit, 
                       ' ':self.spacebar, 'f':self.toggle_fullscreen, 
                       'right':self.next_frame, 'left':self.previous_frame }
        for key,action in shortcuts.items():
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
        
        
        for k,cb in self.checkboxes.items():
            cb.setChecked(True)        
        self.frame_slider.setValue(self.track.frames[0]) # Go to beginning of video.
        self.redraw()
    
    def reload(self):
        self.track = Track(self.input_dir)
        self.redraw()

    def timeout(self):
        if self.play_button.isChecked():
            self.frame_slider.setValue(self.frame_slider.value()+1)
    
    def play_pause(self):
        if self.play_button.isChecked():
            self.play_button.setText('Pause')
            self.timer.start()
        else:
            self.play_button.setText('Play')
            self.timer.stop()
    
    def toggle_fullscreen(self):
        if self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showFullScreen()

    def spacebar(self):
        self.play_button.toggle()
        self.play_pause()

    def next_frame(self):
        # If playing, stop playing.
        if self.play_button.isChecked():
            self.play_button.toggle()
            self.play_pause()
        # Move to next frame.
        self.frame_slider.setValue(self.frame_slider.value()+1)

    def previous_frame(self):
        # If playing, stop playing.
        if self.play_button.isChecked():
            self.play_button.toggle()
            self.play_pause()
        # Move to previous frame.
        self.frame_slider.setValue(self.frame_slider.value()-1)
    
    def redraw(self):
        value = self.frame_slider.value()
        t     = value/self.track.fps
#        self.clock.setText(f'Frame {value} Time {t//60:.0f}:{t%60:05.2f}')
        self.clock.setText(f'Time {t//60:.0f}:{t%60:05.2f} / Frame {value}')
        if value==self.track.current_frame()+1:
            self.track.read_frame()
        else:
            self.track.read_frame(value)
        if self.checkboxes['Subtract Background'].isChecked():
            self.track.subtract_background()
        if self.checkboxes['Subtract Secondary Background'].isChecked():
            self.track.subtract_secondary_background()
        self.track.draw_track( value, show_fish=self.checkboxes['Show Fish'].isChecked(), 
                               show_track=self.checkboxes['Show Track'].isChecked(), 
                               show_extra=self.checkboxes['Show Extra Objects'].isChecked() )
        self.video.setImage(self.cv2pg(self.track.frame))
    
    # Convert an openCV image matrix to a pyqtgraph image matrix. 
    def cv2pg(self,img):
        img = np.swapaxes(img,0,1) # Swap rows and columns.
        img = np.flip(img,2) # Flip color channel order.
        return img
        

if __name__ == '__main__':
#    try:
    app = QtWidgets.QApplication(sys.argv)
    input_dir = sys.argv[1]
    wdg = MainWidget(input_dir)
    wdg.show()
    sys.exit(app.exec_())
#    except:
#        sys.exit(1)

