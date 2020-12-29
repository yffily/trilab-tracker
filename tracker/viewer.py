#! /usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import pandas as pd
import utils as utils
from utils import create_named_window, wait_on_named_window
#from tank import Tank
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimediaWidgets
import pyqtgraph
from pyqtgraph.widgets.RawImageWidget import RawImageWidget


input_dir    = '../output_data/SF_Sat_14dpf_GroupA_n5_2020-06-13-120445-0000'
track_length = 200 # number of frames shown in past trajectory and in future trajectory
dot_size     = 6


class Track:

    def __init__(self, input_dir, track_length=track_length):
        background_file = os.path.join(input_dir,'background.npz')
        self.background = next(iter(np.load(background_file).values()))
        
        trial         = utils.load_pik(os.path.join(input_dir,'trial.pik'))
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
    
    def color(self,i):
        return utils.color_list[i%len(utils.color_list)]

    def draw_track(self,i,show_fish=True,show_track=True,show_extra=True):
        if not (show_track or show_fish):
            return
        l = np.searchsorted(self.frames,i)
        for m in range(self.track.shape[1]):
            color = self.color(m)
            if ( show_track and m<self.n_ind and \
                 self.frames[0]-track_length<i<self.frames[-1]+self.track_length ):
                points = self.track[max(0,l-self.track_length):l+self.track_length,m,:2]
                points = points[~np.any(np.isnan(points),axis=1)]
                points = points.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(self.frame, [points], False, color, 1)
            if show_fish and l<len(self.frames) and self.frames[l]==i:
                if not np.any(np.isnan(self.track[l,m,:2])):
                    x,y = self.track[l,m,:2].astype(np.int32)
                    if m<self.n_ind:
                        cv2.circle(self.frame, (x,y), dot_size, color, -1, cv2.LINE_AA)
                    elif show_extra:
                        cv2.line(self.frame, (x-dot_size,y-dot_size), (x+dot_size,y+dot_size) , color, 4)
                        cv2.line(self.frame, (x-dot_size,y+dot_size), (x+dot_size,y-dot_size) , color, 4)


class MainWidget(QtWidgets.QWidget):
    
    def __init__(self, input_dir, parent=None):
        super().__init__(parent)
        
        self.input_dir = input_dir
        self.track = Track(self.input_dir)
        
        self.video = RawImageWidget(scaled=True)
        self.video.setImage(self.cv2pg(self.track.frame))
        legend_layout = QtWidgets.QVBoxLayout()
        for i in range(self.track.n_tracks):
            row    = QtWidgets.QHBoxLayout()
            label  = QtWidgets.QLabel()
#            label.setStyleSheet(f'background-color: rgb{self.track.color(i)[::-1]}')
            pixmap = QtGui.QPixmap(10,10)
            pixmap.fill(QtGui.QColor(*self.track.color(i)[::-1]))
            label.setPixmap(pixmap)
            row.addWidget(label)
            label  = QtWidgets.QLabel(text=f'fish {i+1}')
            row.addWidget(label)
            legend_layout.addLayout(row)
        legend_layout.addStretch(1)
        video_layout = QtWidgets.QHBoxLayout()
        video_layout.addWidget(self.video)
        video_layout.addLayout(legend_layout)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(self.track.n_frames)
        self.frame_slider.valueChanged[int].connect(self.frame_slider_action)
        
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.play_pause)
        self.subtract_button = QtWidgets.QPushButton('Subtract')
        self.subtract_button.setCheckable(True)
        self.subtract_button.clicked.connect(self.toggle_subtract)
        self.track_button = self.make_show_hide_button('Track', True)
        self.fish_button = self.make_show_hide_button('Fish', True)
        self.extra_button = self.make_show_hide_button('Extra', False)
        self.reload_button = QtWidgets.QPushButton('Reload')
        self.reload_button.clicked.connect(self.reload)
        buttons_layout = QtWidgets.QHBoxLayout()
        for button in [ self.play_button, self.subtract_button, self.fish_button, 
                        self.track_button, self.extra_button, self.reload_button ]:
            buttons_layout.addWidget(button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(video_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.frame_slider)
        self.setLayout(layout)
        self.resize(800,800)
        
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.timeout)
        
        shortcuts  = { 'esc':quit, 'ctrl+q':quit, 'q':quit, 
                       ' ':self.spacebar, 'f':self.toggle_fullscreen, 
                       'right':self.next_frame, 'left':self.previous_frame }
        for key,action in shortcuts.items():
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
            
        self.subtract_button.toggle() # Turn on subtraction by default.
        self.frame_slider.setValue(0) # Go to beginning of video.
        self.frame_slider_action(self.frame_slider.value()) # Redraw current frame.

    def reload(self):
        self.track = Track(self.input_dir)
        self.frame_slider_action(self.frame_slider.value())

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
    
    def toggle_subtract(self):
        self.frame_slider_action(self.frame_slider.value())

    def toggle_show_hide(self,button,text):
        if button.isChecked():
            button.setText('Hide '+text)
        else:
            button.setText('Show '+text)
        self.frame_slider_action(self.frame_slider.value())

    def make_show_hide_button(self,text,initial_value=True):
        button = QtWidgets.QPushButton('Show '+text)
        button.setCheckable(True)
        toggle = lambda _,button=button,text=text: self.toggle_show_hide(button,text)
        button.clicked.connect(toggle)
        if initial_value:
            button.toggle()
        return button

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

    def frame_slider_action(self,value):
        if value==self.track.current_frame()+1:
            self.track.read_frame()
        else:
            self.track.read_frame(value)
        if self.subtract_button.isChecked():
            self.track.subtract_background()
        self.track.draw_track(value, show_fish=self.fish_button.isChecked(), 
                              show_track=self.track_button.isChecked(), 
                              show_extra=self.extra_button.isChecked())
        self.video.setImage(self.cv2pg(self.track.frame))
    
    # Convert an openCV image matrix to a pyqtgraph image matrix. 
    def cv2pg(self,img):
        img = np.swapaxes(img,0,1) # Swap rows and columns.
#        img = np.flip(img,0) # Flip row order.
#        img = np.flip(img,1) # Flip column order.
        img = np.flip(img,2) # Flip color channel order.
        return img
        

if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(sys.argv)
        input_dir = sys.argv[1]
        wdg = MainWidget(input_dir)
        wdg.show()
        sys.exit(app.exec_())
    except:
        sys.exit(1)

