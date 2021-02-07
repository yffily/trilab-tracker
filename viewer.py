#! /usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import pandas as pd
import datetime
from tracker import utils
from tracker.frame import FrameAnalyzer
from tracker.tank import Tank
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimediaWidgets
import pyqtgraph
from pyqtgraph.widgets.RawImageWidget import RawImageWidget
#import argparse
import flatten_dict


marker_size   = 5
marker_lw     = 3
arrow_size    = 10
contour_width = 2
contour_color = (0,255,0) # bgr


class Track:

    def __init__(self, input_dir):
        trial          = utils.load_pik(os.path.join(input_dir,'trial.pik'))
        self.fps       = trial['fps']
        self.frames    = trial['frame_list']
        self.track     = trial['data']
        self.n_ind     = trial['n_ind']
        self.n_tracks  = self.track.shape[1]
        
        self.settings  = utils.load_txt(os.path.join(input_dir,'settings.txt'))
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
    
    def color(self, i):
        return utils.color_list[i%len(utils.color_list)]

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
            color = self.color(m)
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


class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, input_dir, parent=None):
        super().__init__(parent)
        
        if input_dir is None:
            self.choose_input_dir()
        else:
            self.input_dir = input_dir
        self.track = Track(self.input_dir)
        self.video = RawImageWidget(scaled=True)
        
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.timeout)
        
        #--------------------
        # Widgets.
        
        # Create widget to show current frame number.
        self.clock = QtWidgets.QLabel()

        # Create spinboxes.
        self.tunables = {}
        spins  = [ 'n_blur', 'block_size', 'offset', 'min_area', 'max_area', 
                   'ideal_area', 'Read one frame in' ]
        dspins = [ 'max_aspect', 'ideal_aspect', 'contrast_factor', 
                   'secondary_factor', 'Track length (s)' ]
        for k in spins+dspins:
            self.tunables[k] = QtWidgets.QSpinBox() if k in spins \
                                   else QtWidgets.QDoubleSpinBox()
            self.tunables[k].setRange(0,1000)
        self.tunables['Read one frame in'].setRange(-100,100)
        self.reset_tunables()
        
        # Create checkboxes.
        self.checkboxes = {}
        for k in [ 'Subtract Background', 'Subtract Secondary Background', 
                   'Apply Tank Mask', 'Threshold', 'Show Contours', 
                   'Show Fish', 'Show Extra Objects', 'Show Track' ]:
            self.checkboxes[k] = QtWidgets.QCheckBox(k)
        
        # Create sliders.
        self.sliders = {}
        for k in [ 'frame', 'alpha' ]:
            self.sliders[k] = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliders['frame'].setRange(1, self.track.n_frames)
        self.sliders['alpha'].setRange(0, 100)
        
        # Create buttons.
        self.buttons = {}
        self.buttons['reset settings'] = QtWidgets.QPushButton('Reset')
        self.buttons['play'] = QtWidgets.QPushButton('Play')
        self.buttons['play'].setCheckable(True)
        
        # Create right panel tabs.
        tabs = { 'Tune':QtWidgets.QVBoxLayout(), 
                 'View':QtWidgets.QVBoxLayout(), 
                 'Fix':QtWidgets.QVBoxLayout() }
        self.tab_widget = QtWidgets.QTabWidget()
        for k in tabs.keys():
            tab = QtWidgets.QWidget()
            tab.setLayout(tabs[k])
            self.tab_widget.addTab(tab,k)
        
        # Connect widgets to appropriate actions.
        self.tab_widget.currentChanged.connect(self.redraw)
        for k in self.tunables.keys():
            if k in ['contrast_factor', 'secondary_factor']:
                self.tunables[k].valueChanged.connect(self.update_bkgSub)
            self.tunables[k].valueChanged.connect(self.redraw)
        for k in self.checkboxes.keys():
            self.checkboxes[k].stateChanged.connect(self.redraw)
        for k in self.sliders.keys():
            self.sliders[k].valueChanged.connect(self.redraw)
        self.buttons['reset settings'].clicked.connect(self.reset_tunables)
        self.buttons['play'].clicked.connect(self.play_pause)
        
        # Create fish legend as a layout.
        left   = QtWidgets.QVBoxLayout()
        right  = QtWidgets.QVBoxLayout()
        for i in range(self.track.n_tracks):
            row    = QtWidgets.QHBoxLayout()
            label  = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(10,10)
            pixmap.fill(QtGui.QColor(*self.track.color(i)[::-1]))
            label.setPixmap(pixmap)
            row.addWidget(label)
            text   = f'fish {i+1}' if i<self.track.n_ind else f'object {i+1}'
            label  = QtWidgets.QLabel(text=text)
            row.addWidget(label)
            row.addStretch(1)
            if i<self.track.n_tracks/2:
                left.addLayout(row)
            else:
                right.addLayout(row)
        right.addStretch(1)
        legend = QtWidgets.QHBoxLayout()
        legend.addLayout(left)
        legend.addLayout(right)
        
        # Create a widget with a tunable's name and spinbox side-by-side.
        def create_tunable_row(k):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(k))
            row.addWidget(self.tunables[k])
            return row
        
        # Set up 'Tune' tab.
        tab = tabs['Tune']
        for k in [ 'Subtract Background', 'Subtract Secondary Background', 
                   'Apply Tank Mask', 'Threshold', 'Show Contours' ]:
            tab.addWidget(self.checkboxes[k])
        for k in [ 'n_blur', 'block_size', 'offset', 'min_area', 'max_area', 
                   'ideal_area', 'max_aspect', 'ideal_aspect', 
                   'contrast_factor', 'secondary_factor' ]:
            tab.addLayout(create_tunable_row(k))
        tab.addWidget(self.buttons['reset settings'])
        tab.addStretch(1)
        
        # Set up 'View' tab.
        tab = tabs['View']
        tab.addLayout(legend)
        tab.addWidget(QtWidgets.QLabel(' '))
        for k in [ 'Show Fish', 'Show Extra Objects', 'Show Track', 'Show Contours' ]:
            tab.addWidget(self.checkboxes[k])
        tab.addWidget(QtWidgets.QLabel(' '))
        tab.addLayout(create_tunable_row('Track length (s)'))
        tab.addLayout(create_tunable_row('Read one frame in'))
        tab.addWidget(QtWidgets.QLabel(' '))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Overlay transparency'))
        row.addWidget(self.sliders['alpha'])
        tab.addLayout(row)
        tab.addStretch(1)
        
        # Set up 'Fix' tab.
        tab = tabs['Fix']
#        tab.addLayout(legend)
#        tab.addWidget(QtWidgets.QLabel(' '))
#        tab.addWidget(QtWidgets.QLabel('Active Fish:'))
#        self.active_fish = QtWidgets.QComboBox()
#        for i in range(self.track.n_ind):
#            self.active_fish.addItem(f'fish {i+1}')
#        tab.addWidget(self.active_fish)
#        tab.addWidget(QtWidgets.QLabel(' '))
        tab.addWidget(QtWidgets.QLabel('Not implemented yet.'))
        tab.addStretch(1)
        
        #--------------------
        # Window layout.
        
        # Video & right-side panel.
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.video,2)
        layout1.addWidget(self.tab_widget)
        # Playback buttons.
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.clock)
        layout2.addWidget(self.buttons['play'])
        # Final layout.
        layout  = QtWidgets.QVBoxLayout()
        layout.addLayout(layout1)
        layout.addLayout(layout2)
        layout.addWidget(self.sliders['frame'])
        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(800,600)
        self.setWindowTitle(self.input_dir)
        
        #--------------------
        # Menu bar.
        
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        action = QtWidgets.QAction('Open...',self)
        action.triggered.connect(self.open)
        file_menu.addAction(action)
        action = QtWidgets.QAction('Reload',self)
        action.triggered.connect(self.reload)
        file_menu.addAction(action)
        action = QtWidgets.QAction('Exit',self)
        action.triggered.connect(quit)
        file_menu.addAction(action)

        #--------------------
        # Shortcuts.
        
        shortcuts  = { 'esc':quit, 'ctrl+q':quit, 'q':quit, 'f5':self.reload, 
                       ' ':self.spacebar, 'f':self.toggle_fullscreen,
                       'right':self.next_frame, 'left':self.previous_frame }
        for key,action in shortcuts.items():
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(action)
        
        #--------------------
        # Starting options.
        
        self.tab_widget.setCurrentIndex(1) # Start in 'View' tab.
        i = self.track.frames[0] if len(self.track.frames)>0 else 0
        self.sliders['frame'].setValue(i) # Go to beginning of video.
        self.sliders['alpha'].setValue(50) # Go to beginning of video.
        self.checkboxes['Show Fish'].setChecked(True)
        self.checkboxes['Show Extra Objects'].setChecked(True)
        self.checkboxes['Show Track'].setChecked(True)
        self.tunables['Track length (s)'].setValue(10)
        self.tunables['Read one frame in'].setValue(5)
        self.reset_tunables()
    
    
    def choose_input_dir(self):
        input_dir = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', 
                                  '.', QtGui.QFileDialog.ShowDirsOnly)
        if input_dir:
            self.input_dir = input_dir
    
    def open(self):
        self.choose_input_dir()
        self.reload()
    
    def reload(self):
        self.track = Track(self.input_dir)
        self.redraw()
        self.window().setWindowTitle(self.input_dir)

    def timeout(self):
        if self.buttons['play'].isChecked():
            self.sliders['frame'].setValue(self.sliders['frame'].value()+self.tunables['Read one frame in'].value())
    
    def play_pause(self):
        if self.buttons['play'].isChecked():
            self.buttons['play'].setText('Pause')
            self.timer.start()
        else:
            self.buttons['play'].setText('Play')
            self.timer.stop()
    
    def toggle_fullscreen(self):
        if self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showFullScreen()

    def spacebar(self):
        self.buttons['play'].toggle()
        self.play_pause()

    def next_frame(self):
        # If playing, stop playing.
        if self.buttons['play'].isChecked():
            self.buttons['play'].toggle()
            self.play_pause()
        # Move to next frame.
        self.sliders['frame'].setValue(self.sliders['frame'].value()+1)

    def previous_frame(self):
        # If playing, stop playing.
        if self.buttons['play'].isChecked():
            self.buttons['play'].toggle()
            self.play_pause()
        # Move to previous frame.
        self.sliders['frame'].setValue(self.sliders['frame'].value()-1)
    
    def redraw(self):
        value = self.sliders['frame'].value()
        t     = value/self.track.fps
        self.clock.setText(f'Time {t//60:.0f}:{t%60:05.2f} / Frame {value}')
        if value==self.track.current_frame()+1:
            self.track.read_frame()
        else:
            self.track.read_frame(value)
        if self.checkboxes['Show Contours'].isChecked() or \
                self.tab_widget.currentIndex()==0:
            self.track.frame.subtract_background(  
                   self.checkboxes['Subtract Background'].isChecked(), 
                   self.checkboxes['Subtract Secondary Background'].isChecked() )
            if self.checkboxes['Apply Tank Mask'].isChecked():
                self.track.frame.apply_mask()
            s = { k:v.value() for k,v in self.tunables.items() }
            if s['n_blur']>0:
                self.track.frame.blur(s['n_blur'])
            if self.tab_widget.currentIndex()==0 and \
                    not self.checkboxes['Threshold'].isChecked():
                # Stash the frame to allow viewing contours over non-thresholded frame.
                self.track.bgr[:,:,:] = self.track.frame.i8[:,:,None]
            self.track.frame.threshold(s['block_size'], s['offset'])
            if self.checkboxes['Show Contours'].isChecked():
                self.track.frame.detect_contours()
                self.track.frame.analyze_contours( self.track.n_tracks, 
                             s['min_area'], s['max_area'], s['max_aspect'])
            if self.tab_widget.currentIndex()==0 and \
                    self.checkboxes['Threshold'].isChecked():
                self.track.bgr[:,:,:] = self.track.frame.i8[:,:,None]
            if self.checkboxes['Subtract Background'].isChecked():
                np.subtract(255, self.track.bgr, out=self.track.bgr)
        
        track_length = int(self.track.fps*self.tunables['Track length (s)'].value()) if \
                         self.checkboxes['Show Track'].isChecked() else 0
        b = self.track.draw( value, track_length, 
                    show_fish=self.checkboxes['Show Fish'].isChecked(), 
                    show_extra=self.checkboxes['Show Extra Objects'].isChecked(),
                    show_contours=self.checkboxes['Show Contours'].isChecked() )
        
        if b and self.sliders['alpha'].value()>0:
            alpha = self.sliders['alpha'].value()/100.
            mask = np.any(self.track.overlay>0,axis=2)
            self.track.bgr[mask] = ( (1-alpha)*self.track.bgr[mask] + \
                                     alpha*self.track.overlay[mask] ).astype(np.uint8)
        
        # Convert from openCV image matrix to pyqtgraph image matrix then show on screen.
        self.track.bgr = np.swapaxes(self.track.bgr, 0, 1) # Swap rows and columns.
        self.track.bgr = np.flip(self.track.bgr, 2) # Flip color channel order.
        self.video.setImage(self.track.bgr)
    
    def reset_tunables(self):
        for k in self.tunables.keys():
            if k in ['Track length (s)', 'Read one frame in']:
                continue
            k2 = 'bkgSub_options.'+k if k in ['secondary_factor', \
                                              'contrast_factor'] else k
            self.tunables[k].setValue(self.track.settings[k2])
        self.update_bkgSub()
        
    def update_bkgSub(self):
        self.track.frame.contrast_factor = self.tunables['contrast_factor'].value()
        self.track.frame.bkg2 = self.track.frame.bkg2_ * \
                                    self.tunables['secondary_factor'].value() * \
                                    self.tunables['contrast_factor'].value()
#        self.redraw()
    

if __name__ == '__main__':
    
##    try:
    app = QtWidgets.QApplication(sys.argv)
    input_dir = sys.argv[1] if len(sys.argv)>1 else None
    wdg = MainWindow(input_dir)
    wdg.show()
    sys.exit(app.exec_())
##    except:
##        sys.exit(1)

