import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
try:
    from PyQt5.QtGui import QFileDialog
except:
    from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from .gui_classes import *


class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, input_dir, parent=None):
        super().__init__(parent)
        
        if input_dir is None:
            self.choose_input_dir()
        else:
            self.input_dir = input_dir
        self.track   = Track(self.input_dir)
        self.video   = Video()
        self.video.getImageItem().mouseClickEvent = self.video_click
        self.drag    = []
        self.tabs    = SidePanel()
        self.active  = []
        self.history = History()
        self.config  = Config()
        
        self.timer   = QtCore.QTimer(self)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.timeout)
        
        self.keys_down = { k:False for k in [Qt.Key_P, Qt.Key_O, Qt.Key_I, Qt.Key_D]  }
        
        #--------------------
        # Widgets.
        
        # Create widget to show current frame number.
        self.clock = QtWidgets.QLabel()

        # Create spinboxes (editable values).
        self.tunables = {}
        spins  = [ 'n_blur', 'block_size', 'threshold_offset', 'min_area', 'max_area', 
                   'ideal_area', 'Read one frame in' ]
        dspins = [ 'max_aspect', 'ideal_aspect', 'contrast_factor', 
                   'secondary_factor', 'Track length (s)', 'Suspicious Displacement (px)' ]
        # Map settings names to tunables names.
        self.setting2tunable = { 'bkg.secondary_factor':'secondary_factor', 
                                 'bkg.contrast_factor':'contrast_factor' }
        for k in spins+dspins:
            self.tunables[k] = QtWidgets.QSpinBox() if k in spins \
                                   else QtWidgets.QDoubleSpinBox()
            self.tunables[k].setRange(0,1000)
        self.tunables['Read one frame in'].setRange(-100,100)
        def suspicious_displacement(value):
            self.track.bad_displacement = value
            self.track.locate_bad_frames()
        self.tunables['Suspicious Displacement (px)'].valueChanged.connect(suspicious_displacement)
        self.tunables['Suspicious Displacement (px)'].setValue(50)
        self.reset_tunables()
        
        # Create checkboxes.
        self.checkboxes = {}
        for k in [ 'Subtract Background', 'Subtract Secondary Background', 'Apply Tank Mask', 
                   'Threshold', 'Show Contours', 'Show Fish', 'Show Extra Objects', 
                   'Show Track', 'Show Tank', 'Show Suspicious Displacement' ]:
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
        
        # Connect widgets to appropriate actions.
        self.tabs.currentChanged.connect(self.redraw)
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
        
        # Create fish legends as a layouts of checkable QPushButton objects. Later on the legends'
        # buttons are made to reflect the current fish selection stored in self.active.
        self.legends = []
        def create_legend():
            legend = []
            grid   = QtWidgets.QGridLayout()
            for i in range(self.track.n_tracks):
                item = QtWidgets.QPushButton()
                item.setText(f'  fish {i+1}' if i<self.track.n_ind else f'  object {i+1}')
                pixmap = QtGui.QPixmap(10,10)
                pixmap.fill(QtGui.QColor(*get_color(i)[::-1]))
                item.setIcon(QtGui.QIcon(pixmap))
                item.setCheckable(True)
                item.clicked.connect(lambda _,j=i: self.select_fish(j))
                legend.append(item)        
                n = int(np.ceil(self.track.n_tracks/2))
                grid.addWidget(item, i%n, i//n, Qt.AlignTop)
            self.legends.append(legend)
            return grid
        
        # Create a widget with a name and spinbox side-by-side.
        def create_spinbox_row(label):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            row.addWidget(self.tunables[label])
            return row
        
        # Set up 'Tune' tab.
        tab = self.tabs.tune
        for k in [ 'Subtract Background', 'Subtract Secondary Background', 
                   'Apply Tank Mask', 'Threshold', 'Show Contours', 'Show Tank' ]:
            tab.addWidget(self.checkboxes[k])
        for k in [ 'n_blur', 'block_size', 'threshold_offset', 'min_area', 'max_area', 
                   'ideal_area', 'max_aspect', 'ideal_aspect', 
                   'contrast_factor', 'secondary_factor' ]:
            tab.addLayout(create_spinbox_row(k))
        tab.addWidget(self.buttons['reset settings'])
        tab.addStretch(1)
        
        # Set up 'View' tab.
        tab = self.tabs.view
        tab.addLayout(create_legend())
        tab.addWidget(QtWidgets.QLabel(' '))
        for k in [ 'Show Fish', 'Show Extra Objects', 'Show Track' ]:
            tab.addWidget(self.checkboxes[k])
        tab.addWidget(QtWidgets.QLabel(' '))
        tab.addLayout(create_spinbox_row('Track length (s)'))
        tab.addLayout(create_spinbox_row('Read one frame in'))
        tab.addWidget(QtWidgets.QLabel(' '))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Overlay transparency'))
        row.addWidget(self.sliders['alpha'])
        tab.addLayout(row)
        tab.addStretch(1)
        
        # Set up 'Fix' tab.
        tab = self.tabs.fix
        tab.addLayout(create_legend())
        tab.addLayout(create_spinbox_row('Suspicious Displacement (px)'))
        tab.addWidget(self.checkboxes['Show Suspicious Displacement'])
        tab.addWidget(self.history)
#        tab.addStretch(1)
        
        #--------------------
        # Window layout.
        
        # Video & right-side panel.
        layout1 = QtWidgets.QHBoxLayout()
        layout1.addWidget(self.video,2)
        layout1.addWidget(self.tabs)
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
        # Menu bar and shortcuts.
        
        menubar = self.menuBar()
        menus = { 'File': [ ('Open...', self.open, None), 
                            ('Reload', self.reload, 'f5'), 
                            ('Save', self.save, 'ctrl+s'), 
                            ('Exit', self.close, 'ctrl+q') ],
                  'Edit': [ ('Undo Fix', self.undo, 'ctrl+z'), 
                            ('Preferences...', self.update_config, None) ],
                  'View': [ ('Play/Pause', self.spacebar, 'space'),
                            ('Next Frame', lambda:self.next_frame(1), 'right'),
                            ('Previous Frame', self.previous_frame, 'left'),
                            ('Skip Forward', self.skip_forward, 'ctrl+right'),
                            ('Skip Backward', self.skip_backward, 'ctrl+left'),
                            ('Next Issue', self.next_issue, 'shift+right'),
                            ('Previous Issue', self.previous_issue, 'shift+left'),
                            ('Toggle Full Screen', self.toggle_fullscreen, 'f') ], 
                            }
        for menu_name,menu_items in menus.items():
            menu = menubar.addMenu(menu_name)
            for name,function,shortcut in menu_items:
                action = QtWidgets.QAction(name,self)
                action.triggered.connect(function)
                if not shortcut is None:
                    action.setShortcut(QtGui.QKeySequence(shortcut))
                menu.addAction(action)
        
        #--------------------
        # Starting options.
        
        self.tabs.setCurrentWidget(self.tabs.view.parent()) # Start in 'View' tab.
        i = self.track.frames[0] if len(self.track.frames)>0 else 0
        self.sliders['frame'].setValue(i) # Go to beginning of video.
        self.sliders['alpha'].setValue(50) # Go to beginning of video.
        self.checkboxes['Show Fish'].setChecked(True)
        self.checkboxes['Show Track'].setChecked(True)
        self.tunables['Track length (s)'].setValue(10)
        self.tunables['Read one frame in'].setValue(5)
        self.reset_tunables()
        self.load()
    
    
    def choose_input_dir(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', 
                                  '.', QFileDialog.ShowDirsOnly)
        if input_dir:
            self.input_dir = input_dir

    def open(self):
        self.choose_input_dir()
        self.reload()

    def update_config(self, _, dialog=True):
        if dialog:
            self.config.dialog()
        self.track.__dict__.update(self.config.items())
        self.redraw()

    def reload(self):
        self.track = Track(self.input_dir)
        self.track.bad_displacement = self.tunables['Suspicious Displacement (px)'].value()
        self.track.locate_bad_frames()
        self.redraw()
        self.window().setWindowTitle(self.input_dir)

    def save(self, save_tracks=True):
        # Save settings.
        D = self.track.settings.copy()
        for k in D.keys():
            k2 = self.setting2tunable.get(k,k)
            if k2 in self.tunables.keys():
                D[k] = self.tunables[k2].value()
        utils.save_txt(self.track.join('gui_settings.txt'), D)
        # Save fix history and fixed tracks.
        D = { 'fixes':self.history.fixes }
        if save_tracks:
            D['tracks'] = self.track.tracks
        utils.save_pik(self.track.join('gui_fixes.pik'), D)
        utils.save_txt(self.track.join('gui_fixes.txt'), dict(enumerate(self.history.fixes)))

    def load(self):
        try:
            # Load settings.
            D = utils.load_txt(self.track.join('gui_settings.txt'))
            for k,v in D.items():
                k2 = self.setting2tunable.get(k,k)
                t = self.tunables.get(k2)
                if not t is None:
                    if isinstance(t,QtWidgets.QSpinBox):
                        v = int(v)
                    elif isinstance(t,QtWidgets.QDoubleSpinBox):
                        v = float(v)
                    self.tunables[k2].setValue(v)
            self.update_bkgSub()
            # Load fix history and fixed tracks.
            D = utils.load_pik(self.track.join('gui_fixes.pik'))
            self.history.fixes = D['fixes']
            self.history.sync()
            self.apply_fixes()
            return True
        except:
            print(sys.exc_info())
            return False
    
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

    def next_frame(self, skip=1):
        # If playing, stop playing.
        if self.buttons['play'].isChecked():
            self.buttons['play'].toggle()
            self.play_pause()
        # If holding "d", delete position and orientation of selected fish from
        # current to next frame (including frames in between if abs(skip)>1).
        i0 = self.sliders['frame'].value()
        if self.keys_down[Qt.Key_D]:
            for j in self.active:
                for i in range(abs(skip)):
                    self.track.fix('delete_position', i0+np.sign(skip)*i-1, j, history=self.history)
        # Move to next frame.
        self.sliders['frame'].setValue(i0+skip)

    def previous_frame(self):
        self.next_frame(-1)

    def skip_forward(self):
        self.next_frame(self.tunables['Read one frame in'].value())

    def skip_backward(self):
        self.next_frame(-self.tunables['Read one frame in'].value())

    def next_issue(self):
        i = self.sliders['frame'].value()-1
        B = self.track.bad_frames[i:]>0
        I = np.nonzero(B[1:]&~B[:-1])[0] # Start indices of bad frame ranges after i.
        if len(I)>0:
            return self.next_frame(I[0]+1)
        i2 = min(self.sliders['frame'].maximum()-1, len(self.track.bad_frames))
        return self.next_frame(i2-i)

    def previous_issue(self):
        i = self.sliders['frame'].value()-1
        B = self.track.bad_frames[:i]>0
        I = np.nonzero(B[1:]&~B[:-1])[0] # Start indices of bad frame ranges before i.
#        # If inside a group of bad frames, jump to the beginning of the previous 
#        # distinct issue rather than jumping to the beginning of the current issue.
#        if i<len(self.track.bad_frames) and self.track.bad_frames[i]>0 and B[-1] and len(I)>1:
#            return self.next_frame(I[-2]-i+1)
        # If current frame is past the last tracked frame, jump to the first untracked frame.
        n = len(self.track.bad_frames)
        if i>n:
            return self.next_frame(n-i)
        # Jump to the last bad frame following a good frame. If currently in the middle of a
        # group of bad frame, this jumps to the beginning of beginning of this ongoing issue.
        if len(I)>0:
            return self.next_frame(I[-1]-i+1)
        return self.next_frame(-i)

    def redraw(self):
        value = self.sliders['frame'].value()
        # Change slider handle color to indicate missing track data.
        b = self.track.bad_frames[value-1] if value<=len(self.track.bad_frames) else 3 # 3=not tracked yet
        if b==0:
            self.sliders['frame'].setStyleSheet(None)
        else:
            c  =  ['red','orange','yellow'][b-1]
            ss = 'QSlider::handle:horizontal {background-color:%s;}'%c
            self.sliders['frame'].setStyleSheet(ss)
        # Update the time and frame number.
        t     = (value-1)/self.track.fps
        self.clock.setText(f'Time {t//60:.0f}:{t%60:05.2f} / Frame {value}')
        # Draw frame.
        if value==self.track.current_frame()+1:
            self.track.read_frame()
        else:
            self.track.read_frame(value)
        if self.checkboxes['Show Contours'].isChecked() or \
                self.tabs.currentIndex()==0:
            self.track.frame.subtract_background(  
                   self.checkboxes['Subtract Background'].isChecked(), 
                   self.checkboxes['Subtract Secondary Background'].isChecked() )
            if self.checkboxes['Apply Tank Mask'].isChecked():
                self.track.frame.apply_mask()
            s = { k:v.value() for k,v in self.tunables.items() }
            if s['n_blur']>0:
                n_blur = s['n_blur'] + s['n_blur']%2 - 1
                self.track.frame.blur(n_blur)
            if self.tabs.currentIndex()==0 and \
                    not self.checkboxes['Threshold'].isChecked():
                # Stash the frame to allow viewing contours over non-thresholded frame.
                self.track.bgr[:,:,:] = self.track.frame.i8[:,:,None]
            block_size = s['block_size'] + s['block_size']%2 - 1
            self.track.frame.threshold(block_size, s['threshold_offset'])
            if self.checkboxes['Show Contours'].isChecked():
                self.track.frame.detect_contours()
                self.track.frame.analyze_contours( self.track.n_tracks, 
                             s['min_area'], s['max_area'], s['max_aspect'])
            if self.tabs.currentIndex()==0 and \
                    self.checkboxes['Threshold'].isChecked():
                self.track.bgr[:,:,:] = self.track.frame.i8[:,:,None]
            if self.tabs.currentIndex()==0 and \
                    self.checkboxes['Subtract Background'].isChecked():
                np.subtract(255, self.track.bgr, out=self.track.bgr)
        
        track_length = int(self.track.fps*self.tunables['Track length (s)'].value()) if \
                         self.checkboxes['Show Track'].isChecked() else 0
        b = self.track.draw( value, track_length, 
                    show_fish=self.checkboxes['Show Fish'].isChecked(), 
                    show_extra=self.checkboxes['Show Extra Objects'].isChecked(),
                    show_contours=self.checkboxes['Show Contours'].isChecked(), 
                    show_tank=self.checkboxes['Show Tank'].isChecked() )
        if self.checkboxes['Show Suspicious Displacement'].isChecked():
            self.track.draw_scale_bar(int(self.tunables['Suspicious Displacement (px)'].value()))
            b = True
        
        if b and self.sliders['alpha'].value()>0:
            alpha = self.sliders['alpha'].value()/100.
            mask = np.any(self.track.overlay>0,axis=2)
            self.track.bgr[mask] = ( (1-alpha)*self.track.bgr[mask] + \
                                     alpha*self.track.overlay[mask] ).astype(np.uint8)
        
        self.video.setImage_(self.track.bgr) # ! Modifies track.bgr (converts to pyqtgraph image).
    
    def reset_tunables(self):
        for k in self.track.settings.keys():
            k2 = self.setting2tunable.get(k,k)
            if k2 in self.tunables.keys():
                self.tunables[k2].setValue(self.track.settings[k])
        self.update_bkgSub()
        
    def update_bkgSub(self):
        self.track.frame.contrast_factor = self.tunables['contrast_factor'].value()
        self.track.frame.bkg2 = self.track.frame.bkg2_ * \
                                    self.tunables['secondary_factor'].value() * \
                                    self.tunables['contrast_factor'].value()

    def update_fish_selection(self):
        for legend in self.legends:
            for j,item in enumerate(legend):
                item.setChecked(j in self.active)

    def select_fish(self, i):
        if i is None:
            self.active = self.active[1:]
        elif i in self.active:
            self.active.remove(i)
        else:
            self.active.append(i)
        while len(self.active)>2:
            self.active.pop(0)
        self.update_fish_selection()

    def video_click(self,event):
        if event.button()!=QtCore.Qt.LeftButton:
            event.ignore()
            return
        if len(self.active)==1:
            if self.keys_down[Qt.Key_P]:
                j = self.active[-1]
                self.track.fix('position', None, j, np.around(event.pos(),1), history=self.history)
                self.redraw()
                return
            if self.keys_down[Qt.Key_O]:
                i = self.track.current_frame()-1
                j = self.active[-1]
                # New direction goes from current position to clicked position.
                u = np.array(event.pos())-self.track.tracks[i,j,:2]
                a = np.around(np.arctan2(u[1], u[0]), 2)
                self.track.fix('orientation', i, j, a, history=self.history)
                self.redraw()
                return
        j = self.track.select_fish(event.pos())
        self.select_fish(j)
    
    def keyPressEvent(self,event):
        # List of Qt key variables: https://doc.qt.io/qt-5/qt.html#Key-enum
        if event.key() in [Qt.Key_P,Qt.Key_O,Qt.Key_I,Qt.Key_D]:
            self.keys_down[event.key()] = True
            self.update_fish_selection()
            self.active = self.active[-1:]
            if len(self.active)==0:
                return
            if self.keys_down[Qt.Key_I] and self.keys_down[Qt.Key_P]:
                self.track.fix('interpolate_position', None, self.active[0], history=self.history)
                self.redraw()
            if self.keys_down[Qt.Key_I] and self.keys_down[Qt.Key_O]:
                self.track.fix('interpolate_orientation', None, self.active[0], history=self.history)
                self.redraw()
            if self.keys_down[Qt.Key_D] and self.keys_down[Qt.Key_P]:
                self.track.fix('delete_position', None, self.active[0], history=self.history)
                self.redraw()
            if self.keys_down[Qt.Key_D] and self.keys_down[Qt.Key_O]:
                self.track.fix('delete_orientation', None, self.active[0], history=self.history)
                self.redraw()
            return
        if event.key()==Qt.Key_S:
            if len(self.active)==2:
                self.track.fix('swap', None, *self.active, history=self.history)
                self.redraw()
            return
        if event.key()==Qt.Key_R:
            while len(self.active)>1:
                self.active.pop()
            self.update_fish_selection()
            j = self.active[0]
            self.track.fix('reverse', None, j, history=self.history)
            self.redraw()
            return
        if event.key()==Qt.Key_Delete:
            if self.tabs.currentWidget()==self.tabs.fix.parent():
                for item in self.history.selectedItems():
                    self.history.remove_fix(self.history.row(item))
                self.apply_fixes()
            return
        if Qt.Key_0<=event.key()<=Qt.Key_9:
            j = (event.key()-Qt.Key_1)%10
            if j<self.track.n_tracks:
                self.select_fish(j)

    def keyReleaseEvent(self,event):
        if event.key() in self.keys_down.keys():
            self.keys_down[event.key()] = False
    
    def undo(self):
        if self.tabs.currentWidget()==self.tabs.fix.parent() and len(self.history.fixes)>0:
            self.history.remove_fix()
#            self.history.sync()
            self.apply_fixes()
    
    def apply_fixes(self):
        self.track.load_tracks()
        for fix in self.history.fixes:
            self.track.fix(*fix, recompute_bad_frames=False)
        self.track.locate_bad_frames()
        self.redraw()
    
#    def video_drag(self,event):
#        print('drag event')
#        if event.isStart():
#            print('drag event start')
#            # We are already one step into the drag.
#            # Find the point(s) at the mouse cursor when the button was first 
#            # pressed:
#            p1 = event.buttonDownPos()
#            i  = self.track.current_frame() # frame id
#            j  = self.track.select_fish(p1,i) # fish id
#            self.drag = (i,j,p1)
#            if j is None:
#                event.ignore()
#                return
#        elif event.isFinish():
#            print('drag event finish')
#            p2 = event.buttonDownPos()
#            i,j,p1 = self.drag
#            self.track.tracks[i,j,:2] += p2-p1
#            self.redraw()
#            return
#        else:
#            event.ignore()
#            return

#========================================================

def start_gui(input_dir=None):
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWindow(input_dir)
    gui.show()
    return app.exec_()    

