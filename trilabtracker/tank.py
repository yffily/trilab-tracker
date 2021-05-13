from .utils import *

point_radius = 8

class Tank:

    def __init__(self):
        self.points  = []
        self.xc      = 0
        self.yc      = 0
        self.R       = 0
        self.raw_frame = None
        self.frame   = None
        self.wname   = None
        self.thresh  = 8
        self.dilate  = 0
        self.contour = []


    def save_img(self, fname):
        self.frame = self.raw_frame
        self.draw_outline(self.frame)
        cv2.imwrite(fname,self.frame)

    
    def to_dict(self):
        keys = ['points', 'contour', 'xc', 'yc', 'R', 'thresh', 'dilate']
        return { k:self.__dict__[k] for k in keys }


    def save(self, fname):
        save_pik(fname, self.to_dict())


    def load(self, fname):
        try:
            self.__dict__.update(load_pik(fname))
            logging.info(parindent+f'Tank loaded from {fname}')
            return True
        except:
            return False


    #########################
    # Tank locator GUI
    #########################
    
    
    def locate_from_video(self, fvideo, i_frame=None):
        # Open video.
        cap = cv2.VideoCapture(fvideo)
        if not cap.isOpened():
            self.interrupt(f'Could not open {fvideo}.')
        # Pick a frame.
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if i_frame == None:
            i_frame = n_frames//2
        elif i_frame < 0:
            i_frame = n_frames - i_frame
        # Open frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
        ret, frame = cap.read()
        if not ret:
            self.interrupt(f'Could not open frame {i_frame}.')
        # Close video.
        cap.release()
        # Locate the tank.
        return self.locate(frame, fvideo)
        

    def locate(self, img, wname='Locate tank'):
        self.raw_frame = img.copy()
        if len(self.raw_frame.shape)==2:
            self.raw_frame = np.dstack([self.raw_frame,self.raw_frame,self.raw_frame])
        self.frame = self.raw_frame.copy()
        # Wait for user to click on the edge three times.
        self.wname = create_named_window(wname)
        self.point_dragged = None
        cv2.setMouseCallback(self.wname, self.process_mouse)
        cv2.imshow(self.wname,self.frame)
        while True:
            k = wait_on_named_window(self.wname,2000)
            if k == -2:
                self.points  = []
                self.contour = []
                self.raw_frame = None
                self.frame   = None
                self.wname   = None
                wait_on_named_window(self.wname,2000)
                return self.interrupt('Tank detection interrupted.')
            elif k == space_key:
                if len(self.points)>0: # Accept the current shape.
                    return self.interrupt('Tank detection complete.', True)
                else: # Create a first point in the middle of the image.
                    h,w = self.raw_frame.shape[:2]
                    self.points = [(int(w/2),int(h/2))]
                    self.redraw_points()
            elif len(self.points)==1 and k in [plus_key,minus_key,zero_key,lbrack_key,rbrack_key]:
                if k == plus_key:
                    self.thresh = min(self.thresh+1,255)
                elif k == minus_key:
                    self.thresh = max(self.thresh-1,0)
                elif k == zero_key:
                    self.thresh = 8
                    self.dilate = 0
                elif k == lbrack_key:
                    self.dilate = self.dilate-1
                elif k == rbrack_key:
                    self.dilate = self.dilate+1
                self.redraw_points()
                cv2.putText(self.frame, f'threshold={self.thresh}', (20,20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                cv2.putText(self.frame, f'dilate={self.dilate}', (20,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                cv2.imshow(self.wname,self.frame)

    
    # Draw points and tank (used during tank detection).
    def redraw_points(self):
        self.frame = self.raw_frame.copy()
        for p in self.points:
            cv2.circle(self.frame, (int(p[0]), int(p[1])), point_radius, (0,255,0), -1)
        if len(self.points)==1:
            self.calculate_color_region()
            if len(self.contour)>0:
                cv2.drawContours(self.frame, [self.contour], -1, (0,255,0), 1)
        if len(self.points)==2:
            cv2.rectangle(self.frame, self.points[0], self.points[1], (0,255,0), 1)
        if len(self.points)==3:
            self.calculate_circle()
            cv2.circle(self.frame, (int(self.xc),int(self.yc)), int(self.R), (0,255,0), 1)
        if len(self.points)>3:
            points = np.array(self.points).reshape((-1,1,2))
            cv2.polylines(self.frame, [points], True, (0,255,0), 1)
        cv2.imshow(self.wname,self.frame)
    
    
    # Draw tank outline on external image (used during tracking).
    def draw_outline(self, img, color=(0,255,0), thickness=1):
        if len(self.points)==1:
            cv2.drawContours(img, [self.contour], -1, color, thickness)
            return
        if len(self.points)==2:
            cv2.rectangle(img, self.points[0], self.points[1], color, thickness)
            return
        if len(self.points)==3:
            cv2.circle(img, (int(self.xc),int(self.yc)), int(self.R), color, thickness)
            return
        if len(self.points)>3:
            points = np.array(self.points).reshape((-1,1,2))
            cv2.polylines(img, [points], True, color, thickness)
            return


    def process_mouse(self, event, x, y, flags, param):
        # One point looks for a contiguous color region, two defines a rectangle, 
        # three is a circle, more is a polygon.
        
        # Middle click deletes a point.
        if event == cv2.EVENT_MBUTTONDOWN:
            # Check if we clicked on an existing point.
            for i,p in enumerate(self.points):
                if np.sqrt((x-p[0])**2+(y-p[1])**2)<point_radius:
                    del self.points[i]
                    self.redraw_points()
                    return

        # Left click adds a point.
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if we grabbed an existing point.
            for i,p in enumerate(self.points):
                if np.sqrt((x-p[0])**2+(y-p[1])**2)<point_radius:
                    self.point_dragged = i
                    return
            # Create a new point.
            self.points.append((x,y))
            self.redraw_points()
            return
        
        # Click and drag moves a point.
        if event == cv2.EVENT_MOUSEMOVE and self.point_dragged!=None:
            self.points[self.point_dragged] = (x,y)
            self.redraw_points()
            return
        if event == cv2.EVENT_LBUTTONUP and self.point_dragged!=None:
            self.points[self.point_dragged] = (x,y)
            self.redraw_points()
            self.point_dragged = None
            return


    def calculate_color_region(self,tresh=None):
        x,y       = self.points[0]
        img       = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)
        ref       = img[y,x].astype(int)
        
        mask = np.zeros((img.shape[0]+2,img.shape[1]+2),dtype=np.uint8)
        cv2.floodFill(img, mask, seedPoint=(x,y), newVal=255, 
                      loDiff=self.thresh, upDiff=self.thresh, 
                      flags=cv2.FLOODFILL_MASK_ONLY)
        mask = 255*mask[1:-1,1:-1]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)[-2:]
        self.contour = contours[-1] if len(contours)>0 else []
        
        if len(self.contour)>0 and self.dilate!=0:
#            kernel = np.ones((abs(self.dilate),)*2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(abs(self.dilate),)*2)
            mask.fill(0)
            cv2.drawContours(mask,[self.contour],-1,255,-1)
            if self.dilate>0:
                mask = cv2.dilate(mask,kernel)
            if self.dilate<0:
                mask = cv2.erode(mask,kernel)
            
#        if self.dilate>0:
##            kernel = np.ones((abs(self.dilate),)*2)
#            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.dilate,)*2)
#            mask = cv2.dilate(mask,kernel)
#        if self.dilate<0:
#            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-self.dilate,)*2)
#            mask = cv2.erode(mask,kernel)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_NONE)[-2:]
        self.contour = contours[-1] if len(contours)>0 else []
    

    def calculate_circle(self):
        points    = np.array(self.points)
        midpoints = (points[1:]+points[:-1])/2
        vectors   = points[1:]-points[:-1]
        m         = -vectors[:,0]/vectors[:,1] # -1/slopes
        b         = midpoints[:,1]-m*midpoints[:,0] # intercepts
        self.xc = (b[1]-b[0])/(m[0]-m[1])
        self.yc = m[0]*self.xc + b[0]
        self.R = np.sqrt( (self.xc-points[0,0])**2 + 
                             (self.yc-points[0,1])**2 )
    

    def interrupt(self,msg=None,retval=False):
        cv2.destroyAllWindows()
        if msg!=None:
            logging.info(parindent+f'{msg}')
        return retval


    def create_mask(self,shape):
        if len(self.points)==0:
            return np.full(shape=shape[:2], fill_value=255, dtype=np.uint8)
        mask = np.zeros(shape=shape[:2],dtype=np.uint8)
        if len(self.points)==1:
            cv2.drawContours(mask, [self.contour], -1, 255, -1)
        elif len(self.points)==2:
            cv2.rectangle(mask, self.points[0], self.points[1], 255, -1)
        elif len(self.points)==3:
            x,y,R = map(int, (self.xc,self.yc,self.R) )
            cv2.circle(mask, (x,y), R, 255, thickness=-1)
        else: # len(self.point)>3
            points = np.array(self.points).reshape((1,-1,2))
            cv2.fillPoly(mask, points, 255, 1)
        return mask

