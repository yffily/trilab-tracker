import numpy as np
from scipy.signal import savgol_filter
#import warnings
from copy import deepcopy
from . import utils

# # Distance between a point and an ellipse (origin=ellipse center).
# def dist2ellipse(semi_major, semi_minor, xy):
#     px,py = np.absolute(xy)
#     tx,ty = 0.707,0.707
#     a = semi_major
#     b = semi_minor
#     for x in range(0, 3):
#         x   = a * tx
#         y   = b * ty
#         ex  = (a*a - b*b) * tx**3 / a
#         ey  = (b*b - a*a) * ty**3 / b
#         rx  = x - ex
#         ry  = y - ey
#         qx  = px - ex
#         qy  = py - ey
#         r   = np.hypot(ry, rx)
#         q   = np.hypot(qy, qx)
#         tx  = min(1, max(0, (qx * r / q + ex) / a))
#         ty  = min(1, max(0, (qy * r / q + ey) / b))
#         t   = np.hypot(ty, tx)
#         tx /= t 
#         ty /= t 
#     return (np.copysign(a * tx, xy[0]), np.copysign(b * ty, xy[1]))    

def compute_kinematics(trial, orientation='body', wall_distance=False):
    tank,R_cm,n_ind,data,time = map(trial.get,['tank','R_cm','n_ind','data','time'])
    center      = np.array([tank['xc'],tank['yc']])
    px2cm       = R_cm/tank['R']
    n           = n_ind
    pos         = data[:,:n,:3].copy() # discard extra objects
    pos[:,:,:2] = (pos[:,:,:2]-center[None,None,:])*px2cm # convert to centimeters
    pos[:,:,1:3]  = -pos[:,:,1:3] # flip y axis, which also flips the body angle
    for j in range(n): # unwrap orientations
        I          = ~np.isnan(pos[:,j,2])
        pos[I,j,2] = np.unwrap(pos[I,j,2])
    vel         = np.gradient(pos,time,axis=0)
    if orientation=='motion':
        pos[:,:,2] = np.arctan2(vel[:,:,1],vel[:,:,0])
        for j in range(n): # unwrap orientations
            I          = ~np.isnan(pos[:,j,2])
            pos[I,j,2] = np.unwrap(pos[I,j,2])
        vel[:,:,2] = np.gradient(pos[:,:,2],time,axis=0)
    acc         = np.gradient(vel,time,axis=0)
    v           = np.hypot(vel[:,:,0],vel[:,:,1])
    trial.update({ k:v for k,v in locals().items() if k in 
                   ['time', 'pos', 'vel', 'acc', 'v'] })
    if wall_distance:
        d_wall = R_cm - np.hypot(pos[:,:,0],pos[:,:,1])
##        d_wall = lambda xy: dist2ellipse(*ellipse[1],xy)
#        d_wall = lambda xy: cv2.pointPolygonTest(tank['contour'],tuple(xy),True)
#        d_wall = px2cm * np.apply_along_axis(dist,2,trial['data'][:,:,:2])
        trial['d_wall'] = d_wall
    return trial

default_cut_ranges = dict( v=[0,np.inf], v_ang=[-np.inf,np.inf] )
def compute_cuts(trial, ranges, buffer_frames=0):
    # valid array: axis 0 = time, axis 1 = [nan_xy,nan_any,d_wall,v,v_ang,final]
    time,pos,vel,acc,v = map(trial.get,['time','pos','vel','acc','v'])
    valid_time   = np.logical_and(time>=ranges['t'][0],time<ranges['t'][1])
    valid        = np.full(pos.shape[:2]+(6,),np.True_,dtype=np.bool_)
    valid[:,:,0] = np.any(np.isfinite(pos),axis=2)
    valid[:,:,1] = np.any(np.isfinite(vel),axis=2)
    valid[:,:,2] = np.any(np.isfinite(acc),axis=2)
    valid[:,:,3] = np.logical_and(v>=ranges['v'][0],v<ranges['v'][1])
    valid[:,:,4] = np.logical_and(vel[:,:,2]>=ranges['v_ang'][0],vel[:,:,2]<ranges['v_ang'][1])
    valid[:,:,5] = np.all(valid[:,:,:5],axis=2)
    for i in range(buffer_frames):
        valid[1:-1] = valid[:-2] & valid[1:-1] & valid[2:]
#    n_total = valid.shape[0]*valid.shape[1]
    n_total = np.count_nonzero(valid_time)*valid.shape[1]
    n_valid = np.count_nonzero(valid&valid_time[:,None,None],axis=(0,1))
    valid_fraction = { 'nan_pos' : n_valid[0]/n_total, 
                       'nan_vel' : n_valid[1]/n_total, 
                       'nan_acc' : n_valid[2]/n_total, 
                       'v'       : n_valid[3]/n_valid[1], 
                       'v_ang'   : n_valid[4]/n_valid[1], 
                       'final'   : n_valid[5]/n_total     }
    trial.update(valid_time=valid_time, valid=valid, valid_fraction=valid_fraction)
    return trial

def apply_cuts(trial):
    for k in ['valid','frame_list','time']:
        trial[k] = trial[k][trial['valid_time']]
    for k in ['data','pos','vel','acc','v','d_wall']:
        if k in trial.keys():
            trial[k] = trial[k][trial['valid_time']]
            trial[k][~trial['valid'][:,:,5]] = np.nan
    return trial

def apply_smoothing(trial, n, method='savgol'):
    for k in ['data','pos','vel','acc','v','d_wall']:
        if k in trial.keys():
            if method=='savgol':
                trial[k] = savgol_filter(trial[k], window_length=n, 
                                         polyorder=min(2,n-1), axis=0)
#            elif method=='moving_avg':
#                kernel = np.ones(n)
            else:
                logging.warning('Unknown smoothing method:', method)
    return trial

def preprocess_trial(trial, load_timestamps=True, orientation='body', 
                     wall_distance=False, cut_ranges=None, rescale_cut_ranges=False, 
                     n_smooth=0, buffer_frames=0, etho=False):
    if etho:
        raise Exception('Not implemented yet: preprocess_trial for ethovision files.')
    trial.update(utils.load_trial(trial['trial_file'], load_timestamps=load_timestamps))
    if n_smooth>0:
        trial = apply_smoothing(trial, n_smooth)
    trial = compute_kinematics(trial, orientation=orientation, wall_distance=wall_distance)
    if not cut_ranges is None:
        ranges = deepcopy(default_cut_ranges)
        ranges.update(cut_ranges)
        if rescale_cut_ranges:
            for k in {'v'} & ranges.keys():
                ranges[k] = [x*trial['R_cm'] for x in ranges[k]]
        trial = compute_cuts(trial, ranges, buffer_frames)
        trial = apply_cuts(trial)
    return trial

