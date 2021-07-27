import numpy as np
from . import utils

def compute_kinematics(trial, wall_distance=False):
    center      = np.array([trial['tank']['xc'],trial['tank']['yc']])
    px2cm       = trial['R_cm']/trial['tank']['R']
    n           = trial['n_ind']
    pos         = trial['data'][:,:n,:3].copy() # discard extra objects
    pos[:,:,:2] = (pos[:,:,:2]-center[None,None,:])*px2cm # convert to centimeters
    pos[:,:,1]  = -pos[:,:,1] # flip y axis
    for j in range(n): # unwrap orientations
        I          = ~np.isnan(pos[:,j,2])
        pos[I,j,2] = np.unwrap(pos[I,j,2])
    time        = trial['frame_list']/trial['fps']
    vel         = np.gradient(pos,time,axis=0)
    acc         = np.gradient(vel,time,axis=0)
    v           = np.hypot(vel[:,:,0],vel[:,:,1])
    d_wall      = trial['R_cm'] - np.hypot(pos[:,:,0],pos[:,:,1])
# #     dist    = lambda xy: dist2ellipse(*ellipse[1],xy)
#     dist    = lambda xy: cv2.pointPolygonTest(trial['tank']['contour'],tuple(xy),True)
#     d_wall  = px2cm * np.apply_along_axis(dist,2,trial['data'][:,:,:2])
    trial.update({ k:v for k,v in locals().items() if k in 
                   ['time', 'pos', 'vel', 'acc', 'd_wall', 'v'] })
    return trial

def compute_cuts(trial,ranges):
    globals().update(trial)
    # valid array: axis 0 = time, axis 1 = [nan_xy,nan_any,d_wall,v,v_ang,final]
    valid  = np.full(pos.shape[:2]+(6,),np.True_,dtype=np.bool_)
    valid[:,:,0] = np.logical_not(np.any(np.isnan(pos),axis=2))
    valid[:,:,1] = np.logical_not(np.any(np.isnan(vel),axis=2))
    valid[:,:,2] = np.logical_not(np.any(np.isnan(acc),axis=2))
    valid[:,:,3] = np.logical_and(v>=ranges['v'][0],v<=ranges['v'][1])
    valid[:,:,4] = np.logical_and(vel[:,:,2]>=ranges['v_ang'][0],vel[:,:,2]<=ranges['v_ang'][1])
    valid[:,:,5] = np.all(valid[:,:,:5],axis=2)
    n_total = valid.shape[0]*valid.shape[1]
    n_valid = np.count_nonzero(valid,axis=(0,1))
    valid_fraction = { 'nan_pos' : n_valid[0]/n_total, 
                       'nan_vel' : n_valid[1]/n_total, 
                       'nan_acc' : n_valid[2]/n_total, 
                       'v'       : n_valid[3]/n_valid[1], 
                       'v_ang'   : n_valid[4]/n_valid[1], 
                       'final'   : n_valid[5]/n_total     }
    trial.update({'valid':valid, 'valid_fraction':valid_fraction})
    return trial

def apply_cuts(trial):
    globals().update(trial)
    for k in 'data','vel','acc','v':
        trial[k][~valid[:,:,6]] = np.nan
    return trial

default_cut_ranges = dict( v=[0,np.inf], v_ang=[-np.inf,np.inf] )

def preprocess_trial(trial, cut_ranges=None):
    trial.update(utils.load_trial(trial['trial_file']))
    trial = compute_kinematics(trial)
    if not cut_ranges is None:
        ranges = copy(default_cut_ranges)
        ranges.update(cut_ranges)
        trial = compute_cuts(trial, ranges)
        trial = apply_cuts(trial)
    return trial

