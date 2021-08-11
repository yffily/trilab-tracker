import numpy as np
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

def compute_kinematics(trial, wall_distance=False):
    tank,R_cm,n_ind,data,time = map(trial.get,['tank','R_cm','n_ind','data','time'])
    center      = np.array([tank['xc'],tank['yc']])
    px2cm       = R_cm/tank['R']
    n           = n_ind
    pos         = data[:,:n,:3].copy() # discard extra objects
    pos[:,:,:2] = (pos[:,:,:2]-center[None,None,:])*px2cm # convert to centimeters
    pos[:,:,1]  = -pos[:,:,1] # flip y axis
    for j in range(n): # unwrap orientations
        I          = ~np.isnan(pos[:,j,2])
        pos[I,j,2] = np.unwrap(pos[I,j,2])
    vel         = np.gradient(pos,time,axis=0)
    acc         = np.gradient(vel,time,axis=0)
    v           = np.hypot(vel[:,:,0],vel[:,:,1])
    d_wall      = R_cm - np.hypot(pos[:,:,0],pos[:,:,1])
# #     dist    = lambda xy: dist2ellipse(*ellipse[1],xy)
#     dist    = lambda xy: cv2.pointPolygonTest(tank['contour'],tuple(xy),True)
#     d_wall  = px2cm * np.apply_along_axis(dist,2,trial['data'][:,:,:2])
    trial.update({ k:v for k,v in locals().items() if k in 
                   ['time', 'pos', 'vel', 'acc', 'd_wall', 'v'] })
    return trial

def compute_cuts(trial,ranges):
#    globals().update(trial)
    # valid array: axis 0 = time, axis 1 = [nan_xy,nan_any,d_wall,v,v_ang,final]
    pos,vel,acc,v = map(trial.get,['pos','vel','acc','v'])
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
    trial.update(valid=valid, valid_fraction=valid_fraction)
    return trial

def apply_cuts(trial):
    for k in 'data','vel','acc','v':
        trial[k][~trial['valid'][:,:,5]] = np.nan
    return trial

default_cut_ranges = dict( v=[0,np.inf], v_ang=[-np.inf,np.inf] )

def preprocess_trial(trial, cut_ranges=None, load_timestamps=True, etho=False):
    if etho:
        raise Exception('Not implemented yet: preprocess_trial for ethovision files.')
    trial.update(utils.load_trial(trial['trial_file'], load_timestamps=load_timestamps))
    trial = compute_kinematics(trial)
    if not cut_ranges is None:
        ranges = deepcopy(default_cut_ranges)
        ranges.update(cut_ranges)
        for k in {'v'} & ranges.keys():
            ranges[k] = [x*trial['R_cm'] for x in ranges[k]]
        trial = compute_cuts(trial, ranges)
        trial = apply_cuts(trial)
    return trial

