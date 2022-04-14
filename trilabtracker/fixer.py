import numpy as np

## TODO: Use a decorator to log fixes.
#def fix_decorator(fix_name):
#    def wrapper(*args, **kwargs):

# Move fish j to position xy in frame i.
def fix_position(tracks, i, j, xy):
    tracks[i,j,:2] = xy

# Change orientation of fish j to angle a in frame i.
def fix_orientation(tracks, i, j, a):
    tracks[i,j,2] = a

# Reverse direction of fish j from frame i on.
def fix_reverse(tracks, i, j):
    tracks[i:,j,2] = tracks[i:,j,2] + np.pi

# Swap fish j1 and j2 from frame i on.
def fix_swap(tracks, i, j1, j2):
    tracks[i:,[j1,j2]] = tracks[i:,[j2,j1]]

# Fill array X between indices i1 and i2 using linear interpolation.
def interpolate(X, i1, i2, angle=False):
    I  = np.arange(1,i2-i1)
    dx = X[i2]-X[i1]
    if angle:
        dx -= 2*np.pi*np.rint(dx/(2*np.pi))
    for i in range(i1+1,i2):
        X[i] = X[i1] + dx*(i-i1)/(i2-i1)

# Interpolate position of fish j from last known position to current frame (i).
def fix_interpolate_position(tracks, i, j):
    B = np.any(np.isnan(tracks[:i,j,:2]), axis=1)
    if np.any(np.isnan(tracks[i,j,:2])):
        I = np.nonzero(B)[0]
        if len(I)==0 or I[-1]==i-1:
            return False
        interpolate(tracks[:,j,:2], I[-1], i)
        return True
    else:
        I = np.nonzero(~B)[0]
        if len(I)==0 or I[-1]==i-1:
            return False
        interpolate(tracks[:,j,:2], I[-1], i)
        return True

# Interpolate orientation of fish j from last known orientation to current frame (i).
def fix_interpolate_orientation(tracks, i, j):
    B = np.isnan(tracks[:i,j,2])
    if np.isnan(tracks[i,j,2]):
        I = np.nonzero(B)[0]
        if len(I)==0 or I[-1]==i-1:
            return False
        interpolate(tracks[:,j,2], I[-1], i, angle=True)
        return True
    else:
        I = np.nonzero(~B)[0]
        if len(I)==0 or I[-1]==i-1:
            return False
        interpolate(tracks[:,j,2], I[-1], i, angle=True)
        return True

# Delete position of fish j in frame i.
def fix_delete_position(tracks, i, j):
    tracks[i,j,:2] = np.nan

# Delete orientation of fish j in frame i.
def fix_delete_orientation(tracks, i, j):
    tracks[i,j,2] = np.nan

def fix(tracks, *args):
    globals()['fix_'+args[0]](tracks, *args[1:])

