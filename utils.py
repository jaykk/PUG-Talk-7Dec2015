import numpy as np

def showimg(fname):
    import matplotlib.pyplot as plt
    from scipy.misc import imread
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    return plt.imshow(imread(fname))

def add_rect(pic):
    """ Add a red triangle over a 3 channel numpy array """
    h = 0.1
    w = 0.05
    lx, ly, lz = pic.shape

    x0,y0 = ((1 - 2*w) * np.random.random() + w, (1 - 2*h)*np.random.random() + h)
    x0 = x0 * lx
    y0 = y0 * ly
    x1 = x0 + (w * lx)
    y1 = y0 + (h * ly)
    X, Y = np.ogrid[0:lx, 0:ly]
    mask = X + 0*Y > x1
    mask = mask + (0*X + Y > y1)
    mask = mask + (X + 0*Y < x0)
    mask = mask + (0*X + Y < y0)
    mask = ~mask
    p = pic.copy()
    p[mask,:] = 0
    p[mask,0] = 255
    return p

def add_circle(pic):
    """ Add a white triangle over a 3 channel numpy array """
    r = 0.05
    lx, ly, lz = pic.shape
    x = (1-2*r)*np.random.random() + r
    y = (1-2*r)*np.random.random() + r
    x,y = (lx * x, ly * y)
    r = lx * r
    X, Y = np.ogrid[0:lx, 0:ly]
    mask = (X - x) ** 2 + (Y - y) ** 2 < r**2
    p = pic.copy()
    p[mask,:] = 0
    p[mask,0] = 255
    return p
