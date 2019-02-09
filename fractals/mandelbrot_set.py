"""
Mandelbrot set using Python, matplotlib and numpy

Serial and Parallel versions

"""

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import functools

def mandelbrot_set_calc_column(x, w, h, x_off=0, y_off=0, zoom=1, niter=256):
    """ Calculate one column of the mandelbrot set with size wxh """

    col_pixels = np.arange(h, dtype=np.uint16)
        
    zx = 1.5*(x + x_off -3*w/4)/(0.5*zoom*w)
        
    for y in range(h): 
        # calculate the initial real and imaginary part of z,
        # based on the pixel location and zoom and position values
        zy = 1.0*(y + y_off - h/2)/(0.5*zoom*h)

        z = complex(zx, zy)
        c = complex(0, 0)
        
        for i in range(niter):
            if abs(c) > 4: break
            # Iterate till the point is outside
            # the circle with radius 2.             
            # Calculate new positions
            c = c**2 + z

        col_pixels[y] = (i<<21) + (i<<10) + i*8

    return x,col_pixels

def mandelbrot_set_mp(width, height, x_off=0, y_off=0, zoom=1, niter=256):
    """ Mandelbrot set function - using multiprocessing """

    w,h = width, height
    pixels = np.arange(w*h, dtype=np.uint16).reshape(w, h)

    # print('Starting calculation using',width, height,cx,cy)
    pool = mp.Pool(mp.cpu_count())

    mandelbrot_partial = functools.partial(mandelbrot_set_calc_column, 
                                           w=w,h=h,x_off=x_off,y_off=y_off,
                                           niter=niter,zoom=zoom)

    for x,col_pixel in pool.map(mandelbrot_partial, range(w)):
        pixels[x] = col_pixel

    return np.transpose(pixels)

def mandelbrot_set(width, height, zoom=1, x_off=0, y_off=0, niter=256):
    """ A mandelbrot set of geometry (width x height) and iterations 'niter' """

    # Serial version
    w,h = width, height
    pixels = np.arange(w*h, dtype=np.uint16).reshape(h, w)

    # The mandelbrot set represents every complex point "c" for which
    # the Julia set is connected or every julia set that contains
    # the origin (0, 0). Hence we always start with c at the origin

    for y in range(h): 
        for x in range(w):
            # calculate the initial real and imaginary part of z,
            # based on the pixel location and zoom and position values
            # We use (x-3*w/4) instead of (x-w/2) to fully visualize the fractal
            # along the x-axis
            
            zx = 1.5*(x + x_off - 3*w/4)/(0.5*zoom*w)
            zy = 1.0*(y + y_off - h/2)/(0.5*zoom*h)
            
            z = complex(zx, zy)
            c = complex(0, 0)
            
            for i in range(niter):
                if abs(c) > 4: break
                # Iterate till the point c is outside
                # the circle with radius 2.             
                # Calculate new positions
                c = c**2 + z

            pixels[y,x] = (i<<21) + (i<<10) + i*8
  
    return pixels

def display(width=1024, height=768, zoom=1.0, x_off=0, y_off=0, cmap='viridis'):
    """ Display a julia set of width `width` and height `height` and zoom `zoom`
    and offsets (x_off, y_off) """

    pixels = mandelbrot_set_mp(width, height, zoom=zoom, x_off=x_off, y_off=y_off)
    # Let us turn off the axes
    plt.axis('off')
    # to display the created fractal
    plt.imshow(pixels, cmap=cmap)
    plt.show()

if __name__ == "__main__":
    display()
