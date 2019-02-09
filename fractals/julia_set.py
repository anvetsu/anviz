"""
Julia set using Python, matplotlib and numpy

Serial and Parallel versions

# Reimplemented to use complex numbers

"""

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import functools

def julia_set_calc_row(y, w, h, c, niter=256, zoom=1,):
    """ Calculate one row of the julia set with size wxh """

    # image_rows = {}
    row_pixels = np.arange(w)
        
    for x in range(w): 
        # calculate the initial real and imaginary part of z,
        # based on the pixel location and zoom and position values
        zx = 1.5*(x - w/2)/(0.5*zoom*w) 
        zy = 1.0*(y - h/2)/(0.5*zoom*h)
        z = complex(zx, zy)
        
        for i in range(niter):
            radius_sqr = abs(z)
            # Iterate till the point is outside
            # the circle with radius 2.             
            if radius_sqr > 4: break
            # Calculate new positions
            z = z**2 + c

            color = (i >> 21) + (i >> 10)  + i * 8
            row_pixels[x] = color

    return y,row_pixels

def julia_set_calc_column(x, w, h, c, niter = 256, zoom=1,):
    """ Calculate one column of the julia set with size wxh """

    col_pixels = np.arange(h)
        
    for y in range(h): 
        # calculate the initial real and imaginary part of z,
        # based on the pixel location and zoom and position values
        zx = 1.5*(x - w/2)/(0.5*zoom*w) 
        zy = 1.0*(y - h/2)/(0.5*zoom*h)
        z = complex(zx, zy)
        
        for i in range(niter):
            radius_sqr = abs(z)
            # Iterate till the point is outside
            # the circle with radius 2.             
            if radius_sqr > 4: break
            # Calculate new positions
            z = z**2 + c

            color = (i >> 21) + (i >> 10)  + i * 8
            col_pixels[y] = color

    return x,col_pixels

def julia_set_mp(width, height, cx=-0.7, cy=0.27, zoom=1, niter=256):
    """ Julia set function - using multiprocessing """

    w,h = width, height
    pixels = np.arange(w*h, dtype=np.uint32).reshape(h, w)

    # Pick some defaults for the real and imaginary constants
    # This determines the shape of the Julia set.
    c = complex(cx, cy)

    # print('Starting calculation using',width, height,cx,cy)
    pool = mp.Pool(mp.cpu_count())

    julia_partial = functools.partial(julia_set_calc_row, 
                                      w=w,h=h,
                                      c = c,
                                      niter=niter,zoom=zoom)

    for y,row_pixel in pool.map(julia_partial, range(h)):
        pixels[y] = row_pixel

    # Uncomment for column parallel version
    #for x,col_pixel in pool.map(julia_partial, range(w)):
    #    pixels[x] = col_pixel       
  
    # print('Ending calculation')
    
    # return np.transpose(pixels)
    return pixels

            
def julia_set(width, height, cx=-0.7, cy=0.27, zoom=1, niter=256):
    """ A julia set of geometry (width x height) and iterations 'niter' """

    # Simple version
    w,h = width, height
    pixels = np.arange(w*h, dtype=np.uint32).reshape(h, w)

    # Pick some defaults for the real and imaginary constants
    # This determines the shape of the Julia set.
    c = complex(cx, cy)

    #print('Starting calculation using',width, height,cx,cy)
    
    for x in range(w): 
        for y in range(h):
            # calculate the initial real and imaginary part of z,
            # based on the pixel location and zoom and position values
            zx = 1.5*(x - w/2)/(0.5*zoom*w) 
            zy = 1.0*(y - h/2)/(0.5*zoom*h)
            z = complex(zx, zy)
            
            for i in range(niter):
                radius_sqr = abs(z)
                # Iterate till the point is outside
                # the circle with radius 2.             
                if radius_sqr > 4: break
                # Calculate new positions
                z = z**2 + c

            color = (i >> 21) + (i >> 10)  + i * 8
            pixels[y,x] = color
  

    # print('Ending calculation')
    return pixels

def display(width=2048, height=1536):
    """ Display a julia set of width `width` and height `height` """

    pixels = julia_set(width, height)
    # to display the created fractal 
    plt.imshow(pixels)
    plt.show()
    
if __name__ == "__main__":
    display()

    
    
