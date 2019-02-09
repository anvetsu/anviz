"""
mandelbrot.py - Implement the Mandelbrot set in Python using matplotlib & numpy,
by reusing code from the Julia set.


"""

import matplotlib.pyplot as plt
import numpy as np

def mandelbrot_set(width, height, zoom=1, x_off=0, y_off=0, niter=256):
    """ A mandelbrot set of geometry (width x height) and iterations 'niter' """

    w,h = width, height
    pixels = np.arange(w*h, dtype=np.uint16).reshape(h, w)

    # The mandelbrot set represents every complex point "c" for which
    # the Julia set is connected or every julia set that contains
    # the origin (0, 0). Hence we always start with c at the origin

    for x in range(w): 
        for y in range(h):
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

            color = (i << 21) + (i << 10) + i*8
            pixels[y,x] = color
  
    return pixels

def display(width=1024, height=768, zoom=1.0, x_off=0, y_off=0, cmap='viridis'):
    """ Display a julia set of width `width` and height `height` and zoom `zoom`
    and offsets (x_off, y_off) """

    pixels = mandelbrot_set(width, height, zoom=zoom, x_off=x_off, y_off=y_off)
    # Let us turn off the axes
    plt.axis('off')
    # to display the created fractal
    plt.imshow(pixels, cmap=cmap)
    plt.show()
    
if __name__ == "__main__":
    display()
