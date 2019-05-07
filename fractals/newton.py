"""
newton.py - Implement fractals using newton-raphson approximation method
of solving functions.

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import functools
import multiprocessing as mp

from PIL import Image

def rgb2int(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red << 16) + (green << 8) + blue
    return RGBint

def int2rgb(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return (red, green, blue)

# put any complex function here to generate a fractal for it!
def fcube(z):
    return z ** 3 - 1

def f8(z):
    return z**8 + 15*z**4 - 16

def sin(z):
    return math.sin(abs(z))

def cos(z):
    return math.cos(abs(z))

def circle(z):
    return math.sin(abs(z))**2 + math.cos(abs(z))**2

def tan(z):
    return math.tan(abs(z))

def cot(z):
    return 1.0/math.tan(abs(z))

def fcube2(z):
    return z**3 - 2*z

def fsquare(z):
    return z**2 - 1

def fquad(z):
    return z**4 - 1

def fsix(z):
    return z**6  + z**3 - 1

def fcomplex(z):
    return z**(0.5+0.3j) - 1

def ftrig1(z):
    return math.cos(abs(z)) - math.sin(abs(z))


def newton_set_calc_row(y, width, height, function, niter=256, x_off=0, y_off=0, zoom=1):
    """ Calculate one row of the newton set with size width x height """

    row_pixels = np.arange(width, dtype=np.uint32)
    # drawing area
    xa, xb, ya, yb = -2.5, 2.5, -2.5, 2.5

    zy = (y + y_off) * (yb - ya) / (zoom*(height - 1)) + ya   
    
    h = 1e-7 # step size for numerical derivative
    eps = 1e-3 # max error allowed
    a = complex(1, 0)

    for x in range(width): 
        # calculate the initial real and imaginary part of z,
        # based on the pixel location and zoom and position values
        zx = (x + x_off) * (xb - xa) / (zoom*(width - 1)) + xa
        z = complex(zx, zy)
        count = 0
        
        for i in range(niter):
            # complex numerical derivative
            dz = (function(z + complex(h, h)) - function(z)) / complex(h, h)
            if dz == 0:
                break

            count += 1
            if count > 255:
                break

            znext = z - a*function(z) / dz # Newton iteration
            if abs(znext - z) < eps: # stop when close enough to any root
                break
                
            z = znext

        # Color according to iteration count 
        rgb = (i % 16 * 32, i % 8 * 64, i % 4 * 64)                              
        row_pixels[x] = rgb2int(rgb)


    return y,row_pixels

# draw the fractal
def newton_set(width, height, zoom=1, x_off=0, y_off=0, niter=256):
    """ Fractals using newton-raphson """
    
    # drawing area
    xa, xb, ya, yb = -2.5, 2.5, -2.5, 2.5
    # Pixels array
    pixels = np.arange(width*height*3, dtype=np.uint32).reshape(height, width, 3)
    
    h = 1e-7 # step size for numerical derivative
    eps = 1e-3 # max error allowed

    # Bounding roots
    r1 = 1
    r2 = complex(-0.5, math.sin(2*math.pi/3))
    r3 = complex(-0.5, -1*math.sin(2*math.pi/3))

    # Color multiplication factor
    multcol = 5
        
    for y in range(height):
        zy = (y + y_off) * (yb - ya)/ (zoom*(height - 1)) + ya 

        for x in range(width):
            zx = (x + x_off) * (xb - xa)/ (zoom*(width - 1)) + xa 
            z = complex(zx, zy)
            count = 0
            
            for i in range(niter):
                # complex numerical derivative
                dz = (fcube(z + complex(h, h)) - fcube(z)) / complex(h, h)
                if dz == 0:
                    break

                count += 1
                if count > 255:
                    break
                
                znext = z - fcube(z) / dz # Newton iteration
                if abs(znext - z) < eps: # stop when close enough to any root
                    break
                
                z = znext

            # Pixels colored using the roots
            if abs(z-r1)<eps:
                pixels[y,x] = (255 - count*multcol, 0, 0)
            elif abs(z-r2)<=eps:
                pixels[y,x] = (0, 255 - count*multcol, 0)
            elif abs(z-r3)<=eps:
                pixels[y,x] = (0, 0, 255 - count*multcol)
        
    return pixels

def newton_set_mp(width, height, function, zoom=1, x_off=0, y_off=0, niter=256):
    """ Newton-raphson fractal set with multiprocessing """
    
    w,h = width, height
    pixels = np.arange(w*h*3, dtype=np.uint32).reshape(h, w, 3)  

    # print('Starting calculation using',width, height,cx,cy)
    pool = mp.Pool(mp.cpu_count())

    newton_partial = functools.partial(newton_set_calc_row, 
                                      width=width,height=height, function=function,
                                      niter=niter,zoom=zoom,x_off=x_off,y_off=y_off)

    for y,row_pixel in pool.map(newton_partial, range(h)):
        for x in range(w):
            pixels[y, x] = np.array(int2rgb(row_pixel[x]))

    return pixels
        
def display(function, width=1024, height=1024, niter=1024, zoom=1, x_off=0, y_off=0):
    """ Display a newton-raphson fractal """

    # pimg = newton_set(width, height, zoom=zoom, x_off=x_off, y_off=y_off, niter=niter)
    pimg = newton_set_mp(width, height, function, zoom=zoom,x_off=x_off, y_off=y_off, niter=niter) 
    plt.axis('off') 
    plt.imshow(pimg)
    plt.show()

def fsqr(z):
    return z**2 - 1

def fquad(z):
    return z**4 - 1

def fsix(z):
    return z**6  + z**3 - 1

def f8(z):
    return z**8 + 15*z**4 - 16

def fe(z):
    return math.e**z

def flog(z):
    return math.log(abs(z))

if __name__ == "__main__":
    display(f8)
