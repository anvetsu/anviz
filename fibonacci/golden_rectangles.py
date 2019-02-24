"""

Plot the Fibonacci 'Golden Rectangles' which lead to the Golden curve
using matplotlib.

"""


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def fibonacci(n=10):
    """ Generate fibonacci numbers upto 10 starting with 1,2 ... """

    a, b = 0,1
    for i in range(n):
        c = a + b
        yield (c,b)     
        a,b=b,c

def fourth_series(n=10):
    """ Generate 1, 5, 9 ... upto n elements """

    x = 1
    for i in range(n):
        yield x
        x += 4

def sixth_series(n=10):
    """ Generate 4, 10, 16 ... upto n elements """

    x=4
    for i in range(n):
        yield x
        x += 6
        
def golden_rectangles(max_n=10):
    """ Generate and plot successive golden rectangles """

    # List of fibonacci numbers as (fn, fn-1) pair
    fibs = list(fibonacci(max_n))
    # Reverse as we need to generate rectangles
    # from large -> small
    fibs.reverse()

    # Create a sub-plot
    fig, ax = plt.subplots(1)

    last_x, last_y = fibs[0]
    # Make the plot size large enough to hold
    # the largest fibonacci number on both
    # x and y-axis.
    ax.set_xlim(0, last_x + 10)
    ax.set_ylim(0, last_y + 10) 
    # Turn off the axes
    plt.axis('off')

    # First rectangle is centered at (0,0)
    origin = [0, 0]

    # Rectangles
    rects = []
    
    for i,(cur_fn, prev_fn) in enumerate(fibs):
        if i > max_n: break

        if i in fourth_series(max_n):
            # Every 4th rectangle from the 2nd
            # rectangle onwards has its origin-x
            # point shifted by the fibonacci value
            origin[0] = origin[0] + cur_fn

        elif i in sixth_series(max_n):
            # Every 6th rectangle from the 5th
            # rectangle onwards has its origin-y
            # point shifted by the fibonacci value
            origin[1] = origin[1] + cur_fn
            
        if i%2 == 1:
            # Every 2nd rectangle has its orientation
            # switched from lxb to bxl
            cur_fn, prev_fn = prev_fn, cur_fn

        rectangle = Rectangle(origin, cur_fn, prev_fn, angle=0.0, antialiased=True)
        rects.append(rectangle)

    # Add the rectangles to the plot
    rect_pcs = PatchCollection(rects, facecolor='g', alpha=0.4,
                               edgecolor='black')

    ax.add_collection(rect_pcs)
    plt.show()

if __name__ == "__main__":
    golden_rectangles()
