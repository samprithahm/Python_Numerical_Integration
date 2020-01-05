"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Skeleton/solution for Assignment 1: Numerical Integration

By writing my name below and submitting this file, I/we declare that
all additions to the provided skeleton file are my/our own work, and that
I/we have not seen any work on this assignment by another student/group.

Student name(s): Sampritha Hassan Manjunath, Anitha Govindaraju
Student ID(s): 19232922, 19230254

"""

import numpy as np
import sympy
import itertools
import math

def numint_py(f, a, b, n):
    """Numerical integration. For a function f, calculate the definite
    integral of f from a to b by approximating with n "slices" and the
    "lb" scheme. This function must use pure Python, no Numpy.

    >>> abs(numint_py(math.sin, 0, 2*math.pi, 100) - 0) < 10**-8
    True
    >>> round(numint_py(lambda x: 1, 0, 1, 100), 5)
    1.0
    >>> round(numint_py(math.exp, 1, 2, 100), 5)
    4.64746

    """
    A = 0
    w = (b - a) / n # width of one slice
    # calculating numerical integral from 'a' to 'b'
    i = a
    # calculate area under f(x) for each slice until x reaches n
    while i <= b:
        A += f(i)
        i += w
    # multiply the width to get the area
    return A*w

def numint(f, a, b, n, scheme='mp'):
    """Numerical integration. For a function f, calculate the definite
    integral of f from a to b by approximating with n "slices" and the
    given scheme. This function should use Numpy, and eg np.linspace()
    will be useful.
    
    >>> abs(numint(np.sin, 0, 2*math.pi, 100) - 0) < 10**-8
    True
    >>> round(numint(lambda x: np.ones_like(x), 0, 1, 100), 5)
    1.0
    >>> round(numint(np.exp, 1, 2, 100, 'lb'), 5)
    4.64746
    >>> round(numint(np.exp, 1, 2, 100, 'mp'), 5)
    4.67075
    >>> round(numint(np.exp, 1, 2, 100, 'ub'), 5)
    4.69417

    """
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    w = (b-a)/n
    
    # calulate area from x(0) to x(n-1) for n slices
    if scheme=='lb':
        x = np.linspace(a, b-w, n)
    
    # calulate area from x(1) to x(n) for n slices
    if scheme=='ub':
        x = np.linspace(a+w, b, n)
        
    # calulate area from x(0) to x(n) for n slices
    if scheme == 'mp':
        x = np.linspace(a+(b-a)/(2*n), b-(b-a)/(2*n), n)
    
    y = f(x)
    # sum of all the area under f(x) for n slices
    area = np.sum(y)*(b-a)/n
    return area


def true_integral(fstr, a, b):
    """Using Sympy, calculate the definite integral of f from a to b and
    return as a float. Here fstr is an expression in x, as a str. It
    should use eg "np.sin" for the sin function.

    This function is quite tricky, so you are not expected to
    understand it or change it! However, you should understand how to
    use it. See the doctest examples.

    >>> true_integral("np.sin(x)", 0, 2 * np.pi)
    0.0
    >>> true_integral("x**2", 0, 1)
    0.3333333333333333

    """
    x = sympy.symbols("x")
    # make fsym, a Sympy expression in x, now using eg "sympy.sin"
    fsym = eval(fstr.replace("np", "sympy")) 
    A = sympy.integrate(fsym, (x, a, b)) # definite integral
    A = float(A.evalf()) # convert to float
    return A

def numint_err(fstr, a, b, n, scheme):
    """For a given function fstr and bounds a, b, evaluate the error
    achieved by numerical integration on n points with the given
    scheme. Return the true value, absolute error, and relative error
    as a tuple.

    Notice that the relative error will be infinity when the true
    value is zero. None of the examples in our assignment will have a
    true value of zero.

    >>> print("%.4f %.4f %.4f" % numint_err("x**2", 0, 1, 10, 'lb'))
    0.3333 0.0483 0.1450

    """
    f = eval("lambda x: " + fstr) # f is a Python function
    A = true_integral(fstr, a, b)
    
    # calculate numberical integral for f(x)
    B = numint(f, a, b, n, scheme)
    
    # calculate the absolute error: difference between true integral and numberal integral of f(x)
    AbsErr = abs(A - B)
    
    # calculate relative error: absolute error divided by true integral of f(x)
    RelErr = AbsErr / A
    
    return A, AbsErr, RelErr

def make_table(f_ab_s, ns, schemes):
    """For each function f with associated bounds (a, b), and each value
    of n and each scheme, calculate the absolute and relative error of
    numerical integration and print out one line of a table. This
    function doesn't need to return anything, just print. Each
    function and bounds will be a tuple (f, a, b), so the argument
    f_ab_s is a list of tuples.

    Hint: use print() with the format string
    "%s,%.2f,%.2f,%d,%s,%.4g,%.4g,%.4g". Hint 2: consider itertools.

    >>> make_table([("x**2", 0, 1), ("np.sin(x)", 0, 1)], [10, 100], ['lb', 'mp'])
    x**2,0.00,1.00,10,lb,0.3333,0.04833,0.145
    x**2,0.00,1.00,10,mp,0.3333,0.0008333,0.0025
    x**2,0.00,1.00,100,lb,0.3333,0.004983,0.01495
    x**2,0.00,1.00,100,mp,0.3333,8.333e-06,2.5e-05
    np.sin(x),0.00,1.00,10,lb,0.4597,0.04246,0.09236
    np.sin(x),0.00,1.00,10,mp,0.4597,0.0001916,0.0004168
    np.sin(x),0.00,1.00,100,lb,0.4597,0.004211,0.009161
    np.sin(x),0.00,1.00,100,mp,0.4597,1.915e-06,4.167e-06

    """
   
    # using itertools to get argument values
    for func, numSlice, scheme in itertools.product(f_ab_s, ns, schemes):
        # calculating numerical integration errors and storing the values in variables
        trueInt,absErr,relErr = numint_err(func[0],func[1],func[2],numSlice,scheme)
        
        # display all the calculated values as formated output 
        print("%s,%.2f,%.2f,%d,%s,%.4g,%.4g,%.4g" %(func[0],func[1],func[2], numSlice, scheme, trueInt,absErr,relErr))

def main():
    """Call make_table() as specified in the pdf."""
    # calling make_table as per the assignment pdf document
    make_table([("np.cos(x)",0,math.pi/2),("np.sin(2*x)",0,1),("np.exp(x)",0,1)],[10,100,1000],['lb','mp'])
    
    """
    Based on the calualtions made above, we strongly belive that "mp" scheme is 
    efficient in calculating numerical integration since the error is negligebile.
    For example, consider the function numint_err("x**2", 0, 1, 10, 'mp'))
    Expected: True Integral = 0.3333, Absolute Error = 0.0483 and Relative Error = 0.1450
    For lb scheme: True Integral = 0.3333, Absolute Error = 0.0483 and Relative Error = 0.1450
    For ub scheme: True Integral = 0.3333, Absolute Error = 0.0517 and Relative Error = 0.1550
    For mp scheme: True Integral = 0.3333, Absolute Error = 0.0008 and Relative Error = 0.0025
    From the above example we can see that scheme 'mp' results in considerably less error

    """


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
