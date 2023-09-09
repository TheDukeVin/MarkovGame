
s = 100

def f(x, y):
    return x**2 + (y/s)**2

#Gradient descent

def gradient(x, y):
    return (2*x, 2*y/s**2)

lr = 0.1

def grad_descent(start_point):
    p = start_point
    numIter = 0
    while True:
        x, y = p
        if f(x, y) < 1e-07:
            break
        dx, dy = gradient(x, y)
        x -= dy * lr
        y -= dy * lr
        p = (x, y)
        numIter += 1
    print(numIter)

grad_descent((1, 1))

# Line Search

def line_search(start_point):
    p = start_point
    numIter = 0
    while True:
        x, y = p
        if f(x, y) < 1e-07:
            break
        dx, dy = gradient(x, y)
        d = (dx**2 + dy**2) ** 0.5
        dx /= -d
        dy /= -d
        lower = 0
        upper = 1
        while upper - lower > 1e-10:
            middle = (upper + lower) / 2
            x_ = x + dx*middle
            y_ = y + dy*middle
            gx, gy = gradient(x_, y_)
            if dx*gx + dy*gy > 0:
                upper = middle
            else:
                lower = middle
        p = (x_, y_)
        numIter += 1
        print(p)

line_search((1, 1))