def gradient_descent_1d(f, df, x0, lr=0.1, steps=20):
    x = x0

    for step in range(steps):
        grad = df(x)
        x = x - lr * grad

        print(f"step {step}: x={x:.6f}, f(x)={f(x):.6f}")

    return x


def numerical_gradient(f, point, h=1e-7):
    gradient = []

    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)

        point_plus[i] += h
        point_minus[i] -= h

        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)

    return gradient


def gradient_descent_nd(f, x0, lr=0.1, steps=100):
    point = list(x0)

    for step in range(steps):
        grad = numerical_gradient(f, point)

        point = [
            p - lr * g
            for p, g in zip(point, grad)
        ]

        print(f"step {step}: point={point}, f={f(point):.6f}")

    return point

f = lambda x: x**2
df = lambda x: 2*x

minimum = gradient_descent_1d(
    f,
    df,
    x0=5.0,
    lr=0.1,
    steps=20
)

print("Minimum:", minimum)

def f2(point):
    x, y = point
    return x**2 + y**2


minimum_2d = gradient_descent_nd(
    f2,
    x0=[4.0, 3.0],
    lr=0.1,
    steps=30
)

print("2D Minimum:", minimum_2d)