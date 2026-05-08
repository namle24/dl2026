import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def load_data(filename):

    X = []
    Y = []
    with open(filename, "r") as file:
        lines = file.readlines()[1:]
        for line in lines:
            row = line.strip().split(",")
            X.append(float(row[0]))
            Y.append(float(row[1]))

    return X, Y

def loss(x, y, w0, w1):
    return 0.5 * (w1 * x + w0 - y) ** 2

def LinearRegression(X, Y, lr, threshold):

    w0 = 0
    w1 = 1
    N = len(X)

    while True:

        dw0 = 0
        dw1 = 0
        total_loss = 0

        for i in range(N):
            x = X[i]
            y = Y[i]
            error = (w1 * x + w0 - y)
            dw0 += error
            dw1 += x * error
            total_loss += loss(x, y, w0, w1)

        dw0 = dw0 / N
        dw1 = dw1 / N
        total_loss = total_loss / N
        w0 = w0 - lr * dw0
        w1 = w1 - lr * dw1
        print("loss =", total_loss)
        if total_loss < threshold:
            break

    return w0, w1

if __name__ == "__main__":

    X, Y = load_data("lr.csv")

    lr = 0.0001
    threshold = 10

    w0, w1 = LinearRegression(X, Y, lr, threshold)

    print("\nFinal result:")
    print("w0 =", w0)
    print("w1 =", w1)

    plt.scatter(X, Y)

    Y_pred = []

    for x in X:
        Y_pred.append(w1 * x + w0)

    plt.plot(X, Y_pred)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")

    plt.show()
