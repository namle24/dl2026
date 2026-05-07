import csv

def load_data(filename):

    X = []
    Y = []

    with open(filename, "r") as file:

        reader = csv.reader(file)
        next(reader)

        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))

    return X, Y


def loss(x, y, w0, w1):
    return 0.5 * (w1 * x + w0 - y) ** 2


def LinearRegression(X, Y, lr, threshold, max_iter):

    w0 = 0
    w1 = 1

    N = len(X)

    count = 0

    while count < max_iter:

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

        count += 1

        print(count, w0, w1, total_loss)
        print(count, "loss =", total_loss)
        if total_loss < threshold:
            break

    return w0, w1


if __name__ == "__main__":

    X, Y = load_data("house_price.csv")

    lr = 0.00001
    threshold = 0.01
    max_iter = 50000

    w0, w1 = LinearRegression(X, Y, lr, threshold, max_iter)
    
    print("\nFinal result:")
    print("w0 =", w0)
    print("w1 =", w1)
