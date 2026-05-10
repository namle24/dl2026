import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

E = 2.718281828459045

def load_data(filename):
    X1, X2, Y = [], [], []
    with open(filename, "r") as file:
        lines = file.readlines()[1:]
        for line in lines:
            row = line.strip().split(",")
            X1.append(float(row[0]))
            X2.append(float(row[1]))
            Y.append(float(row[2]))
    return X1, X2, Y

def sigmoid(z):
    return 1 / (1 + E**(-z))

def log(x):
    if x <= 0: return -1e9
    n = 1000000.0
    return n * ((x ** (1/n)) - 1)

def loss(y, y_pred):
    eps = 1e-15
    y_pred = max(eps, min(1 - eps, y_pred))
    return -(y * log(y_pred) + (1 - y) * log(1 - y_pred))

def LogisticRegression(X1, X2, Y, lr, threshold):
    w0, w1, w2 = 0.0, 1.0, 2.0
    N = len(Y)
    count = 0

    while True:
        dw0, dw1, dw2 = 0.0, 0.0, 0.0
        total_loss = 0.0

        for i in range(N):
            x1, x2, y = X1[i], X2[i], Y[i]

            z = w1 * x1 + w2 * x2 + w0
            y_pred = sigmoid(z)

            error = y_pred - y

            dw0 += error
            dw1 += x1 * error
            dw2 += x2 * error

            total_loss += loss(y, y_pred)

        dw0 /= N
        dw1 /= N
        dw2 /= N
        total_loss /= N

        w0 -= lr * dw0
        w1 -= lr * dw1
        w2 -= lr * dw2

        count += 1
        print(f"Iter {count}, Loss = {total_loss}")

        if total_loss < threshold:
            break
    return w0, w1, w2

if __name__ == "__main__":
    try:
        X1, X2, Y = load_data("/home/namle/dl2026/Lab2/loan2.csv")
        
        lr = 0.1
        threshold = 0.1983

        w0, w1, w2 = LogisticRegression(X1, X2, Y, lr, threshold)

        print("\nFinal result:")
        print(f"w0 = {w0}")
        print(f"w1 = {w1}")
        print(f"w2 = {w2}")

        plt.scatter(X1, X2, c=Y)
        plt.xlabel("Salary")
        plt.ylabel("Experience")
        plt.title("Logistic Regression")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")

