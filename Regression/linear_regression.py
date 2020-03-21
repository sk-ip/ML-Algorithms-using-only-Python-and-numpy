from sklearn.datasets import make_regression


def display_result(*args, **kwargs):
    print(args)


def loadDataSet():
    x, y = make_regression(n_samples=20, n_features=1, noise=0.2)
    return x, y


def calculateMean(data):
    return sum(data) / float(len(data))


def calculateVariance(data, mean):
    return sum([(x - mean) ** 2 for x in data])


def calculateCovariance(x, x_mean, y, y_mean):
    covr = 0.0
    for i in range(len(x)):
        covr += sum((x[i] - x_mean) * (y[i] - y_mean))

    return covr


def linearRegression(data, result):
    mean_x = calculateMean(data)
    mean_y = calculateMean(result)
    var_x = calculateVariance(data, mean_x)
    vat_y = calculateVariance(result, mean_y)

    # assuming y = B1*x + B0
    b1 = calculateCovariance(x, mean_x, y, mean_y) / calculateVariance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    return b1, b0


if __name__ == "__main__":
    x, y = loadDataSet()
    print("The dataset is")
    print(list(zip(x, y)))
    w, b = linearRegression(x, y)
    print("The slope of the line is: ", w, end="\n")
    print("The intercept of the line is: ", b)
