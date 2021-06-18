import numpy as np
import csv
import sys
import concurrent.futures

from validate import validate

train_X_file_path = "./train_X_lg_v2.csv"
train_Y_file_path = "./train_Y_lg_v2.csv"
validation_split = 0.2
numClasses = 4
tolerance = 0.000001
maxEpochs = 100000
learningRates = [0.01, 0.007]

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_lg.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float128, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float128)
    return test_X, weights


def predict_target_values(test_X, weights):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    out = []
    for i in range(numClasses):
        W, b = weights[i][1:] , weights[i][0] 
        out.append(sigmoid(np.dot(test_X,W)+b))
    out = np.transpose(out)
    out = np.argmax(out, axis=1)
    return out

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def getAccuracy(X, Y, W, b):
    Yhat = sigmoid(np.dot(X,W)+b)
    Yhat[Yhat == 1] = 0.999999
    Yhat[Yhat == 0] = 0.000001
    Yhat = (Yhat >= 0.5).astype(int)
    return sum(Yhat == Y)[0]/Y.shape[0]

def getF1score(X, Y, W, b):
    Yhat = sigmoid(np.dot(X,W)+b)
    Yhat[Yhat == 1] = 0.999
    Yhat[Yhat == 0] = 0.001
    Yhat = (Yhat >= 0.5).astype(int)
    from sklearn.metrics import f1_score
    return f1_score(Y, Yhat, average = 'weighted')

def computeCost(X, Y, W, b):
    m = len(X)
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return cost

def getGrads(X, Y, W, b):
    m = len(X)
    Z = np.dot(X,W) + b
    Yhat = sigmoid(Z)
    dW = 1/m * np.dot(X.T, (Yhat - Y))
    db = 1/m * np.sum(Yhat - Y)
    return dW, db

def trainValSplit(X,Y):
    train_X = np.copy(X)
    train_Y = np.copy(Y)
    valIndex = -int(validation_split*(train_X.shape[0]))
    val_X = train_X[valIndex:]
    val_Y = train_Y[valIndex:]
    train_X = train_X[:valIndex]
    train_Y = train_Y[:valIndex]
    return (train_X, train_Y, val_X, val_Y)

def builModel(X, Y):
    train_X, train_Y, val_X, val_Y = trainValSplit(X,Y)
    minCost = np.inf 
    minW = 0
    minB = 0
    for learningRate in learningRates:
        W = np.random.normal(0, 1, (X.shape[1], 1))
        b = np.random.rand() 
        numEpochs = 0
        prevCost = 0
        # print(f"lr : {learningRate}, regParam : {regParam}, lrDecay : {lrDecay}")
        while numEpochs < maxEpochs:
            dW, dB = getGrads(train_X, train_Y, W, b) 
            W = W - learningRate*(dW) 
            b = b - learningRate*(dB)  
            curCost = computeCost(val_X, val_Y, W, b) 
            acc = getAccuracy(val_X, val_Y, W, b)
            f1 = getF1score(val_X, val_Y, W, b)
            # print(f"\tEpoch : {numEpochs}, Weighted F1 score : {f1}")
            if curCost < minCost:
                minCost = curCost 
                minW, minB = W, b
            if abs(curCost - prevCost) < tolerance:
                break
            numEpochs += 1
            prevCost = curCost
    return (minW, minB)

def predict(test_X_file_path):
    train_X = np.genfromtxt(train_X_file_path, delimiter=",", dtype=np.float128, skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter="\n", dtype=np.int64, skip_header=0)
    train_X = np.array(train_X)
    train_Y = np.array(train_Y).reshape(train_X.shape[0],1)
    train_X = (train_X - np.mean(train_X, axis=0))/np.std(train_X, axis=0)
    Ws = []
    Xs = []
    Ys = []
    for i in range(numClasses):
        X = np.copy(train_X)
        Y = np.copy(train_Y)
        Y = (Y==i).astype(int)
        Xs.append(X)
        Ys.append(Y)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        classes = list(range(numClasses))
        WsandBs = executor.map(builModel, Xs, Ys)
        for W, b in WsandBs:
            theta = [b]
            theta.extend(W)
            Ws.append(theta)    
    Ws = np.array(Ws, dtype=object)
    np.savetxt("WEIGHTS_FILE.csv",Ws,delimiter=",")
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    test_X = (test_X - np.mean(test_X, axis=0))/np.std(test_X, axis=0)
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 