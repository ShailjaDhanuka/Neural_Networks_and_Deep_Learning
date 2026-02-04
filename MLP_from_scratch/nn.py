"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
import pickle
import os


# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """ 
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()
        self.x=None
    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        self.x=x
        RELUed = np.maximum(0,x)
        return RELUed
    

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        grad_wrt_relu = grad_wrt_out * (self.x > 0)
        return grad_wrt_relu


class Softmax(Transform):
    def __init__(self):
        super(Softmax, self).__init__()
        self.x=None

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        self.x=x
        shifted = x - np.max(self.x, axis=0, keepdims=True)

        exp_x = np.exp(shifted)
        softmax = exp_x / np.sum(exp_x, axis=0, keepdims=True)

        return softmax



class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 * np.random.rand(outdim, indim)
        self.bias = 0.01 * np.random.rand(outdim, 1)
        self.lr = lr
        self.x=None
        


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.x=x
        out = self.weights @ x + self.bias
        return out
    

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
    
        """
        batch_size = self.x.shape[1]
        #compute grad_wrt_weights
        self.grad_weights = grad_wrt_out @ self.x.T / batch_size

        #compute grad_wrt_bias
        self.grad_bias = np.sum(grad_wrt_out, axis=1, keepdims=True) / batch_size

        #compute & return grad_wrt_input
        grad_wrt_x = self.weights.T @ grad_wrt_out
        return grad_wrt_x

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights -= self.lr * self.grad_weights
        self.bias -= self.lr * self.grad_bias


class SoftmaxCrossEntropyLoss(object):  # need to figure this out again
    def forward(self, logits, labels,softmax_p):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        self.logits = logits
        self.labels=labels # storing for accuracy
        self.batch_size = logits.shape[1]  #need to add loss here too!! not sure why need to figure

        
        # shifted = self.logits - np.max(x, axis=0, keepdims=True) 
        # exp_x = np.exp(shifted)
        # softmax = exp_x / np.sum(exp_x, axis=0, keepdims=True)

        self.softmax=softmax_p
        loss = -np.sum(labels * np.log(self.softmax + 1e-12)) / self.batch_size
        return loss



    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        return (self.softmax - self.labels)
    
    def getAccu(self):
        """
        return accuracy here
        """
        preds = np.argmax(self.softmax, axis=0)
        targets = np.argmax(self.labels, axis=0)
        return np.sum(preds == targets)



class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()

        self.layer1 = LinearMap(indim,hidden_layer,lr)
        self.relu = ReLU()
        self.layer2 = LinearMap(hidden_layer,outdim,lr)
        


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        q = self.layer1.forward(x)
        h=self.relu.forward(q)
        o=self.layer2.forward(h)
        


        return o
    


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        grad_layer2 = self.layer2.backward(grad_wrt_out)
        grad_relu = self.relu.backward(grad_layer2)
        grad_layer1 = self.layer1.backward(grad_relu)
        return grad_layer1
    

    
    def step(self):
        """update model parameters"""
        self.layer1.step()
        self.layer2.step()
        


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

#####################################################################################
# Implementation
#######################################################################################

if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """

    indim = 60
    outdim = 2
    hidden_dim = 32
    lr = 0.1
    batch_size = 32
    epochs = 500

    #dataset
    Xtrain = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\X_train.csv")
    Ytrain = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\X_test.csv")
    Ytest = pd.read_csv(r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1\data\y_test.csv")
    Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m2, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP( indim, outdim, hidden_dim, lr)
    criterion = SoftmaxCrossEntropyLoss()
    
    
    #construct the training process
    train_losses=[]
    train_accs=[]
    test_losses = []
    test_accs = []

    softmax=Softmax()

    for epoch in range(epochs):
        epoch_train_loss=0
        correct = 0
        total = 0

    # -------- TRAIN --------
        # if (epoch + 1) % 10 == 0:
        #     print("TRAINING!!!")
        for x, y in train_loader:
            x = x.numpy().T                          # (indim, batch)
            y = labels2onehot(y.numpy()).T           # (outdim, batch)

            # train 
            
            logits = model.forward(x)
            p = softmax.forward(logits)
            loss = criterion.forward(logits,y,p)

            epoch_train_loss += loss

            # backward
            grad_logits = criterion.backward()
            model.backward(grad_logits)

            # update
            model.step()

            # accuracy
            correct += criterion.getAccu()
            total += y.shape[1]

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # -------- TEST --------
        epoch_test_loss = 0
        test_correct = 0
        test_total = 0

        # if (epoch + 1) % 10 == 0:
            # print("TESTING!!!")
        for x, y in test_loader:
            x = x.numpy().T
            y_onehot = labels2onehot(y.numpy()).T

            # Forward pass only
            logits = model.forward(x)
            p = softmax.forward(logits)
            loss = criterion.forward(logits, y_onehot, p)

            epoch_test_loss += loss
            test_correct += criterion.getAccu()
            test_total += y_onehot.shape[1]

        avg_test_loss = epoch_test_loss / len(test_loader)
        avg_test_acc = test_correct / test_total
        test_losses.append(avg_test_loss)
        test_accs.append(avg_test_acc)

        # to check after every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
                    f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")


print("Training completed , Testing completed!!!!!")

save_dir = r"C:\Users\shail\Desktop\Shailja_everything\CMU_courses\NN_and_DL\HW1 (3)\HW1"
save_path = os.path.join(save_dir, "self_metrics.pkl")

with open(save_path, "wb") as f:
    pickle.dump((train_losses,  train_accs, test_losses, test_accs), f)
print(f"\nMetrics saved to: {save_path}")

##############################################3
##########################################