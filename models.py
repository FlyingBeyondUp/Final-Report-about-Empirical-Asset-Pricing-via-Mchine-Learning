# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 14:24:49 2025

@author: zy01
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:50:20 2025

@author: LZ166
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import HuberRegressor,SGDRegressor,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import xgboost as xgb
import numpy as np



 
# class myPLS(BaseEstimator, RegressorMixin):
#     def __init__(self, n_components=None):  # Provide a default value
#         self.n_components = n_components
#         # self.pca = PCA(n_components=self.n_components)
#         # self.regressor = HuberRegressor()
#         self.model=PLSRegression(n_components)
#         self.scaler=MinMaxScaler()

#     def fit(self, X_train, Y_train):
#         self.scaler.fit(X_train[:,:X_train.shape[1]-74])
#         X_scaled_train_real_valued=self.scaler.transform(X_train[:,:X_train.shape[1]-74])
#         X_scaled_train=np.concatenate((X_scaled_train_real_valued,X_train[:,X_train.shape[1]-74:]),axis=1)
#         # self.pca.fit(X_scaled_train)
#         # X_train_pca = self.pca.transform(X_scaled_train)  # Use transform, not fit_transform
#         # self.regressor.fit(X_train_pca, Y_train)\
#         self.model.fit(X_scaled_train,Y_train)
#         return self  # Important to return self

#     def predict(self, X_test):
#         X_scaled_test_real_valued=self.scaler.transform(X_test[:,:X_test.shape[1]-74])
#         X_scaled_test=np.concatenate((X_scaled_test_real_valued,X_test[:,X_test.shape[1]-74:]),axis=1)
# #         X_test_pca = self.pca.transform(X_scaled_test)  # Use the fitted PCA to transform
# # #         return self.regressor.predict(X_test_pca)
#         return self.model.predict(X_scaled_test)
    
class myPCR(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=None):  # Provide a default value
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.regressor = LinearRegression()

    def fit(self, X_train, Y_train):
        self.pca.fit(X_train)
        X_train_pca = self.pca.transform(X_train)  
        self.regressor.fit(X_train_pca, Y_train)
        return self  # Important to return self

    def predict(self, X_test):
        X_test_pca = self.pca.transform(X_test)  # Use the fitted PCA to transform
        return self.regressor.predict(X_test_pca)

class myENet(BaseEstimator, RegressorMixin):
    def __init__(self,alpha,l1_ratio=0.5,epsilon=1.35,max_iter=100):
        self.model=SGDRegressor(
        loss='huber',        # Huber‐loss
        penalty='elasticnet',# L1+L2 penalty
        alpha=alpha,          # overall regularization strength
        l1_ratio=l1_ratio,        # fraction of L1 in the penalty
        epsilon=epsilon,        # Huber transition point
        max_iter=max_iter,
        warm_start=True,
        learning_rate='optimal',
        tol=1e-4,
        random_state=0)
    
    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train)
        nonzero_mask = np.abs(self.model.coef_) > 1e-5
        self.complexity=nonzero_mask.sum()
    
    def predict(self,X):
        return self.model.predict(X)

class myRF(BaseEstimator, RegressorMixin):
    def __init__(self,max_depth,max_features,n_estimators=300,n_jobs=12):
        self.model=RandomForestRegressor(max_depth=max_depth,max_features=max_features,
                                         n_estimators=n_estimators,n_jobs=n_jobs)
    
    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train)
        depths = np.array([tree.get_depth() for tree in self.model.estimators_])
        self.complexity=depths.mean()
    
    def predict(self,X):
        return self.model.predict(X)

class myGBRT(BaseEstimator, RegressorMixin):
    def __init__(self,max_depth,lr,n_estimators=200):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',  # or another regression objective
            n_estimators=n_estimators,  # Number of boosting rounds
            learning_rate=lr,
            max_depth=2,
            tree_method='hist',  # or 'approx', or 'exact' - Use CPU-based histogram algorithm or other CPU methods
            #device='cpu', # Remove or comment out this line; CPU is the default if tree_method is not 'gpu_hist'
            random_state=42
        )
    
    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train)
        bst = self.model.get_booster()
        fscore = bst.get_score(importance_type='weight')  
        # e.g. {'f12': 23, 'f3': 8, …}

        self.complexity = len(list(fscore.keys()))
    def predict(self,X):
        return self.model.predict(X)

    
    




class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, optimizer=optim.Adam, lr=0.001, epochs=100, batch_size=1e4, l1_ratio=0.1,criterion=nn.MSELoss):
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.l1_ratio=l1_ratio
        self.batch_size = int(batch_size)
        self.criterion = criterion()  # Instantiate the loss function
        self.optimizer_ = None  # Will be initialized in fit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
        self.scaler=StandardScaler()

    def fit(self, X, y):

        self.model.to(self.device)
        self.optimizer_ = self.optimizer(self.model.parameters(), lr=self.lr)

        self.model.train()  # Set the model to training mode

        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                end_i = int(min(i + self.batch_size, X.shape[0]))
                X_batch = torch.tensor(X[i:end_i,:],dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y[i:end_i],dtype=torch.float32).to(self.device)

                self.optimizer_.zero_grad()  # Clear gradients
                outputs = self.model(X_batch).squeeze() # Remove extra dimension
                loss = self.criterion(outputs, y_batch)
                l1_loss = 0
                for param in self.model.parameters():
                    l1_loss += torch.abs(param).sum()
                    
                loss=loss+self.l1_ratio*l1_loss
                loss.backward()  # Backpropagation
                self.optimizer_.step()  # Update weights

        return

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation during inference
            outputs = self.model(X)
        return outputs.cpu().numpy().flatten() # Flatten the output
    
class NN1(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

        self.initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

class NN2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)

        self.initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)

class NN3(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(8, 1)  # Corrected fc3 to fc4

        self.initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc2(x)
        out = self.bn2(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc3(x)
        out = self.bn3(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc4(x) # Corrected fc3 to fc4
        return out

    def initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.fc4.bias)

class NN4(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(8, 4)
        self.bn4 = nn.BatchNorm1d(4)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(4, 1)  # Corrected fc4 to fc5

        self.initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc4(out)
        out = self.bn4(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc5(out)  # Corrected fc4 to fc5
        return out

    def initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.fc4.bias)
        init.zeros_(self.fc5.bias)

class NN5(nn.Module):
    def __init__(self, input_size):
        super().__init__()  # Corrected super() call
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(8, 4)
        self.bn4 = nn.BatchNorm1d(4)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(4, 2)  # Corrected fc4 to fc5
        self.bn5 = nn.BatchNorm1d(2)  # Added batch norm
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(2, 1)  # Corrected fc4 to fc6

        self.initialize_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc2(x)
        out = self.bn2(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc3(x)
        out = self.bn3(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc4(x)
        out = self.bn4(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc5(x)  # Corrected fc4 to fc5
        out = self.bn5(out)  # Apply batch norm
        out = self.relu(out)
        out = self.fc6(x)  # Corrected fc4 to fc6
        return out

    def initialize_weights(self):
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc6.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.fc4.bias)
        init.zeros_(self.fc5.bias)
        init.zeros_(self.fc6.bias)
    
class myElasticNet(BaseEstimator, RegressorMixin):
    """
    Elastic Net regression using PyTorch. Mimics the scikit-learn API.
    Handles data batch-wise on the specified device without shuffling.

    Parameters:
        alpha (float): Overall regularization strength.
        l1_ratio (float): Mixing parameter (0 for L2, 1 for L1).
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        tol (float): Tolerance for loss change to declare convergence.
        delta (float): Delta parameter for the Huber loss.
        random_state (int): Random seed for reproducibility of weight initialization.
        device (str): 'cpu' or 'cuda' or specific cuda device like 'cuda:0'.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, epochs=100, batch_size=10000, tol=1e-2, delta=1.0, random_state=None, device='cuda'):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.delta = delta # Store delta for Huber loss
        self.random_state = random_state
        self.device = device

    def _huber_loss(self, y_predicted, y_true):
        """ Compute the Huber loss. """
        residual = y_true - y_predicted # Order: (true - predicted)
        abs_residual = torch.abs(residual)
        # Ensure delta is a tensor on the correct device
        delta_tensor = torch.tensor(self.delta, device=y_predicted.device, dtype=y_predicted.dtype)
        condition = abs_residual < delta_tensor
        squared_loss = 0.5 * residual**2
        linear_loss = delta_tensor * (abs_residual - 0.5 * delta_tensor)
        return torch.mean(torch.where(condition, squared_loss, linear_loss))

    def fit(self, X, y):
        """
        Fit the Elastic Net model without shuffling data.

        Parameters:
            X (array-like): Training data features (time series).
            y (array-like): Training data target values (time series).

        Returns:
            self: Fitted estimator.
        """
        self.n_features_in_ = X.shape[1]
        # self.scaler_ = StandardScaler() # Optional scaling
        # X_scaled = self.scaler_.fit_transform(X)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(self.random_state) # Seed all GPUs if applicable
            
        # ---------------------------------------

        # Define the model and move it to the target device
        self.model_ = nn.Linear(self.n_features_in_, 1).to(self.device)

        # Define the optimizer
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self.prev_loss_ = float('inf') # Initialize previous loss for convergence check

        # Training loop
        for epoch in range(self.epochs):
            self.model_.train() # Set model to training mode
            epoch_loss = 0.0
            num_batches = 0

            # --- Iterate through data sequentially (no shuffling) ---
            for i in range(0, X.shape[0], self.batch_size):
                # Define the end index for the batch
                end_i = int(min(i + self.batch_size, X.shape[0]))

                # --- Move only the CURRENT BATCH to GPU ---
                X_batch = torch.tensor(X[i:end_i,:],dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y[i:end_i],dtype=torch.float32).to(self.device)
                # ------------------------------------------

                # Forward pass
                outputs = self.model_(X_batch)
                huber_loss = self._huber_loss(outputs, y_batch) # Pass (predicted, true)

                # Elastic Net regularization
                # Calculate penalties based on model parameters on the correct device
                l1_penalty = self.alpha * self.l1_ratio * sum(p.abs().sum() for p in self.model_.parameters())
                # Standard L2 includes a 0.5 factor often, adjust if needed based on definition
                l2_penalty = self.alpha * (1 - self.l1_ratio) * 0.5 * sum(p.pow(2).sum() for p in self.model_.parameters())
                loss = huber_loss + l1_penalty + l2_penalty

                # Backward and optimize
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()

                epoch_loss += loss.item() # Accumulate loss for the epoch
                num_batches += 1
            # --- End of sequential batch iteration ---

            # Check for convergence (optional - based on average epoch loss change)
            current_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

            if epoch > 0 and abs(current_avg_loss - self.prev_loss_) < self.tol:
                print(f"Converged at epoch {epoch} with average loss {current_avg_loss:.4f}")
                break # Exit training loop

            self.prev_loss_ = current_avg_loss

            # Optional: Print progress every N epochs
            # if (epoch + 1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{self.epochs}], Avg Loss: {current_avg_loss:.4f}')

        # Use for-else to print if loop completes without breaking (convergence)
        else:
            print(f"Finished {self.epochs} epochs without converging by tolerance. Final avg loss: {self.prev_loss_:.4f}")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        # Predict using the fitted Elastic Net model.
        
        # X = self.scaler_.transform(X) # Scale the data using the fitted scaler
    
        self.model_=self.model_.to('cpu')
        with torch.no_grad():
            predictions = self.model_(torch.tensor(X,dtype=torch.float32)).cpu().numpy().flatten()
        return predictions
