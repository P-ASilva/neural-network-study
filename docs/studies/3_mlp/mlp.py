import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

def generate_synthetic_data(n_samples=1000, n_classes=2, n_features=2, 
                          clusters_per_class=None, random_state=42):
    if clusters_per_class is None:
        clusters_per_class = [1] * n_classes
    
    X_parts = []
    y_parts = []
    samples_per_class = n_samples // n_classes
    
    for class_idx, n_clusters in enumerate(clusters_per_class):
        if n_clusters > 1:
            # Generate multi-cluster class with equal cluster sizes
            cluster_size = samples_per_class // n_clusters
            X_class_parts = []
            
            for cluster_idx in range(n_clusters):
                X_cluster, y_cluster = make_classification(
                    n_samples=cluster_size,
                    n_features=n_features,
                    n_informative=n_features,
                    n_redundant=0,
                    n_clusters_per_class=1,
                    n_classes=1,
                    random_state=random_state + class_idx * 10 + cluster_idx
                )
                # Shift cluster to separate them
                X_cluster += np.random.normal(cluster_idx * 3, 0.8, (cluster_size, n_features))
                X_class_parts.append(X_cluster)
            
            X_class = np.vstack(X_class_parts)
            # Add remaining samples if division wasn't perfect
            if len(X_class) < samples_per_class:
                extra_samples = samples_per_class - len(X_class)
                X_extra, _ = make_classification(
                    n_samples=extra_samples,
                    n_features=n_features,
                    n_informative=n_features,
                    n_redundant=0,
                    n_clusters_per_class=1,
                    n_classes=1,
                    random_state=random_state + class_idx * 10 + n_clusters
                )
                X_class = np.vstack([X_class, X_extra])
                
        else:
            # Generate single cluster class
            X_class, y_class = make_classification(
                n_samples=samples_per_class,
                n_features=n_features,
                n_informative=n_features,
                n_redundant=0,
                n_clusters_per_class=1,
                n_classes=1,
                random_state=random_state + class_idx
            )
        
        X_parts.append(X_class)
        y_parts.append(np.full(len(X_class), class_idx))
    
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    
    return X, y

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh'):
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros(layer_sizes[i+1]))
    
    def activate(self, x):
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
    
    def activate_derivative(self, x):
        if self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == 'relu':
            return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            current = self.activate(z)
            self.activations.append(current)
        
        return current
    

    def compute_loss(self, y_pred, y_true):
        """
        Softmax Cross-Entropy Loss.
        y_pred: predicted probabilities per class (softmax)
        y_true: one-hot true labels
        """
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # sum over classes, mean over samples
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    
    def backward(self, X, y_true, y_pred, learning_rate):
        m = X.shape[0]
        dZ = 2 * (y_pred - y_true) / m  # MSE derivative
        
        for i in reversed(range(len(self.weights))):
            dW = self.activations[i].T @ dZ
            dB = np.sum(dZ, axis=0)
            
            if i > 0:
                dA = dZ @ self.weights[i].T
                dZ = dA * self.activate_derivative(self.z_values[i-1])
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB
    
    def train(self, X, y, epochs=100, learning_rate=0.01, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                           random_state=42)
        
        # Ensure y is 2D for matrix operations
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        
        loss_history = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_pred, y_train)
            loss_history.append(loss)
            
            # Backward pass
            self.backward(X_train, y_train, y_pred, learning_rate)
        
        # Calculate final accuracies
        train_acc = self.accuracy(X_train, y_train.flatten())
        test_acc = self.accuracy(X_test, y_test.flatten())
        
        return {
            'loss_history': loss_history,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
    
    def accuracy(self, X, y):
        y_pred = self.forward(X)
        if y_pred.shape[1] == 1:  # Binary classification
            predictions = (y_pred > 0.5).astype(int).flatten()
        else:  # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y) * 100