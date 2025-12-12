import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mlp import MLP, generate_synthetic_data

def run_binary_classification():
    """Run Experiment 2: Binary Classification"""
    X, y = generate_synthetic_data(n_samples=1000, n_classes=2, n_features=2, 
                                 clusters_per_class=[1, 2], random_state=42)
    
    mlp = MLP(input_size=2, hidden_sizes=[4], output_size=1, activation='tanh')
    results = mlp.train(X, y, epochs=100, learning_rate=0.01)
    
    # Generate decision boundary plot
    plt.figure(figsize=(8, 6))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Binary Classification Decision Boundary')
    plt.savefig('./docs/studies/3_mlp/assets/binary_boundary.png', dpi=300)
    plt.close()
    
    return results

def run_multiclass_classification():
    """Run Experiment 3: Multi-Class Classification"""
    X, y = generate_synthetic_data(n_samples=1500, n_classes=3, n_features=4,
                                 clusters_per_class=[2, 3, 4], random_state=42)
    
    mlp = MLP(input_size=4, hidden_sizes=[4], output_size=3, activation='relu')
    results = mlp.train(X, y, epochs=100, learning_rate=0.01)
    
    # Generate PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Multi-class PCA Visualization')
    plt.savefig('./docs/studies/3_mlp/assets/multiclass_pca.png', dpi=300)
    plt.close()
    
    return results

def run_deep_mlp():
    """Run Experiment 4: Deep MLP"""
    X, y = generate_synthetic_data(n_samples=1500, n_classes=3, n_features=4,
                                 clusters_per_class=[2, 3, 4], random_state=42)
    
    mlp = MLP(input_size=4, hidden_sizes=[8, 4], output_size=3, activation='tanh')
    results = mlp.train(X, y, epochs=100, learning_rate=0.01)
    
    return results

if __name__ == "__main__":
    # Create assets directory
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    print("Running binary classification experiment...")
    binary_results = run_binary_classification()
    
    print("Running multi-class classification experiment...")
    multiclass_results = run_multiclass_classification()
    
    print("Running deep MLP experiment...")
    deep_results = run_deep_mlp()
    
    print("All experiments completed successfully!")
    print(f"Binary - Train Acc: {binary_results['train_accuracy']:.2f}%, Test Acc: {binary_results['test_accuracy']:.2f}%")
    print(f"Multi-class - Train Acc: {multiclass_results['train_accuracy']:.2f}%, Test Acc: {multiclass_results['test_accuracy']:.2f}%")
    print(f"Deep MLP - Train Acc: {deep_results['train_accuracy']:.2f}%, Test Acc: {deep_results['test_accuracy']:.2f}%")