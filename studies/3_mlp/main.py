import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from mlp import MLP, generate_synthetic_data

def manual_mlp_calculation():
    x = np.array([0.5, -0.2])
    y = 1.0
    W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
    b1 = np.array([0.1, -0.2])
    W2 = np.array([0.5, -0.3])
    b2 = 0.2
    eta = 0.3
    
    z1 = W1 @ x + b1
    h1 = np.tanh(z1)
    u2 = W2 @ h1 + b2
    y_hat = np.tanh(u2)
    loss = (y - y_hat) ** 2
    
    dL_dyhat = -2 * (y - y_hat)
    dL_du2 = dL_dyhat * (1 - np.tanh(u2)**2)
    dL_dW2 = dL_du2 * h1
    dL_db2 = dL_du2
    
    dL_dh1 = dL_du2 * W2
    dL_dz1 = dL_dh1 * (1 - np.tanh(z1)**2)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1
    
    W2_new = W2 - eta * dL_dW2
    b2_new = b2 - eta * dL_db2
    W1_new = W1 - eta * dL_dW1
    b1_new = b1 - eta * dL_db1
    
    return {
        'z1_1': z1[0], 'z1_2': z1[1],
        'h1_1': h1[0], 'h1_2': h1[1],
        'u2': u2, 'y_hat': y_hat, 'loss': loss,
        'dL_dyhat': dL_dyhat, 'dL_du2': dL_du2,
        'dL_dW2_1': dL_dW2[0], 'dL_dW2_2': dL_dW2[1],
        'dL_db2': dL_db2,
        'dL_dh1_1': dL_dh1[0], 'dL_dh1_2': dL_dh1[1],
        'dL_dz1_1': dL_dz1[0], 'dL_dz1_2': dL_dz1[1],
        'dL_dW1_11': dL_dW1[0,0], 'dL_dW1_12': dL_dW1[0,1],
        'dL_dW1_21': dL_dW1[1,0], 'dL_dW1_22': dL_dW1[1,1],
        'dL_db1_1': dL_db1[0], 'dL_db1_2': dL_db1[1],
        'W2_new_1': W2_new[0], 'W2_new_2': W2_new[1],
        'b2_new': b2_new,
        'W1_new_11': W1_new[0,0], 'W1_new_12': W1_new[0,1],
        'W1_new_21': W1_new[1,0], 'W1_new_22': W1_new[1,1],
        'b1_new_1': b1_new[0], 'b1_new_2': b1_new[1]
    }

def run_experiments():
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Experiment 2: Binary Classification
    X_bin, y_bin = generate_synthetic_data(n_samples=1000, n_classes=2, n_features=2, 
                                         clusters_per_class=[1, 2], random_state=42)
    mlp_binary = MLP(input_size=2, hidden_sizes=[4], output_size=1, activation='relu')
    binary_results = mlp_binary.train(X_bin, y_bin, epochs=100, learning_rate=0.01)
    
    # Experiment 3: Multi-Class Classification
    X_multi, y_multi = generate_synthetic_data(n_samples=1500, n_classes=3, n_features=4,
                                              clusters_per_class=[2, 3, 4], random_state=42)
    mlp_multi = MLP(input_size=4, hidden_sizes=[4], output_size=3, activation='relu')
    multi_results = mlp_multi.train(X_multi, y_multi, epochs=100, learning_rate=0.01)
    
    # Experiment 4: Deep MLP
    mlp_deep = MLP(input_size=4, hidden_sizes=[8, 4], output_size=3, activation='relu')
    deep_results = mlp_deep.train(X_multi, y_multi, epochs=100, learning_rate=0.01)
    
    # Generate plots
    plt.figure(figsize=(10, 6))
    plt.plot(binary_results['loss_history'], label='Binary')
    plt.plot(multi_results['loss_history'], label='Multi-class')
    plt.plot(deep_results['loss_history'], label='Deep MLP')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = assets_dir / "loss_comparison.png"
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Accuracy comparison plot
    plt.figure(figsize=(10, 6))
    epochs = range(len(binary_results['loss_history']))
    plt.plot(epochs, [binary_results['train_accuracy']] * len(epochs), label='Binary Train')
    plt.plot(epochs, [binary_results['test_accuracy']] * len(epochs), '--', label='Binary Test')
    plt.plot(epochs, [multi_results['train_accuracy']] * len(epochs), label='Multi-class Train')
    plt.plot(epochs, [multi_results['test_accuracy']] * len(epochs), '--', label='Multi-class Test')
    plt.plot(epochs, [deep_results['train_accuracy']] * len(epochs), label='Deep MLP Train')
    plt.plot(epochs, [deep_results['test_accuracy']] * len(epochs), '--', label='Deep MLP Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    accuracy_plot_path = assets_dir / "accuracy_comparison.png"
    plt.savefig(accuracy_plot_path)
    plt.close()
    
    return {
        'binary_hidden_layers': 1,
        'binary_neurons_per_layer': [4],
        'binary_activation': 'relu',
        'binary_loss': 'MSE',
        'binary_learning_rate': 0.01,
        'binary_final_loss': binary_results['loss_history'][-1],
        'binary_train_accuracy': binary_results['train_accuracy'],
        'binary_test_accuracy': binary_results['test_accuracy'],
        'binary_epochs': len(binary_results['loss_history']),
        'binary_boundary_plot': 'assets/binary_boundary.png',
        
        'multiclass_hidden_layers': 1,
        'multiclass_neurons_per_layer': [4],
        'multiclass_activation': 'relu',
        'multiclass_loss': 'Categorical Cross-Entropy',
        'multiclass_learning_rate': 0.01,
        'multiclass_final_loss': multi_results['loss_history'][-1],
        'multiclass_train_accuracy': multi_results['train_accuracy'],
        'multiclass_test_accuracy': multi_results['test_accuracy'],
        'multiclass_epochs': len(multi_results['loss_history']),
        'multiclass_pca_plot': 'assets/multiclass_pca.png',
        
        'deep_hidden_layers': 2,
        'deep_neurons_per_layer': [8, 4],
        'deep_activation': 'relu',
        'deep_loss': 'Categorical Cross-Entropy',
        'deep_learning_rate': 0.01,
        'deep_final_loss': deep_results['loss_history'][-1],
        'deep_train_accuracy': deep_results['train_accuracy'],
        'deep_test_accuracy': deep_results['test_accuracy'],
        'deep_epochs': len(deep_results['loss_history']),
        
        'loss_comparison_plot': 'assets/loss_comparison.png',
        'accuracy_comparison_plot': 'assets/accuracy_comparison.png'
    }

def main():
    manual_results = manual_mlp_calculation()
    experiment_results = run_experiments()
    
    return {**manual_results, **experiment_results}

if __name__ == "__main__":
    result = main()
    print("MLP study data generated successfully")