# Study 2

Separated in two scenarios for different approaches, before discussing the result we must highlight the setup.

## Perceptron Training

```python
def perceptron_train(X, y, eta=0.01, max_epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    accuracies = []

    for epoch in range(max_epochs):
        errors = 0
        for xi, target in zip(X, y):
            if target * (np.dot(w, xi) + b) <= 0:
                w += eta * target * xi
                b += eta * target
                errors += 1
        acc = np.mean(np.sign(np.dot(X, w) + b) == y)
        accuracies.append(acc)
        if errors == 0:
            break
    return w, b, accuracies
```
## Study/Scenario Builder

Used to orchestrate the training of a new perceptron based in core variables used to generate classes (mean, covariance)
```python
def run_perceptron_study(mean0, cov0, mean1, cov1, study_name):
    np.random.seed(42)
    class0 = np.random.multivariate_normal(mean0, cov0, 1000)
    class1 = np.random.multivariate_normal(mean1, cov1, 1000)
    X = np.vstack((class0, class1))
    y = np.hstack((-1*np.ones(1000), np.ones(1000)))

    w, b, accuracies = perceptron_train(X, y)
    
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    boundary_path, accuracy_path = generate_plots(X, y, w, b, accuracies, assets_dir, study_name.lower().replace(" ", "_"))

    return {
        "final_weights": w,
        "final_bias": b,
        "final_accuracy": accuracies[-1] * 100,
        "epochs_used": len(accuracies),
        "boundary_plot_path": boundary_path,
        "accuracy_plot_path": accuracy_path
    }
```
The `run_perceptron_study` can be used directly, in this instance it was used through a `main()` in order to generate data for both of the following experiments:

# Study 2A - Linear Separability Analysis

**Final Weights:** [0.01985622, 0.01711828]  
**Final Bias:** -0.1200  
**Final Accuracy:** 100.00%  
**Epochs until convergence:** 12

### Analysis
The perceptron converged quickly because the data is linearly separable. Clusters are compact and far apart, so the decision boundary is learned in few epochs.

![Decision Boundary](assets/study_2a_boundary.png)  
![Accuracy Curve](assets/study_2a_accuracy.png)

---

# Study 2B - Non-Linear Separability Challenge

**Final Weights:** [0.01568527, 0.04336965]  
**Final Bias:** -0.0300  
**Final Accuracy:** 50.15%  
**Epochs until convergence:** 100

### Analysis
Here, the means are closer and the variance is higher, causing overlap between classes. This prevents perfect linear separation, so the perceptron may not converge to 100% accuracy. Training may oscillate or plateau, highlighting the model's limitation with non-separable data.

![Decision Boundary](assets/study_2b_boundary.png)  
![Accuracy Curve](assets/study_2b_accuracy.png)


#### Graph generation code:
```python
def generate_plots(X, y, w, b, accuracies, assets_dir, prefix):
    plt.figure(figsize=(8,6))
    plt.scatter(X[y==-1,0], X[y==-1,1], color="red", label="Class 0")
    plt.scatter(X[y==1,0], X[y==1,1], color="blue", label="Class 1")

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0]*x_vals + b)/w[1]
    plt.plot(x_vals, y_vals, 'k--', label="Decision boundary")

    preds = np.sign(np.dot(X, w) + b)
    misclassified = X[preds != y]
    if len(misclassified) > 0:
        plt.scatter(misclassified[:,0], misclassified[:,1],
                    facecolors='none', edgecolors='k', s=100, label="Misclassified")

    plt.legend()
    plt.title(f"{prefix} Decision Boundary")
    boundary_path = assets_dir / f"{prefix}_boundary.png"
    plt.savefig(boundary_path)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} - Training Accuracy")
    accuracy_path = assets_dir / f"{prefix}_accuracy.png"
    plt.savefig(accuracy_path)
    plt.close()

    return f"assets/{prefix}_boundary.png", f"assets/{prefix}_accuracy.png"
```