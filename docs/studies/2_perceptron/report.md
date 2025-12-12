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

Used to orquestrate the training of a new perceptron based in core variables used to generate classes (mean, covariance)
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