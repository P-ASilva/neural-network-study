# Study 1 – Exploring Class Separability in 2D

## 1. Data Generation

Generated a synthetic dataset with 400 samples (100 per class) using Gaussian distributions defined by the given means and standard deviations:


- **Class 0:** Mean = [2, 3], Std = [0.8, 2.5]  

- **Class 1:** Mean = [5, 6], Std = [1.2, 1.9]  

- **Class 2:** Mean = [8, 1], Std = [0.9, 0.9]  

- **Class 3:** Mean = [15, 4], Std = [0.5, 2.0]  

### Code Snipet

Code used for class generation, present in `main.py` 

```python
    class_params = {
        0: {"mean": [2, 3], "std": [0.8, 2.5]},
        1: {"mean": [5, 6], "std": [1.2, 1.9]},
        2: {"mean": [8, 1], "std": [0.9, 0.9]},
        3: {"mean": [15, 4], "std": [0.5, 2.0]},
    }
    
    X, y = [], []
    for cls, p in class_params.items():
        data = np.random.normal(loc=p["mean"], scale=p["std"], size=(samples_per_class, 2))
        X.append(data)
        y.append(np.full(samples_per_class, cls))

    X = np.vstack(X)
    y = np.hstack(y)
    
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Scatter Plot generation, next topic
    
    return {
        "samples_per_class": samples_per_class,
        "num_classes": len(class_params),
        "class_params": class_params,
        "scatter_plot_path": "assets/scatter.png"
    }
```
## 2. Visualization: Scatter Plot

![scatter](assets/scatter.png)

### Code Snippet

Code used for image/graphic generation.

```python
    plt.figure(figsize=(10, 6))
    colors = ["red", "blue", "green", "orange"]
    for cls in class_params.keys():
        plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"Class {cls}", alpha=0.6)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Synthetic 2D Data for 4 Classes")
    plt.legend()
    plt.grid(True)
    
    scatter_path = assets_dir / "scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
```
## 3. Analysis and Decision Boundaries

### a. Distribution and Overlap
- **Class 0** is spread vertically due to a large standard deviation in the y-axis.  
- **Class 1** clusters around (5,6), moderately spread.  
- **Class 2** lies near the bottom-center region, concentrated.
- **Class 3** is far apart on the right side, with significant vertical spread.  
- There is some overlap between Classes 0 and 1, and between Classes 1 and 2.  
- Class 3 is clearly separable from the others due to its distance.

### b. Linear Separability
A single global **linear boundary cannot perfectly separate all classes**, because Classes 0, 1, and 2 overlap. However, piecewise linear or nonlinear decision boundaries could achieve good separation.

### c. Decision Boundaries
A neural network would likely:
- Draw **nonlinear curved boundaries** between Classes 0, 1 and 2.  
- Use a **clear vertical cut** to separate Class 3 from the others.

As such, a trained model would need at least moderately complex decision boundaries to classify all four classes correctly.