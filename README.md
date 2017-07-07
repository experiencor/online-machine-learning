
This is the implementation of the Follow the Regularized Leader with Adaptive Decaying Proximal algorithm [to be published]. Please refer to https://experiencor.github.io/ftrl_adp.html for details.

# Code structure

1. ftrl_adp.py => the implementation of FTRL-ADP
2. Algorithm Visualization.ipynb => visualization that shows the online learning process of FTRL-ADP
3. dataset.txt => a simulated classification dataset that contains dynamic concept drifting behavior

# Usage

cd ftrl_adp

```python
from ftrl_adp import FTRL_ADP

X_input, Y_label = load_svmlight_file('dataset.txt')

ftrl_adp = FTRL_ADP(L1=1., L2=1., LP = 1., adaptive=True, n_inputs=X_input.shape[1])

for i in xrange(X_input.shape[0]):
    indices = X_input[row].indices
    x = X_input[row].data
    y = y_label[i]
    
    p, decay = classifier.fit(indices, x, y)
    error = [int(np.abs(y-p)>0.5)]
```
