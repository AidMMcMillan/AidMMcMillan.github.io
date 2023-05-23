from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

def plot_regions(model, X, y):
    
    x0 = X[X.columns[0]]
    x1 = X[X.columns[1]]
    qual_features = X.columns[2:]
    
    fig, axarr = plt.subplots(1, len(qual_features), figsize = (7, 3))

    x0_min = x0.min()-1
    x0_max = x0.max()+1
    x1_min = x1.min()-0.5
    x1_max = x1.max()+0.5
    
    
    # create a grid
    grid_x = np.linspace(x0_min,x0_max,501)
    grid_y = np.linspace(x1_min,x1_max,501)
    xx, yy = np.meshgrid(grid_x, grid_y)
    
    XX = xx.ravel()
    YY = yy.ravel()

    for i in range(len(qual_features)):
        axarr[i].set_xlim([x0_min, x0_max])
        axarr[i].set_ylim([x1_min, x1_max])
        
        XY = pd.DataFrame({
            X.columns[0] : XX,
            X.columns[1] : YY
        })
        
        for j in qual_features:
            XY[j] = 0
        
        XY[qual_features[i]] = 1

        p = model.predict(XY)
        p = p.reshape(xx.shape)
        
        # use contour plot to visualize the predictions
        axarr[i].contourf(xx, yy, p, cmap = "jet", alpha = 0.2, vmin = 0, vmax = 2)
        
        ix = X[qual_features[i]] == 1
        
        # plot the data
        axarr[i].scatter(x0[ix], x1[ix], c = y[ix], cmap = "jet", vmin = 0, vmax = 2)
        
        axarr[i].set(xlabel = X.columns[0], 
                     ylabel  = X.columns[1])
        
        patches = []
        for color, spec in zip(["red", "green", "blue"], ["Adelie", "Chinstrap", "Gentoo"]):
            patches.append(Patch(color = color, label = spec))
        
        plt.legend(title = "Species", handles = patches, loc = "best")
        
        plt.tight_layout()
