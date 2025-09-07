import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- funciones usadas en EDA previo ---
def hist(s, title="", bins=50, xlabel="", ylabel="freq"):
    plt.figure()
    pd.Series(s).dropna().plot(kind="hist", bins=bins)
    plt.title(title)
    plt.xlabel(xlabel or s.name or "")
    plt.ylabel(ylabel)
    plt.tight_layout()

def bar_series(s, title="", xlabel="", ylabel="", rotate=0):
    s = pd.Series(s)
    plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate)
    plt.tight_layout()

def scatter_xy(df, x, y, title="", alpha=0.5):
    plt.figure()
    plt.scatter(df[x], df[y], alpha=alpha)
    plt.title(title)
    plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout()

def corr_heatmap(df_num, title="Correlación (Pearson)"):
    import numpy as np
    import matplotlib.pyplot as plt
    corr = df_num.corr()
    plt.figure()
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()

# --- visualizaciones extra para no supervisado ---
def line_xy(x, y, title="", xlabel="", ylabel=""):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def bar_counts(series, title="", xlabel="", ylabel="count", rotate=0, top=None):
    s = pd.Series(series)
    if top is not None:
        s = s.sort_values(ascending=False).head(top)
    plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate)
    plt.tight_layout()

def scatter_2d(X2, labels=None, title="", alpha=0.5):
    plt.figure()
    if labels is None:
        plt.scatter(X2[:,0], X2[:,1], alpha=alpha)
    else:
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            plt.scatter(X2[idx,0], X2[idx,1], alpha=alpha, label=str(lab))
        plt.legend(title="grupo", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.xlabel("comp-1"); plt.ylabel("comp-2")
    plt.tight_layout()

def degree_hist(G, title="Degree distribution"):
    deg = [d for _, d in G.degree()]
    plt.figure()
    plt.hist(deg, bins=30)
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("freq")
    plt.tight_layout()

