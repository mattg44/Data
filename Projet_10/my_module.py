# Personal functions for data analysis
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency as chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA

def search_uniqueness(df):
    """
    Function checking uniqueness of the variables in a dataframe
    Input: dataframe (df)
    Output: Uniqueness status or repetitiveness value
    January 2022
    """

    for col in df.columns:
        nb_dup = df[col].duplicated().sum()
        if nb_dup == 0:
            print(f"{col}: uniqueness checked, potential primary key")
        else:
            print(f"The variable {col} is repeated {nb_dup} times")


def search_cardinality(df):
    """
    Function looking for unique values / cardinality for each variable of a dataframe
    Input: dataframe (df)
    Output: unique values list if <= 10 or cardinality if > 10
    January 2022
    """

    for col in df.columns:
        print(col)
        nb_value = df[col].nunique()
        if nb_value > 10:
            print(f"Cardinality - The variable {col} has {nb_value} distinct values.")
            print("-" * 80)
        else:
            print(df[col].unique())
            print("-" * 80)


def drop_outliers(df, variable):
    """
    Remove outliers form a dataframe with the interquartile method (1.5 * IRQ) based on a given variable/column
    Input: dataframe (df) and variable ("variable" - quotes required)
    Output: dataframe without outlier
    January 2022
    """

    df_no_outlier = df.copy()
    q3, q1 = np.percentile(df_no_outlier[variable], [75, 25])
    irq = q3 - q1
    df_no_outlier.drop(
        df_no_outlier[df_no_outlier[variable] > q3 + 1.5 * irq].index, inplace=True
    )
    df_no_outlier.drop(
        df_no_outlier[df_no_outlier[variable] < q1 - 1.5 * irq].index, inplace=True
    )
    return df_no_outlier


def remove_duplicates(df):
    """
    Drop duplicates and print status
    Input: dataframe (df)
    Output: dataframe without duplicates
    January 2022
    """
    
    duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"{duplicates} duplicates have been removed")

    
def cramers_v(x, y):
    """
    Source: https://scribe.rip/the-search-for-categorical-correlation-a1cf7f1888c9
    Calculate the strenght between 2 categorical variables according Cramér’s V
    Input: 2 categorical variables
    Output: Value between 0 (low) to 1 (high) evaluating the strenght of the association
    """
    
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    p_value = chi2_contingency(confusion_matrix)[1]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def gini(v):
    """
    Calculation of the gini coefficient
    Input: the cumsum serie
    Output: print of the gini coeff
    """
    bins = np.linspace(0, 100, 100)
    total = np.sum(v)
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / pe_area
    print(f"Gini value: {gini_val:.4f}") 

    
    
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    '''
    Cercle des corrélations
    Auteur : Openclassrooms
    '''
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(10,10))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
   


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''
    Scatter plot
    Auteur : Openclassrooms
    '''
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(12,10))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

          
        
def display_scree_plot(pca):
    '''
    Eboulis des valeurs propres (diagramme de Pareto)
    Auteur : Openclassrooms
    '''
    plt.figure(figsize=(10, 8))
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

    
    
def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()