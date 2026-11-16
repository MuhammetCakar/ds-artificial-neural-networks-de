import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    """
    Berechnet die Sigmoid-Aktivierung
    
        Args:
        Z: Eingabewerte als numpy-Wert
        
        Returns:
        Werte der Sigmoidfunktion
    """
    
    return 1 / (1 + np.exp(-Z))


def visualize_decision_boundary(X, Y, model, padding=.5, step_size=.01):
    """
    Visualisiert die Grenze eines Modells im 2D-Feature-Raum
    
        Args:
        X:         Features (n_samples, n_features)
        Y:         tatsächliche Zielwerte (n_samples, 1)
        model:     Vorhersagefunktion eines Modells
        padding:   Rand um den dargestellten Bereich
        step_size: Schrittweite für das Vorhersagegitter
        
        Returns:
        Streudiagramm der Features mit den tatsächlichen Klassenlabels
        und der Entscheidungsgrenze des Modells
    """
    # Grenzen des Feature-Raums für die Darstellung festlegen
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
        
    # Punktgitter im Feature-Raum mit der Schrittweite
    # 'step_size' zwischen den Punkten erzeugen
    x1, x2 = np.meshgrid(np.arange(x_min, x_max, step_size), 
                         np.arange(y_min, y_max, step_size))
    
    # Mit der 'predict'-Funktion des Modells die Vorhersagewerte
    # für alle Gitterpunkte berechnen
    z = model(np.c_[x1.ravel(), x2.ravel()])
    z = z.reshape(x1.shape)
    
    # Streudiagramm und Konturen darstellen, die die 
    # Entscheidungsgrenze zeigen
    plt.contourf(x1, x2, z)    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision boundary')
    plt.scatter(X[:,0], X[:,1], c=Y, edgecolors='w')
    