# Künstliche neuronale Netze

Eine praxisnahe Einführung in künstliche neuronale Netze mit TensorFlow und Keras. Drei Tage lang arbeitest du dich mit Aufgaben von dichten Netzen zu Faltungsnetzen und Transfer Learning vor. Dazu kommt ein optionaler Bonus, in dem du ein Netz von Grund auf erstellst.

## Lernziele

Am Ende dieses Repositorys kannst du:

- dichte neuronale Netze für Regression und Klassifikation mit TensorFlow und Keras erstellen und trainieren.
- Over- und Underfitting diagnostizieren und Regularisierung, Dropout und Early Stopping anwenden, um dem entgegenzuwirken.
- trainierte Keras-Modelle speichern und für die Wiederverwendung neu laden.
- ein Convolutional Neural Network für die Bildklassifikation implementieren.
- Transfer Learning anwenden, um ein vortrainiertes Netz an eine neue Aufgabe anzupassen.
- ein dichtes Netz und seine Rückwärtspropagierung von Grund auf mit NumPy implementieren.

## Lernpfad

Die Notebooks sind nach Tagen gruppiert und bauen der Reihe nach aufeinander auf:


### Tag 1 - Dichte neuronale Netze

Implementiere dichte neuronale Netze für Regression und Klassifikation. Der Kurs verwendet durchgehend TensorFlow und Keras.

| Datei | Beschreibung |
|---|---|
| [**01 - Regression mit TensorFlow und Keras**](day_1/01_regression_tensorflow_keras.ipynb) | Ein dichtes Netz erstellen, um eine stetige Zielvariable vorherzusagen. |
| [**02 - Klassifikation mit TensorFlow und Keras**](day_1/02_classification_tensorflow_keras.ipynb) | Ein dichtes Netz für ein Klassifikationsproblem erstellen. |

### Tag 2 - Regularisierung und Speichern von Modellen

Diagnostiziere Over- und Underfitting, speichere dann trainierte Modelle und lade sie neu.

| Datei | Beschreibung |
|---|---|
| [**03 - Over- und Underfitting**](day_2/03_overfit_underfit.ipynb) | Die Kapazität verschiedener Modelle vergleichen und Regularisierung und Dropout anwenden. |
| [**04 - Modelle speichern und laden**](day_2/04_load_saved_models.ipynb) | Ein trainiertes Keras-Modell dauerhaft speichern und neu laden (optional). |

### Tag 3 - Faltungen und Transfer Learning

Geh von dichten Netzen zu Faltungsnetzen über und verwende dann ein vortrainiertes Netz für eine neue Aufgabe wieder.

| Datei | Beschreibung |
|---|---|
| [**05 - CNN mit Keras**](day_3/cnn/cnn_keras.ipynb) | Ein Convolutional Neural Network für die Bildklassifikation erstellen. |
| [**06 - Vortrainierte Netze und Transfer Learning**](day_3/pretrained_transfer_learning/pretrained_networks-transfer_learning.ipynb) | Ein vortrainiertes VGG16-Netz anpassen, um neue Bilder zu klassifizieren. |

### Bonus

Optionale Vertiefung, wenn du mit den anderen Tagen fertig bist. In der Praxis erstellst du selten ein Netz von Grund auf, aber wenn du es einmal gemacht hast, verstehst du das Training viel leichter.

| Datei | Beschreibung |
|---|---|
| [**00 - DNN von Grund auf**](bonus/00_dnn_from_scratch.ipynb) | Ein dichtes Netz und seine Rückwärtspropagierung mit NumPy implementieren. |

### Weitere Ordner und Dateien

| Datei / Ordner | Beschreibung |
|---|---|
| [**Solutions**](solutions/) | Musterlösungen. |
| [**pyproject.toml**](pyproject.toml) | Projektkonfiguration und Abhängigkeiten. |
| [**uv.lock**](uv.lock) | Lock-Datei für die Abhängigkeiten. |

## Setup

Verwende [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 oder neuer
und Python 3.13.17 oder ein neueres Patch-Release von 3.13, wie in
[`.python-version`](.python-version) festgelegt.

> [!NOTE]
> In allen Schritten ist Text in spitzen Klammern wie `<repo-name>` ein **Platzhalter**. Ersetz ihn samt den Klammern `< >` durch deinen eigenen Wert. Aus `cd <repo-name>` wird zum Beispiel `cd my-ann-project`.

### 1. Repository aus dem Template erstellen

Klick auf GitHub auf **Use this template**.

Beim Erstellen des Repositorys:

- Leg dich selbst als **Owner** fest
- Wähl einen Namen für das Repository
- Deaktiviere **Include all branches**
- Klick auf **Create repository**

> [!IMPORTANT]
> Wenn du zu zweit oder in einer Gruppe arbeitest, erledigt nur **eine Person** diesen Schritt.

---

### 2. Collaborators hinzufügen (nur bei Paar- oder Gruppenarbeit)

Wenn du mit Teammitgliedern arbeitest:

1. Öffne das Repository auf GitHub
2. Geh zu **Settings → Collaborators**
3. Füg deine Teammitglieder als Collaborators hinzu
4. Teil den Link zum Repository mit deinem Team

Die Teammitglieder sollten die Einladung annehmen, bevor sie weitermachen.

---

### 3. Repository klonen

Kopiere die SSH-URL über den Button **Code** auf GitHub und führ dann Folgendes aus:

```bash
git clone <copied-ssh-url>
```

Die kopierte SSH-URL hat die Form `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. In den Projektordner wechseln und Abhängigkeiten installieren

Das installiert alle Abhängigkeiten und erstellt eine virtuelle Umgebung in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Daten entpacken

Die Datensätze sind in ZIP-Archiven gebündelt, die in von Git ignorierte Ordner `data/` entpackt werden. Es gibt zwei davon, an unterschiedlichen Stellen. Diese Befehle verwenden die Umgebung aus `uv sync`:

Die in den Notebooks von Tag 1 und Tag 2 verwendeten tabellarischen Datensätze (`boston.csv`, `titanic.csv`, `vehicle_emissions.csv`):

```bash
uv run python -c "import zipfile; zipfile.ZipFile('data.zip').extractall()"
```

Der Bilddatensatz, den das Notebook zum Transfer Learning verwendet:

```bash
uv run python -c "import zipfile; zipfile.ZipFile('day_3/pretrained_transfer_learning/data.zip').extractall('day_3/pretrained_transfer_learning')"
```

---

### 6. Notebooks öffnen

> [!NOTE]
> Achte darauf, VS Code im Stammverzeichnis des Projekts zu öffnen, damit es die von `uv sync` erstellte Umgebung automatisch erkennt.

Starte VS Code im Stammverzeichnis des Projekts:

```bash
code .
```

Öffne dann ein Notebook und wähl die von `uv sync` erstellte Python-Umgebung als Kernel aus.

## Quellen & weiterführende Literatur

- [**Keras: The high-level API for TensorFlow**](https://www.tensorflow.org/guide/keras): Offizieller Leitfaden zum Erstellen von Modellen mit Keras.
- [**Overfit and Underfit**](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit): TensorFlow-Tutorial zum Diagnostizieren und Steuern der Anpassung eines Modells.
- [**Transfer Learning and Fine-tuning**](https://keras.io/guides/transfer_learning/): Keras-Leitfaden zur Wiederverwendung vortrainierter Netze.
- [**Get Started with TensorBoard**](https://www.tensorflow.org/tensorboard/get_started): Trainingsmetriken und Modellgraphen visualisieren.
- [**A Neural Network Playground**](https://playground.tensorflow.org/): Mit einem kleinen neuronalen Netz im Browser experimentieren, um ein intuitives Verständnis zu entwickeln.
