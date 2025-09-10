# 🎬 YouTube ML Project  
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)  
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)  
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)  

---

## 📌 Objetivo del proyecto
Este proyecto busca **predecir la probabilidad de que un video de YouTube se convierta en un “hit”** (top 10% en *engagement rate* dentro de su cohorte categoría × duración) **antes de su publicación**.  

Se abordan dos enfoques complementarios:  
- **No supervisado**: segmentación temática de videos mediante *K-Means* y *Greedy Modularity* (detección de comunidades en grafos de similitud).  
- **Supervisado**: predicción binaria (`hit_er`) usando **Logistic Regression, Random Forest y XGBoost**, con validación cruzada y tuning de hiperparámetros.  

---

## 🗂️ Estructura del repositorio
```

youtube-ml-project/
├── data/                # Dataset original (CSV)
├── notebook/            # Jupyter notebooks de análisis y modelado
│   ├── 01\_EDA.ipynb
│   ├── 02\_Preprocessing\_Verification.ipynb
│   ├── 03\_Unsupervised.ipynb
│   └── 04\_Supervised.ipynb
├── src/                 # Código modular en Python
│   ├── utils.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── evaluation.py
│   └── models.py
├── requirements.txt     # Dependencias del proyecto
└── README.md

````

---

## ⚙️ Instalación
1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuusuario/youtube-ml-project.git
   cd youtube-ml-project
````

2. Crea un entorno virtual e instala dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

   pip install -r requirements.txt
   ```

---

## 🚀 Uso

1. Coloca tu dataset CSV en la carpeta `data/` con el nombre `youtube_data.csv`.

2. Ejecuta los notebooks en orden:

   * `01_EDA.ipynb`: análisis exploratorio de los datos.
   * `02_Preprocessing_Verification.ipynb`: validación del pipeline de preprocesamiento.
   * `03_Unsupervised.ipynb`: segmentación temática con K-Means y Greedy Modularity.
   * `04_Supervised.ipynb`: entrenamiento de modelos supervisados, tuning y comparación.

3. Los resultados incluyen:

   * **Gráficas** de distribución, clusters, comunidades y métricas.
   * **Tablas comparativas** de desempeño (AUC-PR, AUC-ROC, Precision\@10%).

---

## 📊 Resultados principales

* **Clusters temáticos** identificados: música/covers, gaming/let’s play, episodios/series, tutoriales/how-to.
* **Comunidades en grafos** con modularidad Q > 0.7, reflejando estructura temática clara.
* **Modelos supervisados**:

  * **XGB\_tuned** → mejor ranking global (AUC-PR y AUC-ROC).
  * **RF\_tuned** → mejor en Precision\@10% (shortlist de hits).
  * **LogReg\_tuned** → interpretable y competitivo.

---

## 🛠️ Requisitos

* Python 3.10+
* scikit-learn 1.3+
* XGBoost 1.7+
* pandas, numpy, matplotlib, networkx

Instalables con:

```bash
pip install -r requirements.txt
```

---

## 📌 Trabajo futuro

* Mejorar la calibración de probabilidades.
* Probar embeddings (Word2Vec, BERT) para enriquecer los títulos.
* Extender el análisis por categoría/duración.
* Incorporar métricas de negocio personalizadas para definir “hit”.