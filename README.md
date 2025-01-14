# Fake News Classifier

## Overview
Une application web de machine learning conçue pour détecter les fake news en utilisant des techniques TAL.

- Le jeu de données Kaggle (fake_train,fake_test)

- Le preprocessing du texte et l'entrainement des données.

- Le modèle de Régression Logistique adopté comme modèle final pour ce projet et construit une application Flask avec ce modèle. 

### Fonctionnalités

* Détection de fake news en temps réel
* Interface web
* Expérience utilisateur simple

### Détails

* Techniques :
    - Tokenisation
    - Lemmatisation
    - Mots-vides
    - Sac de mots
    - GridSearchCV
    - Validation croisée K-fold

### Détails Techniques 

Backend : 
* Flask
* Machine Learning :
    * Régression Logistique
    * scikit-learn
    * NLTK
* Frontend : HTML, CSS

#### Processus de Prétraitement du Texte
Le modèle utilise plusieurs techniques de traitement du langage naturel pour préparer le texte :

1. **Nettoyage du Texte**
   - Suppression des caractères non alphabétiques
   - Conversion en minuscules
   - Tokenisation (séparation en mots)

2. **Élimination des Mots Vides (Stopwords)**
   - Suppression des mots courants comme "the", "a", "an" qui n'apportent pas de sens significatif

3. **Stemmatisation**
   - Réduction des mots à leur racine (ex: "running" → "run")
   - Utilisation de l'algorithme de Porter Stemmer

#### Vectorisation
- Utilisation de CountVectorizer pour convertir le texte en vecteurs numériques
- Transformation du texte en un ensemble de caractéristiques (features) numériques

#### Modèle de Classification
- Régression Logistique
- Techniques :
  - GridSearchCV pour l'optimisation des hyperparamètres
  - Validation croisée K-fold

### Installation

pip install -r requirements.txt

python app1.py

### How to Use
1. Entrez le texte de la nouvelle dans la zone de saisie
2. Cliquez sur "Predict"
3. Recevez la classification (Fake news ou Real news)
