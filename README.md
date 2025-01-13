# Fake News Classifier

## Overview
Une application web de machine learning conçue pour détecter les fake news en utilisant des techniques avancées de traitement du langage naturel.
- Le jeu de données de Kaggle, et le preprocessing du texte et l'entrainement des données avec 3 modèles:
1. Naive Bayes Multinomial   
2. Régression Logistique  
3. Réseau de Neurones Profonds LSTM  
- Le modèle de Régression Logistique adopté comme modèle final pour ce projet et construit une application Flask avec ce modèle. 

### Fonctionnalités

* Détection de fausses nouvelles en temps réel
* Interface web
* Modèle de machine learning avec haute précision
* Expérience utilisateur simple et intuitive


### Détails

* Précision : 93,5%
* Techniques :
    - Tokenisation
    - Lemmatisation
    - Mots-vides
    - Sac de mots
    - GridSearchCV
    - Validation croisée K-fold

### Détails Techniques du Modèle de Classification

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
- Précision : 93.52%
- Techniques avancées :
  - GridSearchCV pour l'optimisation des hyperparamètres
  - Validation croisée K-fold

#### Workflow de Prédiction
1. Prétraitement du texte d'entrée
2. Vectorisation
3. Prédiction binaire :
   - 0 : Vraie Nouvelle (Real News)
   - 1 : Fausse Nouvelle (Fake News)

Backend : 
* Flask
* Machine Learning :
    * Régression Logistique
    * scikit-learn
    * NLTK
* Frontend : HTML, CSS


### How to Use
1. Entrez le texte de la nouvelle dans la zone de saisie
2. Cliquez sur "Predict"
3. Recevez instantanément la classification (Fake news ou Real news)

### Installation

pip install -r requirements.txt

python app.py
