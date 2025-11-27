# Reconnaissance de formes avec TensorFlow

Ce projet illustre un workflow complet pour la reconnaissance de chiffres manuscrits en utilisant le dataset MNIST. Il couvre l'entraînement du modèle avec Keras (TensorFlow) et l'inférence via une application Python dédiée.

La base de données **MNIST (Modified National Institute of Standards and Technology)** est une vaste collection de chiffres manuscrits, largement utilisée pour l'entraînement et les tests en apprentissage automatique.

*   **Contenu** : Elle contient 60 000 images d'entraînement et 10 000 images de test.
*   **Format** : Chaque image est en niveaux de gris de 28x28 pixels, représentant un chiffre de 0 à 9.
*   **Objectif** : Elle sert de référence pour les algorithmes de classification.

## Architecture du Modèle utilisé: CNN

Un Réseau de Neurones Convolutif (CNN ou ConvNet) est une architecture de Deep Learning spécialement conçue pour traiter des données structurées en grille, telles que les images. Contrairement aux réseaux classiques qui traitent les pixels individuellement, les CNN utilisent des filtres pour scanner l'image et en extraire des motifs.

**Pourquoi ce choix ?**
Le CNN est l'architecture standard pour la classification d'images (comme MNIST) pour plusieurs raisons :
1.  **Invariance spatiale** : Il peut reconnaître un chiffre peu importe où il se trouve dans l'image.
2.  **Extraction de caractéristiques** : Il apprend automatiquement à identifier des formes simples (lignes, courbes) puis complexes (boucles des chiffres) grâce à ses couches de convolution.
3.  **Efficacité** : Il réduit considérablement le nombre de paramètres à apprendre par rapport à un réseau dense classique, tout en étant plus performant pour capturer les dépendances spatiales entre les pixels.

## Outils Principaux

### 1. TensorFlow
Une plateforme open-source de bout en bout pour l'apprentissage automatique développée par Google. Elle est optimisée pour le calcul numérique haute performance, exploitant les GPU et TPU.

### 2. Keras
Une API de haut niveau pour les réseaux de neurones, écrite en Python et capable de s'exécuter au-dessus de TensorFlow. Elle offre des API cohérentes et simples.

### 3. NumPy
Le package fondamental pour le calcul scientifique avec Python. Il fournit un objet tableau multidimensionnel haute performance et des outils pour travailler avec les tableaux.

## Prérequis & Utilisation

### Prérequis
*   **Python** : Version 3.13 ou supérieure.
*   **Poetry** : Outil de gestion des dépendances.

### Comment Utiliser

1.  **Installation** :
    Installez les dépendances du projet.
    ```bash
    poetry install
    ```

2.  **Entraînement du Modèle** :
    Lancez le script d'entraînement pour générer et sauvegarder le modèle.
    ```bash
    poetry run python mnist_keras.py
    ```

3.  **Exécuter l'Inférence** :
    Lancez l'application pour tester les performances du modèle.
    ```bash
    poetry run python app.py
    ```

4.  **Lancer les Tests** :
    Exécutez la suite de tests pour assurer l'intégrité du code.
    ```bash
    poetry run pytest
    ```

## Structure du Projet

### 1. Entraînement (`mnist_keras.py`)
Ce script est responsable de la construction et de l'entraînement du réseau de neurones.

*   **Analyse de la Méthode Principale** :
    *   **Préparation des Données** : Charge le dataset MNIST et le prépare (normalisation, redimensionnement).
    *   **Construction du Modèle** : Construit un Réseau de Neurones Convolutif (CNN) en utilisant l'API Fonctionnelle de Keras. L'architecture inclut des couches de Convolution, de Max Pooling, d'Aplatissement (Flattening), et des couches Denses avec Dropout pour la régularisation.
    *   **Compilation** : Configure le modèle avec l'optimiseur Adam et la fonction de perte "Categorical Crossentropy".
    *   **Entraînement** : Ajuste le modèle aux données d'entraînement sur un nombre spécifié d'époques et d'étapes.
    *   **Export** : Sauvegarde le modèle entraîné dans le répertoire `model_keras/` au format SavedModel de TensorFlow.

### 2. Inférence (`app.py`)
Ce script représente la couche de production/application qui consomme le modèle entraîné.

*   **Rôle** : Il démontre comment charger un modèle sauvegardé et l'utiliser pour faire des prédictions sur de nouvelles données.
*   **Processus** :
    *   Charge le dataset de test via `mnist_utils`.
    *   Charge le modèle pré-entraîné depuis `model_keras/` en utilisant `tf.saved_model.load`.
    *   Itère sur les images de test, exécute l'inférence sur chacune, et compare la prédiction avec l'étiquette réelle.
    *   Calcule et rapporte la précision globale du modèle.
