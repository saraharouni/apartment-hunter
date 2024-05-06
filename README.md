# Projet Apartment-hunter

**SUJET :**  

Un Data Scientist décide de se reconvertir en agent immobilier après avoir visionné une émission de **'Chasseur d'appart'**.  

Entouré par 2 collaborateurs, ils décident de mettre en avant leurs compétences de Data Scientist afin de créer un modèle capable de faire des estimations de prix de biens immobiliers.  

# Régression linéaire


![image](https://github.com/saraharouni/apartment-hunter/assets/60361890/cf33ec91-8296-44ef-a52c-f3937f7fa2fe)


**Définition :** La régression linéaire est une méthode de modélisation permettant d’établir une relation linéaire entre une variable continue dite "variable expliquée" ou dépendante et un
ensemble d’autres variables continues dites "variables explicatives" ou indépendantes.

**Formule de la régression linéaire simple :**  

<sup>$$ y= \beta_0 + \beta_1 \times X+ \varepsilon $$</sup>

où :  
* **y** : variable dépendante (à prédire)
* **X** : variable indépendante (prédicteur)
* **β0** : constante (ordonnée à l'origine)
* **β1** : coefficient de pente
* **ε** : erreur aléatoire


**Formule de la régression linéaire multiple :**  

$$
y = \beta_0 + \beta_1 \times X_1 + \beta_2 \times X_2 + \ldots + \beta_n \times X_n + \varepsilon
$$

où:  
**X1 ,X2 ,…,Xn** : variables indépendantes multiples


### Régression ElasticNetCV :

La régression ElasticNetCV est une régression linéaire régularisée qui combine à la fois la pénalisation L1 (Lasso) et L2 (Ridge).
Elle est utile lorsque les variables explicatives sont fortement corrélées entre elles.

**Fonctionnement de la régression ElasticNetCV :**

- Elle minimise à la fois la somme des carrés des erreurs (MSE) et une combinaison pondérée des pénalités L1 et L2.
- L'équation de la régression ElasticNet est donnée par :

$$ y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n + \varepsilon $$

- Avec la pénalisation L1 (Lasso) et L2 (Ridge) incorporée pour éviter le surajustement.


### Régression polynomiale :

La régression polynomiale est une forme de régression linéaire dans laquelle la relation entre la variable indépendante \(X\) et la variable dépendante \(y\) est modélisée comme un polynôme de degré \(n\).
Elle peut capturer des relations non linéaires entre les variables en ajoutant des termes polynomiaux.

**Fonctionnement de la régression polynomiale :**

- L'équation de la régression polynomiale est donnée par :

$$ y = \beta_0 + \beta_1 X + \beta_2 X^2 + \ldots + \beta_n X^n + \varepsilon $$

- Elle modélise la relation entre \(X\) et \(y\) sous forme de polynôme de degré \(n\).
- Les termes \(X^2\), \(X^3\), ..., \(X^n\) permettent de capturer des relations non linéaires entre \(X\) et \(y\).

### Random Forest Regressor :

Le Random Forest Regressor est un algorithme d'apprentissage supervisé basé sur l'ensemble qui utilise de multiples arbres de décision pour prédire une variable continue.
Il est capable de capturer des relations complexes entre les variables et est moins susceptible de surajuster que les modèles de régression linéaire.

**Fonctionnement du Random Forest Regressor :**

- Il utilise un ensemble de plusieurs arbres de décision pour prédire une variable continue.
- Chaque arbre de décision est formé sur un sous-ensemble aléatoire des données d'entraînement et des fonctionnalités.
- La prédiction finale est la moyenne des prédictions de tous les arbres dans la forêt.

Ces modèles de régression sont couramment utilisés pour modéliser les relations entre les variables dans les données et pour faire des prédictions sur une variable continue. Chacun de ces modèles a ses propres avantages et inconvénients, et le choix du modèle dépend souvent du contexte spécifique du problème et des préférences en termes de performance et d'interprétabilité.


# Métriques d'évaluation d'une regression :

**Mean Absolute Error (MAE) :**

La MAE est une mesure de la différence entre deux variables continues.
Elle représente la moyenne des valeurs absolues des écarts entre les prédictions et les vraies valeurs.

La formule de la MAE est :

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_{i} - \hat{y_{i}}| vbnet $$


La MAE est utilisée pour évaluer la performance d'un modèle de régression en mesurant la magnitude moyenne des erreurs de prédiction.

**Mean Squared Error (MSE) :**

La MSE est également une mesure de la différence entre deux variables continues.
Elle représente la moyenne des carrés des écarts entre les prédictions et les vraies valeurs.

La formule de la MSE est :

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y_{i}})^{2} $$

La MSE est utilisée pour évaluer la performance d'un modèle de régression en mesurant la moyenne des carrés des erreurs de prédiction.

**Root Mean Squared Error (RMSE) :**

La RMSE est la racine carrée de la MSE.
La formule de la RMSE est :

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y_{i}})^{2}} $$


La RMSE est utilisée pour évaluer la performance d'un modèle de régression de la même manière que la MSE, mais elle donne une mesure de l'erreur dans les mêmes unités que la variable cible, ce qui la rend plus facilement interprétable.


# Features Selection :
Le choix de la ou des variables explicatives est important pour construire un modèle. Il est important de vérifier en amont si ces variables sont en multicolinéarité (Variables explicatives corrélées entre-elles).  
Pour vérifier cela, il est possible d'utiliser le VIF.  

### VIF (Variance Inflation Factor) :

Le VIF est une mesure de l'ampleur de la multicolinéarité dans un modèle de régression linéaire. Il mesure à quel point l'erreur standard d'un coefficient de régression est affectée par la multicolinéarité dans le modèle.

**VIF pour une variable \( X_i \) :**

$$ VIF_i = \frac{1}{1 - R^2_{X_i | X_{-i}}} $$


**Interprétation du VIF :**
- Si \( VIF_i > 10 \) : Une variable avec un VIF supérieur à 10 indique multicolinéarité.

### Multicolinéarité :

La multicolinéarité se produit lorsque deux ou plusieurs variables indépendantes dans un modèle de régression linéaire sont fortement corrélées entre elles.

**Effets de la multicolinéarité :**
* Augmente la variabilité des coefficients estimés.
* Rend les estimations des coefficients instables.
* Rend les tests d'hypothèse sur les coefficients peu fiables.

### Boruta :

Boruta est un algorithme de sélection de fonctionnalités qui identifie les fonctionnalités importantes dans un ensemble de données. Il utilise une approche de type wrapper pour sélectionner les fonctionnalités en évaluant leur importance par rapport à des variables aléatoires (ombrage).

**Fonctionnement de Boruta :**
- Il compare les scores de permutation des variables réelles avec ceux des variables ombragées.
- Les variables réelles qui ont un score de permutation significativement plus élevé que les variables ombragées sont considérées comme importantes.

**Utilisation de Boruta :**
- Boruta est souvent utilisé pour la sélection automatique des fonctionnalités dans les ensembles de données où il y a un grand nombre de fonctionnalités et où il est difficile de déterminer quelles fonctionnalités sont importantes pour le modèle.

En résumé, le VIF est utilisé pour détecter la multicolinéarité entre les variables indépendantes dans un modèle de régression, tandis que Boruta est utilisé pour la sélection automatique des fonctionnalités dans les ensembles de données. La multicolinéarité est un problème commun dans les modèles de régression et peut entraîner une instabilité dans les estimations des coefficients.


# Procédure

* Faire une analyse exploratoire des données sur PowerBi
* Concevoir plusieurs modèles de régression linéaire optimisés
* Créer une application Flask qui fera les prédictions
* Créer une image Docker de l'application

# Application

Cette application permet de faire des prédictions sur le prix de biens immobiliers. 
Pour l'utiliser, il suffit de répondre aux questions et de cliquer sur le bouton *prédiction*

## Utilisation

Pour utiliser cette application, suivez ces étapes :

1. Clonez ce dépôt sur votre machine locale :

    ```bash
    git clone https://github.com/saraharouni/apartment-hunter.git
    ```

2. Accédez au répertoire de l'application :

    ```bash
    cd votre-repo
    ```

3. Construisez l'image Docker :

    ```bash
    docker compose up --build
    ```

4. Lancez le conteneur Docker :

    ```bash
    docker run -p 8080:8080 apartment-hunter-main-server
    ```

5. Accédez à l'application dans votre navigateur en utilisant l'URL suivante :

    ```
    http://localhost:8080
    ```

**Analyse exploratoire PowerBI :**  

Après avoir fait notre nettoyage de données, mis en place l’application pour la prédiction de ventes
ainsi que Docker, nous avons récupérer le fichier CSV afin de préparer le Dashboard Power Bi.
Dans un premier temps nous avons créé de nouvelle mesures et avons regrouper dans une même
colonne les habitations de notre fichier CSV pour une analyse plus pertinente.
Nous nous sommes concentrer sur les différents aspects afin de montrer l’évolution de prix d’achats
des habitations par rapport à leur emplacement dans les quartiers que nous avions dans notre jeu de
données.
Ainsi nous avons pu constater que par rapport aux différents caractéristiques le prix moyen d’achat
des logements a fortement augmenté en 1900.
Nous avons aussi une proportion de construction d’appartement qui est beaucoup plus forte
contrairement aux penthouses, maison ou duplex.
Notre Dashboard est dynamique ce qui nous permet en en seul coup d’œil d’obtenir des
informations pertinentes et ainsi connaitre le prix des biens sur le marché à différentes périodes
comprises entre 1800 et 2022.

**Conclusion :**
Nous avons utilisé plusieurs modèles de régression pour estimer le prix d'achat de biens immobiliers:  

* la régression linéaire simple comme baseline qui affiche :
    * un R² de 0.64 sur les données de test. 
    * La MAE est assez élevée (152 321 euros) ainsi que la RMSE (214 441 euros).

* le modèle ElasticNetCV sans features selection :  
    * un R² de 0.91 sur les données de test.
    * La MAE est deux fois moins importante que celle de la baseline (63 165 euros) ainsi que la RMSE à 104653 euros.

* le modèle ElasticNetCV avec features selection (librairie Boruta) :  
    * un R² de 0.90 sur les données de test.
    * La MAE est deux fois moins importante que celle de la baseline (64 161 euros) ainsi que la RMSE à 110487 euros. 
    * Ce modèle est moins performant que le premier modèle d'ElasticNetCV sans features selection mais plus performant que la baseline.

* le modèle de régression polynomiale à 2 degrés :  
    * un R² de 0.99 sur les données de test.
    * La MAE est très basse (148 euros) ainsi que la RMSE à 778 euros. 
    * Ce modèle est très performant avec ce score ce qui nous a amené à faire beaucoup plus de tests en amont afin de voir s'il n'était pas en overfitting.  
* le modèle de random forest regressor :  
    * un R² de 0.99 sur les données de test.
    * La MAE est très basse (2625 euros) ainsi que la RMSE à 10637 euros. 
    * Ce modèle est aussi très performant mais la MAE ainsi que la RMSE sont légèrement plus élevées que celles de la regression polynomiale.
    
Nous décidons donc d'utiliser le modèle de régression polynomiale pour notre application.

        
