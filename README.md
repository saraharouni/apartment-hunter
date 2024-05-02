# Projet Apartment-hunter

**SUJET :**  

Un datascientist décide de se reconvertir en agent immobilier après avoir visionné une émission de 'chasseur d'appart'.  
Entouré par 2 collaborateurs, ils décident de mettre en avant leurs compétences de Data Scientist afin de créer un modèle capable de faire des estimations de prix de biens immobiliers.  


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

        
