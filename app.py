from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Charger le fichier CSV
houses_in_madrid = pd.read_csv('visuel.csv', index_col=0)

# Charger le modèle
model = joblib.load('model_en_2.pkl')
poly = joblib.load('poly.pkl')
scaler = joblib.load('scaler.pkl')  # Charger RobustScaler

prediction_result = None
prediction_visu = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global prediction_result
    global prediction_visu
    if request.method == 'POST':
        # Récupérer les données du formulaire
        m2_construit = int(request.form['m2_construit'])
        #nb_etages = int(request.form['nb_etages'])
        prix_achat_m2 = int(request.form['prix_achat_m2'])
        travaux_necessaire = bool(request.form.get('travaux_necessaire', False))
        jardin = bool(request.form.get('jardin', False))
        piscine = bool(request.form.get('piscine', False))
        terrace = bool(request.form.get('terrace', False))
        balcon = bool(request.form.get('balcon', False))
        salle_rangement = bool(request.form.get('salle_rangement', False))
        espace_vert = bool(request.form.get('espace_vert', False))
        parking = bool(request.form.get('parking', False))
        num_quartier = int(request.form['num_quartier'])
        type_duplex = bool(request.form.get('type_duplex', False))
        type_maison = bool(request.form.get('type_maison', False))
        type_penthouse = bool(request.form.get('type_penthouse', False))
        
        # Préparer les données pour la prédiction
        data = [[m2_construit, prix_achat_m2, travaux_necessaire,
                 jardin, piscine, terrace, balcon ,salle_rangement, espace_vert, parking, 
                 num_quartier, type_duplex, type_maison, type_penthouse]]
        
        # Appliquer la transformation polynomiale sur les données
        data_poly = poly.transform(data)
        
        # Mettre à l'échelle les données
        data_scaled = scaler.transform(data_poly)  # Mettre à l'échelle avec RobustScaler
        
        # Faire la prédiction
        prediction_result = round(model.predict(data_scaled)[0], 2)
        prediction_visu = prediction_result

        return redirect(url_for('predict'))

    return render_template('index.html', prediction_result=prediction_result)

@app.route('/predict')
def predict():
    global prediction_result
    session['prediction_result'] = prediction_result
    return render_template('predict.html', prediction_result=prediction_result)

@app.route('/houses_in_madrid')
def get_houses_in_madrid():
    global prediction_visu
    if prediction_visu is not None:
        similar_houses = houses_in_madrid[
            (houses_in_madrid['prix_achat'] >= prediction_visu - 1500) &
            (houses_in_madrid['prix_achat'] <= prediction_visu + 1500)
        ].to_dict(orient='records')
        return jsonify(similar_houses)
    else:
        return jsonify([])
    
if __name__ == '__main__':
    app.run(debug=True)
