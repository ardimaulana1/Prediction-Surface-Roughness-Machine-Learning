####################################
# Import libraries
####################################
from flask import Flask, render_template, request
import joblib
import pandas as pd
####################################
# Memulai Flask app setup
####################################
# Setup Flask app
# Deklarasi aplikasi Flask
app = Flask(__name__)
print("Flask app initialized")

model = joblib.load(open('model/linear_regression_model.pkl', 'rb'))
poly_features = joblib.load(open('model/poly_features.pkl', 'rb'))
    
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print_speed = int(request.form['printing_speed'])
    layer_thick = float(request.form['layer_thickness'])
    temp = int(request.form['tempterature_extruder'])
    getaran_x = int(request.form['getaran_x'])
    getaran_y = int(request.form['getaran_y'])
    getaran_z = int(request.form['getaran_z'])
    
    data_baru = {
                 "printing_speed": print_speed, 
                 "layer_thickness": layer_thick, 
                 "tempterature_extruder": temp, 
                 "getaran_x":getaran_x, 
                 "getaran_y":getaran_y, 
                 "getaran_z":getaran_z, 
                } 
    data_baru = pd.DataFrame(data_baru, index = [0])

    data_poly_baru = poly_features.transform(data_baru)
    prediction = model.predict(data_poly_baru)
    
    prediction = round(prediction.item(), 2)

    return render_template('index.html', hasil_prediksi=prediction)

# Run app
if __name__ == '__main__':
    app.run(debug=True)