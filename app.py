from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pickle
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime
import io

app = Flask(__name__)
app.secret_key = "diabetes_prediction_secret_key"

# Database setup
def init_db():
    conn = sqlite3.connect('diabetes_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction INTEGER,
        probability REAL,
        high_bp INTEGER,
        high_chol INTEGER,
        chol_check INTEGER,
        stroke INTEGER,
        heart_disease INTEGER,
        bmi REAL,
        smoker INTEGER,
        phys_activity INTEGER,
        fruits INTEGER,
        veggies INTEGER,
        hvy_alcohol INTEGER,
        gen_health INTEGER,
        phys_health INTEGER,
        ment_health INTEGER,
        diff_walk INTEGER,
        sex INTEGER,
        age INTEGER,
        education INTEGER,
        income INTEGER,
        any_healthcare INTEGER,
        timestamp DATETIME
    )
    ''')
    conn.commit()
    conn.close()

# Load the saved model and scaler
def load_model_and_scaler():
    try:
        model_path = os.path.join("saved_models", "diabetes_prediction_xgb_2.pkl")
        scaler_path = os.path.join("saved_models", "scaler.pkl")
        
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)
            
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

# Save prediction to database
def save_prediction(input_data, prediction, probability):
    conn = sqlite3.connect('diabetes_predictions.db')
    cursor = conn.cursor()
    
    # Prepare data for insertion
    data = {
        'prediction': int(prediction),
        'probability': float(probability),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **input_data
    }
    
    # Create placeholders for SQL query
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?' for _ in data])
    values = tuple(data.values())
    
    # Execute query
    cursor.execute(
        f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
        values
    )
    
    conn.commit()
    conn.close()

# Export predictions to CSV
def export_predictions_to_csv():
    conn = sqlite3.connect('diabetes_predictions.db')
    query = "SELECT * FROM predictions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Create in-memory CSV
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return output

# Initialize
init_db()
model, scaler = load_model_and_scaler()

# Define constants
AGE_GROUPS = {
    1: "18-24 years old",
    2: "25-29 years old",
    3: "30-34 years old",
    4: "35-39 years old",
    5: "40-44 years old",
    6: "45-49 years old",
    7: "50-54 years old",
    8: "55-59 years old",
    9: "60-64 years old",
    10: "65-69 years old",
    11: "70-74 years old",
    12: "75-79 years old",
    13: "80 years and older"
}

EDUCATION_LEVELS = {
    1: "Never attended school or only kindergarten",
    2: "Grades 1-8 (Elementary)",
    3: "Grades 9-11 (Some high school)",
    4: "Grade 12 or GED (High school graduate)",
    5: "College 1-3 years (Some college or technical school)",
    6: "College 4 years or more (College graduate)"
}

INCOME_LEVELS = {
    1: "Less than $10,000",
    2: "$10,000 to less than $15,000",
    3: "$15,000 to less than $20,000",
    4: "$20,000 to less than $25,000",
    5: "$25,000 to less than $35,000",
    6: "$35,000 to less than $50,000",
    7: "$50,000 to less than $75,000",
    8: "$75,000 or more"
}

@app.route('/')
def index():
    return render_template('index.html', 
                           age_groups=AGE_GROUPS,
                           education_levels=EDUCATION_LEVELS,
                           income_levels=INCOME_LEVELS)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Convert Yes/No to 1/0
            binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
            
            # Get form data
            input_data = {
                'high_bp': binary_map[request.form.get('highbp')],
                'high_chol': binary_map[request.form.get('highchol')],
                'chol_check': binary_map[request.form.get('cholcheck')],
                'stroke': binary_map[request.form.get('stroke')],
                'heart_disease': binary_map[request.form.get('heart_disease')],
                'bmi': float(request.form.get('bmi')),
                'smoker': binary_map[request.form.get('smoker')],
                'phys_activity': binary_map[request.form.get('phys_activity')],
                'fruits': binary_map[request.form.get('fruits')],
                'veggies': binary_map[request.form.get('veggies')],
                'hvy_alcohol': binary_map[request.form.get('hvy_alcohol')],
                'gen_health': int(request.form.get('gen_health')),
                'phys_health': int(request.form.get('phys_health')),
                'ment_health': int(request.form.get('ment_health')),
                'diff_walk': binary_map[request.form.get('diff_walk')],
                'sex': binary_map[request.form.get('sex')],
                'age': int(request.form.get('age')),
                'education': int(request.form.get('education')),
                'income': int(request.form.get('income')),
                'any_healthcare': binary_map[request.form.get('any_healthcare')]
            }
            
            # Create DataFrame for model input (mapping keys to match model expectations)
            model_input = {
                'PhysHlth': input_data['phys_health'],
                'HvyAlcoholConsump': input_data['hvy_alcohol'],
                'Smoker': input_data['smoker'],
                'CholCheck': input_data['chol_check'],
                'Stroke': input_data['stroke'],
                'GenHlth': input_data['gen_health'],
                'MentHlth': input_data['ment_health'],
                'Sex': input_data['sex'],
                'Income': input_data['income'],
                'BMI': input_data['bmi'],
                'PhysActivity': input_data['phys_activity'],
                'HeartDiseaseorAttack': input_data['heart_disease'],
                'AnyHealthcare': input_data['any_healthcare'],
                'DiffWalk': input_data['diff_walk'],
                'HighChol': input_data['high_chol'],
                'Veggies': input_data['veggies'],
                'Education': input_data['education'],
                'Age': input_data['age'],
                'HighBP': input_data['high_bp'],
                'Fruits': input_data['fruits']
            }
            
            input_df = pd.DataFrame([model_input])
            
            if model is not None and scaler is not None:
                # Scale features
                scaled_features = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(scaled_features)[0]
                prediction_proba = model.predict_proba(scaled_features)[0][1]  # Probability of class 1
                
                # Save to database
                save_prediction(input_data, prediction, prediction_proba)
                
                return render_template('result.html', 
                                      prediction=prediction,
                                      probability=prediction_proba * 100)
            else:
                flash("Error: Model not loaded properly", "danger")
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f"Error processing your request: {str(e)}", "danger")
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/export-csv')
def export_csv():
    output = export_predictions_to_csv()
    return send_file(
        output,
        mimetype='text/csv',
        download_name='diabetes_predictions.csv',
        as_attachment=True
    )

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('saved_models', exist_ok=True)
    app.run(debug=True)