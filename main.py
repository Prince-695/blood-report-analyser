from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

# Load the main dataset
df_main = pd.read_csv("Training.csv")

# Diseases
disease = {0: 'Anemia', 1: 'Polycythemia', 2: 'Leukocytosis', 3: 'Leukopenia', 4: 'Thrombocytopenia',
           5: 'Thrombocytosis', 6: 'Neutropenia', 7: 'Neutrophilia', 8: 'Lymphocytopenia', 9: 'Lymphocytosis',
           10: 'Monocytes high', 11: 'Eosinophil high', 12: 'Basophil high', 13: 'Normal'}

# Causes
Rea = {0: [' - Anemia due to blood loss \n'
            ' - Bone marrow disorders \n'
            ' - Nutritional deficiency \n'
            ' - Chronic Kidney disease  \n'
            ' - Chronic inflammatory disease \n'],
       1: ['- Dehydration, such as from severe diarrhea \n'
           '- tumours \n'
           '- Lung diseases \n'
           '- Smoking \n'
           '- Polycythemia vera \n'],
       2: ['- Infection \n'
           '- Leukemia \n'
           '- Inflammation \n'
           '- Stress, allergies, asthma \n'],
       3: ['- Viral infection \n'
           '- Severe bacterial infection \n'
           '- Bone marrow disorders \n'
           '- Autoimmune conditions \n'
           '- Lymphoma \n'
           '- Dietary deficiencies \n'],
       4: ['- Cancer, such as leukemia or lymphoma \n'
           '- Autoimmune diseases \n'
           '- Bacterial infection \n'
           '- Viral infection like dengue \n'
           '- Chemotherapy or radiation therapy \n'
           '- Certain drugs, such as nonsteroidal anti-inflammatory drugs (NSAIDs) \n'],
       5: ['- Bone marrow disorders \n'
           '- Essential thrombocythemia \n'
           '- Anemia \n'
           '- Infection \n'
           '- Surgical removal of the spleen \n'
           '- Polycythemia vera \n'
           '- Some types of leukemia \n'],
       6: ['- Severe infection \n'
           '- Immunodeficiency \n'
           '- Autoimmune disorders \n'
           '- Dietary deficiencies \n'
           '- Reaction to drugs \n'
           '- Bone marrow damage \n'],
       7: ['- Acute bacterial infections \n'
           '- Inflammation \n'
           '- Stress, Trauma \n'
           '- Certain leukemias \n'],
       8: ['- Autoimmune disorders \n'
           '- Infections \n'
           '- Bone marrow damage \n'
           '- Corticosteroids \n'],
       9: ['- Acute viral infections \n'
           '- Certain bacterial infections \n'
           '- Chronic inflammatory disorder \n'
           '- Lymphocytic leukemia, lymphoma \n'
           '- Acute stress \n'],
       10: ['- Chronic infections \n'
            '- Infection within the heart \n'
            '- Collagen vascular diseases \n'
            '- Monocytic or myelomonocytic leukemia \n'],
       11: ['- Asthma, allergies such as hay fever \n'
            '- Drug reactions \n'
            '- Parasitic infections \n'
            '- Inflammatory disorders \n'
            '- Some cancers, leukemias or lymphomas \n'],
       12: ['- Rare allergic reactions \n'
            '- Inflammation \n'
            '- Some leukemias \n'
            '- Uremia \n'],
       13: ['- Normal \n']}

# Train model once at startup
x = df_main.drop(columns=['Disease'], axis=1)
y = df_main['Disease']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=40)
rf_model.fit(x_train, y_train)

# Function to predict using the trained model
def predict_disease(W, R, H, P, N, L, M, E, B):
    t = np.array([W, R, H, P, N, L, M, E, B]).reshape(1, -1)
    res = rf_model.predict(t)[0]
    return res

# Function to get model comparison results
def get_model_comparison():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=40),
        'Decision Tree': DecisionTreeClassifier(random_state=40),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=40),
        'SVM': SVC(random_state=40),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    
    for model_name, model in models.items():
        # Train the model
        model.fit(x_train, y_train)
        
        # Make predictions
        y_pred = model.predict(x_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'model_name': model_name,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'confusion_matrix': cm.tolist()
        })
    
    # Sort by F1 score descending
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        W = float(request.form['WBC'])
        R = float(request.form['RBC'])
        H = float(request.form['HGB'])
        P = float(request.form['PLT'])
        N = float(request.form['NEUT'])
        L = float(request.form['LYMPH'])
        M = float(request.form['MONO'])
        E = float(request.form['EO'])
        B = float(request.form['BASO'])

        # Predict the disease
        result = predict_disease(W, R, H, P, N, L, M, E, B)

        # Get the cause of the disease
        cause = Rea[result][0]

        # Render result template with results
        return render_template('index.html', disease=disease[result], cause=cause, show_result=True)

    return render_template('index.html', show_result=False)

@app.route('/comparison')
def comparison():
    # Get model comparison results
    comparison_results = get_model_comparison()
    return render_template('comparison.html', results=comparison_results)

if __name__ == '__main__':
    app.run(debug=True)
