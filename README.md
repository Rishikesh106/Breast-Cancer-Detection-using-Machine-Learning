AI Breast Cancer Diagnostic System
A Streamlit-based machine learning app for breast cancer diagnosis using a trained Support Vector Machine model. The app supports manual feature entry, batch CSV predictions, and a model overview with SHAP-based explanations.

Features
Manual prediction using diagnostic measurements
Batch prediction from CSV uploads
Probability estimates for benign and malignant outcomes
Basic risk-level indicator
Model explanation section forfeature contribution insights
Downloadable CSV template and prediction results
Project Structure
app.py: Main Streamlit application
models/final_svm_model.pkl: Trained classification model
models/scaler.pkl: Preprocessing scaler
notebooks/01_eda.ipynb: Exploratory data analysis notebook
notebooks/app/app.py: Streamlit app source copy

Requirements
Python 3.10 or newer recommended
Streamlit
NumPy
Pandas
Joblib
scikit-learn
SHAP

Installation
Clone the repository
git clone https://github.com/Rishikesh106/Breast-Cancer-Detection-using-Machine-Learning.git
cd Breast-Cancer-Detection-using-Machine-Learning

Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install streamlit numpy pandas joblib scikit-learn shap

If you already have a requirements file, install from it instead:

pip install -r requirements.txt

Run the App
streamlit run app.py

If your Streamlit file is inside the notebooks/app folder, run:

streamlit run app.py

How It Works
The app loads a pre-trained Support Vector Machine model and a scaler from the models folder.
User input features are scaled before prediction.
The model returns the predicted class and class probabilities.
For batch mode, uploaded CSV files must match the expected feature columns exactly.
A Random Forest model is used to provide SHAP-based explanation visuals.
Expected CSV Format
The uploaded CSV must contain the breast cancer feature columns in the same order as the training dataset. The app provides a downloadable template to help with the correct format.

Disclaimer
This project is intended for educational and demonstration purposes only. It is not a substitute for professional medical diagnosis, clinical judgment, or healthcare advice.

Model Performance
Based on the current project notes:

Test Accuracy: approximately 97%
ROC-AUC: approximately 0.995


