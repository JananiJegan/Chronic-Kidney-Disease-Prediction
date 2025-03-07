from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from django.http import HttpResponseRedirect
from .models import PatientData

# Create your views here.
def index(request):
    return render(request,"index.html")

def register(request):
    if request.method=="POST":
        first=request.POST['fname']
        last=request.POST['lname']
        uname=request.POST['uname']
        email=request.POST['email']
        pwd=request.POST['pwd']
        cpwd=request.POST['cpwd']
        if pwd==cpwd:
            if User.objects.filter(email=email).exists():
                messages.info(request,"Email exists")
                return render(request,"register.html")
            elif User.objects.filter(username=uname).exists():
                messages.info(request,"Username exists")
                return render(request,"register.html")
            else:
                user=User.objects.create_user(first_name=first,last_name=last,username=uname,email=email,password=cpwd)
                user.save()
                return HttpResponseRedirect('login')
        else:
            messages.info(request,"password did not match")
            return render(request,"register.html")
    else:
        return render(request,"register.html")
    
def login(request):
    if request.method=="POST":
        uname=request.POST['uname']
        pwd=request.POST['pwd']
        user=auth.authenticate(username=uname,password=pwd)
        if user is not None:
            auth.login(request,user)
            return HttpResponseRedirect('/')
        else:
            messages.info(request,"Invalid Credentials")
    return render(request,"login.html")

def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/')

import os
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import numpy as np

def data(request):
    os.environ['OMP_NUM_THREADS'] = '1'  # Set to avoid memory leak on Windows with MKL
    
    if request.method == "POST":
        # Collect form data from POST request
        age = int(request.POST.get('age'))
        gender = request.POST.get('gender')  # Should be 'M' or 'F'
        bmi = float(request.POST.get('bmi'))
        systolic_bp = int(request.POST.get('systolic_bp'))
        diastolic_bp = int(request.POST.get('diastolic_bp'))
        fasting_blood_sugar = float(request.POST.get('fasting_blood_sugar'))
        hba1c = float(request.POST.get('hba1c'))
        serum_creatinine = float(request.POST.get('serum_creatinine'))
        gfr = float(request.POST.get('gfr'))
        smoking = int(request.POST.get('smoking'))  # Binary: 0 (No) or 1 (Yes)
        
        # Load the dataset
        df = pd.read_csv("static/datasets/ckd_dataset.csv")

        # Check for null values and handle them using interpolation
        df.interpolate(inplace=True)

        # Filter the dataset to keep only relevant columns
        df = df[['Age', 'Gender', 'BMI', 'SystolicBP', 'DiastolicBP', 
                 'FastingBloodSugar', 'HbA1c', 'SerumCreatinine', 'GFR', 'Smoking', 'Diagnosis']]
        
        # Encode categorical features
        le = LabelEncoder()
        le.fit(['M', 'F'])  # Convert 'M' and 'F' to numerical values

        # Prepare features (X) and labels (y)
        X = df.drop(['Diagnosis'], axis=1)
        y = df['Diagnosis']

        # Train the Random Forest model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))

        # Prepare the input for prediction
        gender_encoded = le.transform([gender])[0]
        pred_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'BMI': [bmi],
            'SystolicBP': [systolic_bp],
            'DiastolicBP': [diastolic_bp],
            'FastingBloodSugar': [fasting_blood_sugar],
            'HbA1c': [hba1c],
            'SerumCreatinine': [serum_creatinine],
            'GFR': [gfr],
            'Smoking': [smoking]
        })

        # Make the prediction
        pred_outcome = model.predict(pred_input)

        # Initialize CKD stage to None (for non-CKD cases)
        ckd_stage_mapped = None

        # Predict CKD stage using KMeans based on GFR value only if CKD is predicted (outcome == 1)
        if pred_outcome == 1:  # If prediction is 1 (CKD)
            # Stage-based GFR values for clustering
            gfr_values = np.array([
                90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,  # Stage 1 values
                60, 61, 62, 63, 64, 65, 66, 67, 68, 69,       # Stage 2 values
                70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 
                80, 81, 82, 83, 84, 85, 86, 87, 88, 89,  
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,       # Stage 3 values
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24,       # Stage 4 values
                25, 26, 27, 28, 29, 
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,    # Stage 5 values
                13, 14
            ])
            
            gfr_values = gfr_values.reshape(-1, 1)

            # Fit the KMeans model
            kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
            kmeans.fit(gfr_values)
            print("Cluster Centers:", kmeans.cluster_centers_)
            # Predict CKD stage based on GFR
            ckd_stage = kmeans.predict([[gfr]])
            print("Input GFR for prediction:", gfr)
            print("Predicted CKD stage (cluster):", ckd_stage)
            # Map the predicted cluster to CKD stages (1 to 5)
            stage_mapping = {
                0: 1,  # Cluster 0 -> Stage 1 (highest GFR, lowest risk)
                1: 2,  # Cluster 1 -> Stage 2
                2: 3,  # Cluster 2 -> Stage 3
                3: 4,  # Cluster 3 -> Stage 4
                4: 5   # Cluster 4 -> Stage 5 (lowest GFR, highest risk)
            }
            ckd_stage_mapped = stage_mapping[ckd_stage[0]]
            print(ckd_stage_mapped)

        # Render the results in the template
        return render(request, "predict.html", {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "fasting_blood_sugar": fasting_blood_sugar,
            "hba1c": hba1c,
            "serum_creatinine": serum_creatinine,
            "gfr": gfr,
            "smoking": smoking,
            "prediction": pred_outcome,  # 0 or 1 for CKD prediction
            "ckd_stage": ckd_stage_mapped  # Only provided if CKD is predicted (1), else None
        })
    
    return render(request, "data.html")

def predict(request):
    return render(request, 'predict.html')

import pandas as pd
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

def recommend_hospital(request):
    context = {
        'hospitals': None,  # Initialize hospitals as None
        'message': None,  # Initialize message as None
    }

    if request.method == 'POST':
        city = request.POST.get('city').lower()  # Get the city input from the user
        state = request.POST.get('state').lower()  # Get the state input from the user

        # Load the dataset
        hospitals_df = pd.read_csv("static/datasets/HospitalsInIndia.csv")

        # Preprocess the data: convert city and state names to numeric labels
        le_city = LabelEncoder()
        le_state = LabelEncoder()
        
        # Encode city and state columns
        hospitals_df['City_encoded'] = le_city.fit_transform(hospitals_df['City'].str.lower())
        hospitals_df['State_encoded'] = le_state.fit_transform(hospitals_df['State'].str.lower())

        # Convert the input city and state into their numeric labels
        city_encoded = le_city.transform([city])[0]
        state_encoded = le_state.transform([state])[0]

        # Filter hospitals by the specified city and state
        filtered_hospitals = hospitals_df[
            (hospitals_df['City'].str.lower() == city) & 
            (hospitals_df['State'].str.lower() == state)
        ]

        if filtered_hospitals.empty:
            context['message'] = 'No hospitals found in the specified city and state.'
        else:
            context['message'] = 'Hospitals found:'

            # Create a KNN model using the filtered hospitals
            knn = NearestNeighbors(n_neighbors=min(3, len(filtered_hospitals)), metric='euclidean')
            knn.fit(filtered_hospitals[['City_encoded', 'State_encoded']])

            # Prepare input for kneighbors in DataFrame format
            input_data = pd.DataFrame([[city_encoded, state_encoded]], columns=['City_encoded', 'State_encoded'])

            # Find the nearest neighbors to the input city and state
            distances, indices = knn.kneighbors(input_data, n_neighbors=min(3, len(filtered_hospitals)))

            # Retrieve hospital details for the nearest neighbors
            nearest_hospitals = filtered_hospitals.iloc[indices[0]]

            # Pass hospital details to the context
            context['hospitals'] = nearest_hospitals.to_dict('records')

    return render(request, 'recommend.html', context)
