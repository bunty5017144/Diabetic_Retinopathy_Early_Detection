# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
#import pandas as pd
import os
import requests 
import config
import pickle
import io
from PIL import Image 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd 
# ==============================================================================================
# ==============================================================================================
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

gmail_list=[]
password_list=[]
gmail_list1=[]
password_list1=[]
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Load the dataset
#df = pd.read_csv('balanced_seizure_dataset_with_ids.csv')

# Drop unnamed index column if present
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Load the trained model
#with open("rf_model.pkl", "rb") as file:
#    loaded_model = pickle.load(file)



#disease_dic= ["Eye Spot","Healthy Leaf","Red Leaf Spot","Redrot","Ring Spot"]



from model_predict2  import pred_skin_disease

from model_predict2un import pred_skin_disease3

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page



#@ app.route('/')
#def home():
#    title = 'Multiple cancer Identification using Deeplearning'
#    return render_template('index.html', title=title)  
@app.route('/')
def home():
    return render_template('home1.html')        


 

@app.route('/patient',methods=['POST',"GET"])
def patient():
    return render_template('login44.html')    


@app.route('/admin',methods=['POST',"GET"])
def admin():
    return render_template('login442.html') 

@app.route('/register22',methods=['POST',"GET"])
def register22():
    return render_template('register442.html') 

@app.route('/register2',methods=['POST',"GET"])
def register2():
    return render_template('register44.html')  
import pickle
@app.route('/logedin',methods=['POST'])
def logedin():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
    

    name =int_features3[0]

    # Save to a file
    with open("name.pkl", "wb") as f:
        pickle.dump(name, f)

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('index.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               


@app.route('/logedin2',methods=['POST'])
def logedin2():
    
    int_features3 = [str(x) for x in request.form.values()]
    print(int_features3)
    logu=int_features3[0]
    passw=int_features3[1]
   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root","","ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list)
    

    cursor1= db.cursor()
    cursor1.execute("SELECT password FROM user_register")
    result2=cursor1.fetchall()
              #print(result1)
              #print(gmail1)
    for row2 in result2:
                      print(row2)
                      print(row2[0])
                      password_list.append(str(row2[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(password_list)
    print(gmail_list.index(logu))
    print(password_list.index(passw))
    
    if gmail_list.index(logu)==password_list.index(passw):
        return render_template('patient_info.html')
    else:
        return jsonify({'result':'use proper  gmail and password'})
                  
                                               

import pandas as pd
from flask import request, render_template

@app.route('/get-patient-info', methods=['GET', 'POST'])
def get_patient_info():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')

        try:
            # Load the CSV file
            df = pd.read_csv('patient_data.csv')

            # Look for the row with matching patient ID
            row = df[df['patient_id'] == patient_id]

            if not row.empty:
                # Convert the row to dictionary (all columns)
                data = row.iloc[0].to_dict()
                
                # Optional: Join hospital lists if they are stored as comma-separated strings
                if 'india_hospitals' in data:
                    data['india_hospitals'] = str(data['india_hospitals'])
                if 'usa_hospitals' in data:
                    data['usa_hospitals'] = str(data['usa_hospitals'])

                return render_template('patient_info.html', data=data)
            else:
                return render_template('patient_info.html', error="Patient ID not found.")

        except Exception as e:
            return render_template('patient_info.html', error=f"Error: {str(e)}")

    # For GET request, just show the form
    return render_template('patient_info.html')


    

              
              # int_features3[0]==12345 and int_features3[1]==12345:
               #                      return render_template('index.html')
        
@app.route('/register',methods=['POST'])
def register():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login44.html')

@app.route('/register24',methods=['POST'])
def register24():
    

    int_features2 = [str(x) for x in request.form.values()]
    #print(int_features2)
    #print(int_features2[0])
    #print(int_features2[1])
    r1=int_features2[0]
    print(r1)
    
    r2=int_features2[1]
    print(r2)
    logu1=int_features2[0]
    passw1=int_features2[1]
        
    

    

   # if int_features2[0]==12345 and int_features2[1]==12345:

    import MySQLdb


# Open database connection
    db = MySQLdb.connect("localhost","root",'',"ddbb" )

# prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1=cursor.fetchall()
              #print(result1)
              #print(gmail1)
    for row1 in result1:
                      print(row1)
                      print(row1[0])
                      gmail_list1.append(str(row1[0]))
                      
                      #gmail_list.append(row1[0])
                      #value1=row1
                      
    print(gmail_list1)
    if logu1 in gmail_list1:
                      return jsonify({'result':'this gmail is already in use '})  
    else:

                  #return jsonify({'result':'this  gmail is not registered'})
              

# Prepare SQL query to INSERT a record into the database.
                  sql = "INSERT INTO user_register(user,password) VALUES (%s,%s)"
                  val = (r1, r2)
   
                  try:
   # Execute the SQL command
                                       cursor.execute(sql,val)
   # Commit your changes in the database
                                       db.commit()
                  except:
   # Rollback in case there is any error
                                       db.rollback()

# disconnect from server
                  db.close()
                 # return jsonify({'result':'succesfully registered'})
                  return render_template('login442.html')                      
# render crop recommendation form page

from flask import request, render_template
from PIL import Image
import os
import pandas as pd
from datetime import datetime

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Diabetic Retinopathy Detection using Deep Learning'

    if request.method == 'POST':
        file = request.files.get('file')

        # Load patient name/id from pickle
        with open("name.pkl", "rb") as f:
            patient_id = pickle.load(f)

        print("Patient ID:", patient_id)
        if not file or not patient_id:
            return render_template('rust.html', title=title)

        # Save uploaded image
        img = Image.open(file)
        img.save('output.png')

        #prediction2, confidencescore12 = pred_skin_disease3("output.png")
        #print("Prediction result:", prediction2)
        #if prediction2=="unknown":
        #    return render_template('error_page.html')

        # Make prediction
        prediction, confidencescore1 = pred_skin_disease("output.png")  # Your model prediction
        print("Prediction result:", prediction)

        # --- Detailed Disease Database (Diabetic Retinopathy) ---
        disease_info = {
            "no_dr": {
                "cause": "No signs of diabetic retinopathy. Retina appears healthy.",
                "treatment": "Maintain normal blood sugar and blood pressure levels.",
                "homeopathy": "Syzygium jambolanum, Gymnema sylvestre.",
                "allopathy": "Regular eye check-up, control of diabetes with insulin or metformin.",
                "ayurveda": "Neem, karela (bitter gourd), amla juice for sugar control.",
                "india_hospitals": ["AIIMS Delhi", "LV Prasad Eye Institute Hyderabad", "Aravind Eye Hospital Madurai"],
                "usa_hospitals": ["Bascom Palmer Eye Institute", "Johns Hopkins Wilmer Eye Institute", "Mayo Clinic"],
                "cost_india": "₹10,000–25,000 (annual check-ups)",
                "cost_usa": "$300–1,000 (annual monitoring)",
                "success_rate": "100% (no damage if maintained)"
            },
            "mild": {
                "cause": "Small swelling in retina’s blood vessels due to early diabetes effects.",
                "treatment": "Control sugar levels, monitor progression every 6 months.",
                "homeopathy": "Phosphorus, Natrum muriaticum.",
                "allopathy": "Blood sugar control through medication and diet.",
                "ayurveda": "Triphala, turmeric, and gokshura for improving microcirculation.",
                "india_hospitals": ["Sankara Nethralaya Chennai", "AIIMS Delhi", "Aravind Eye Hospital Madurai"],
                "usa_hospitals": ["Cleveland Clinic", "Mayo Clinic", "Johns Hopkins Hospital"],
                "cost_india": "₹25,000–60,000",
                "cost_usa": "$1,000–2,500",
                "success_rate": "95–98% (with timely management)"
            },
            "moderate": {
                "cause": "Blocked retinal blood vessels and leakage cause mild vision blurring.",
                "treatment": "Laser therapy or anti-VEGF injections to prevent further damage.",
                "homeopathy": "Arsenicum album, Aurum metallicum.",
                "allopathy": "Intravitreal injections like ranibizumab or aflibercept.",
                "ayurveda": "Guduchi and turmeric to reduce inflammation.",
                "india_hospitals": ["LV Prasad Eye Institute", "Aravind Eye Hospital", "Apollo Hospitals Chennai"],
                "usa_hospitals": ["Bascom Palmer Eye Institute", "Mayo Clinic", "Stanford Health Care"],
                "cost_india": "₹80,000–1.5 lakhs",
                "cost_usa": "$2,000–5,000 per eye",
                "success_rate": "85–90% (vision preservation with treatment)"
            },
            "severe": {
                "cause": "Many blood vessels are blocked, reducing oxygen supply to the retina.",
                "treatment": "Panretinal photocoagulation (laser treatment) and anti-VEGF therapy.",
                "homeopathy": "Phosphorus, Aurum muriaticum.",
                "allopathy": "Laser surgery, anti-VEGF injections, or corticosteroid implants.",
                "ayurveda": "Triphala ghee and gokshura-based formulations.",
                "india_hospitals": ["AIIMS Delhi", "LV Prasad Hyderabad", "Aravind Eye Hospital Madurai"],
                "usa_hospitals": ["Johns Hopkins Hospital", "Mayo Clinic", "Cleveland Clinic"],
                "cost_india": "₹1.5–3 lakhs",
                "cost_usa": "$5,000–12,000",
                "success_rate": "75–85% (depends on stage)"
            },
            "proliferative_dr": {
                "cause": "Advanced stage with abnormal new vessel growth causing bleeding and detachment.",
                "treatment": "Vitrectomy surgery, laser therapy, and anti-VEGF injections.",
                "homeopathy": "Phosphorus, Secale cornutum.",
                "allopathy": "Surgical vitrectomy with retinal laser or cryotherapy.",
                "ayurveda": "Herbal retina detox with triphala, turmeric, and amalaki.",
                "india_hospitals": ["Aravind Eye Hospital", "LV Prasad Eye Institute", "AIIMS Delhi"],
                "usa_hospitals": ["Bascom Palmer Eye Institute", "Johns Hopkins Hospital", "UCLA Stein Eye Institute"],
                "cost_india": "₹2–4 lakhs",
                "cost_usa": "$8,000–20,000",
                "success_rate": "60–75% (depends on vision loss severity)"
            }
        }

        # --- Get Predicted Disease Details ---
        prediction_type = prediction.lower().strip()
        details = disease_info.get(prediction_type, {})

        # Extract details safely
        cause = details.get("cause", "Information not available.")
        treatment = details.get("treatment", "Information not available.")
        homeopathy = details.get("homeopathy", "Information not available.")
        allopathy = details.get("allopathy", "Information not available.")
        ayurveda = details.get("ayurveda", "Information not available.")
        india_hospitals = ", ".join(details.get("india_hospitals", []))
        usa_hospitals = ", ".join(details.get("usa_hospitals", []))
        cost_india = details.get("cost_india", "N/A")
        cost_usa = details.get("cost_usa", "N/A")
        success_rate = details.get("success_rate", "N/A")

        # --- CSV Database Update ---
        csv_file = 'patient_data.csv'
        columns = [
            'patient_id', 'skin_cancer', 'date', 'cause', 'treatment',
            'homeopathy', 'allopathy', 'ayurveda', 'india_hospitals',
            'usa_hospitals', 'cost_india', 'cost_usa', 'success_rate'
        ]

        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.read_csv(csv_file)

        current_date = datetime.now().strftime("%Y-%m-%d")

        new_data = {
            'patient_id': patient_id,
            'skin_cancer': prediction_type,
            'date': current_date,
            'cause': cause,
            'treatment': treatment,
            'homeopathy': homeopathy,
            'allopathy': allopathy,
            'ayurveda': ayurveda,
            'india_hospitals': india_hospitals,
            'usa_hospitals': usa_hospitals,
            'cost_india': cost_india,
            'cost_usa': cost_usa,
            'success_rate': success_rate
        }

        if patient_id in df['patient_id'].values:
            idx = df[df['patient_id'] == patient_id].index[0]
            for key, value in new_data.items():
                df.at[idx, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        df.to_csv(csv_file, index=False)

        # --- Render Template with All Info ---
        return render_template(
            'rust-result.html',
            prediction=prediction.capitalize(),
            cause=cause,
            treatment=treatment,
            homeopathy=homeopathy,
            allopathy=allopathy,
            ayurveda=ayurveda,
            india_hospitals=india_hospitals,
            usa_hospitals=usa_hospitals,
            cost_india=cost_india,
            cost_usa=cost_usa,
            success_rate=success_rate,
            title="Diabetic Retinopathy Diagnosis Result"
        )

    return render_template('rust.html', title=title)














# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
