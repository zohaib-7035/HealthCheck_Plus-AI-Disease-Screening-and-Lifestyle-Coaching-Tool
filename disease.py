import pickle 
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
import matplotlib.pyplot as plt
import numpy as np

st.markdown("<h1 style='text-align: center; color: #0078D7;'>🧠 Multi-Disease Prediction System</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        h1, h2, h3 {
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)


diabetes_model = pickle.load(open('C:/Users/asifn/OneDrive/Desktop/Multiple Prediction System/diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('C:/Users/asifn/OneDrive/Desktop/Multiple Prediction System/heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('C:/Users/asifn/OneDrive/Desktop/Multiple Prediction System/parkinsons_model.sav','rb'))


def get_health_recommendations(condition_name):
    API_KEY = "AIzaSyCRR1sgVdjFtzhU98v-VKS3Q2Pdsz-EdeQ"  # Replace with your Gemini API key
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    prompt_text = f"""
    Provide a detailed health recommendation for a person with {condition_name}. 
    Include:
    - Diet plans
    - Exercise routines
    - Medication reminders
    - Specialist suggestions
    """

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt_text
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except:
            return "No detailed recommendation returned by Gemini."
    else:
        return f"Error {response.status_code}: {response.text}"


def chat_with_gemini(user_input):
    API_KEY = "AIzaSyCRR1sgVdjFtzhU98v-VKS3Q2Pdsz-EdeQ"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [{"text": user_input}]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except:
            return "❌ Could not get a proper reply from Gemini."
    else:
        return f"❌ Error {response.status_code}: {response.text}"






def plot_user_vs_average_matplotlib(user_input, average_data):
    labels = list(average_data.keys())
    user_values = user_input.copy()
    avg_values = list(average_data.values())

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    user_values += user_values[:1]
    avg_values += avg_values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, user_values, color='blue', linewidth=2, label='Your Input')
    ax.fill(angles, user_values, color='blue', alpha=0.25)

    ax.plot(angles, avg_values, color='green', linewidth=2, label='Average Patient')
    ax.fill(angles, avg_values, color='green', alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)

def plot_risk_trend_matplotlib():
    glucose_vals = np.linspace(50, 200, 100)
    risk = [1 / (1 + np.exp(-0.1 * (g - 125))) for g in glucose_vals]

    fig, ax = plt.subplots()
    ax.plot(glucose_vals, risk, color='red', linewidth=2)
    ax.set_title('Diabetes Risk vs Glucose Level')
    ax.set_xlabel('Glucose Level')
    ax.set_ylabel('Predicted Risk')

    st.pyplot(fig)




with st.sidebar:
    selected = option_menu('Multiple Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)
    
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    with st.expander("📥 Fill Patient Information"):
     col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

      user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]

      user_input = [float(x) for x in user_input]

      diab_prediction = diabetes_model.predict([user_input])

      if diab_prediction[0] == 1:
        diab_diagnosis = 'The person is diabetic'
        st.success(diab_diagnosis)
        st.markdown("---")

        with st.spinner("Generating AI-powered health recommendations..."):
            advice = get_health_recommendations("diabetes")
        st.subheader("💡 AI-Powered Health Recommendations")
        with st.expander("📋 View Recommendations"):
         st.info(advice)
         st.markdown("---")



        # 📊 Risk Comparison Chart
        average_diabetes = {
            'Pregnancies': 2.5, 'Glucose': 120, 'BloodPressure': 70,
            'SkinThickness': 20, 'Insulin': 80, 'BMI': 30,
            'DiabetesPedigreeFunction': 0.5, 'Age': 35
        }

        st.subheader("📊 Risk Analysis")
        tab1, tab2 = st.tabs(["📍 Risk Factor Comparison", "📈 Glucose Risk Curve"])

        with tab1:
         plot_user_vs_average_matplotlib(user_input.copy(), average_diabetes)

        with tab2:
         plot_risk_trend_matplotlib()
         st.markdown("---")



      else:
        diab_diagnosis = 'The person is not diabetic'
        st.success(diab_diagnosis)

    


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    st.title('❤️ Heart Disease Prediction using ML')

    with st.expander("📥 Fill Patient Information"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age')
        with col2:
            sex = st.text_input('Sex')
        with col3:
            cp = st.text_input('Chest Pain types')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
        with col3:
            exang = st.text_input('Exercise Induced Angina')

        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')

        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            st.success('The person is having heart disease')
            st.markdown("---")

            with st.spinner("Generating AI-powered health recommendations..."):
                advice = get_health_recommendations("heart disease")

            st.subheader("💡 AI-Powered Health Recommendations")
            with st.expander("📋 View Recommendations"):
                st.info(advice)

            st.markdown("---")
            average_heart = {
                'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 130,
                'chol': 240, 'fbs': 0, 'restecg': 1, 'thalach': 150,
                'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 1
            }

            st.subheader("📊 Risk Analysis")
            tab = st.tabs(["📍 Risk Factor Comparison"])[0]
            with tab:
                plot_user_vs_average_matplotlib(user_input.copy(), average_heart)

        else:
            st.success('The person does not have any heart disease')

    
if selected == "Parkinsons Prediction":

    st.title("🧠 Parkinson's Disease Prediction using ML")

    with st.expander("📥 Fill Patient Information"):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')
        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')
        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')
        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')
        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

        with col1:
            RAP = st.text_input('MDVP:RAP')
        with col2:
            PPQ = st.text_input('MDVP:PPQ')
        with col3:
            DDP = st.text_input('Jitter:DDP')
        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')
        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')
        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')
        with col3:
            APQ = st.text_input('MDVP:APQ')
        with col4:
            DDA = st.text_input('Shimmer:DDA')
        with col5:
            NHR = st.text_input('NHR')

        with col1:
            HNR = st.text_input('HNR')
        with col2:
            RPDE = st.text_input('RPDE')
        with col3:
            DFA = st.text_input('DFA')
        with col4:
            spread1 = st.text_input('spread1')
        with col5:
            spread2 = st.text_input('spread2')

        with col1:
            D2 = st.text_input('D2')
        with col2:
            PPE = st.text_input('PPE')

    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]
        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            st.success("The person has Parkinson's disease")
            st.markdown("---")

            with st.spinner("Generating AI-powered health recommendations..."):
                advice = get_health_recommendations("Parkinson's disease")

            st.subheader("💡 AI-Powered Health Recommendations")
            with st.expander("📋 View Recommendations"):
                st.info(advice)

            st.markdown("---")
            average_parkinsons = {
                'fo': 150, 'fhi': 180, 'flo': 90, 'Jitter_percent': 0.005,
                'Jitter_Abs': 0.00005, 'RAP': 0.003, 'PPQ': 0.004,
                'DDP': 0.009, 'Shimmer': 0.03, 'Shimmer_dB': 0.3,
                'APQ3': 0.02, 'APQ5': 0.025, 'APQ': 0.03, 'DDA': 0.04,
                'NHR': 0.02, 'HNR': 22, 'RPDE': 0.45, 'DFA': 0.7,
                'spread1': -5, 'spread2': 0.1, 'D2': 2.4, 'PPE': 0.2
            }

            st.subheader("📊 Risk Analysis")
            tab = st.tabs(["📍 Risk Factor Comparison"])[0]
            with tab:
                plot_user_vs_average_matplotlib(user_input.copy(), average_parkinsons)

        else:
            st.success("The person does not have Parkinson's disease")




st.markdown("---")
st.header("💬  AI Chat Assistant for Health Queries")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["bot"])

user_input = st.chat_input("Ask me anything about diseases, health, or lifestyle...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("💡 Thinking..."):
        bot_response = chat_with_gemini(user_input)

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
