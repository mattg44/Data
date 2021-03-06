import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("billets.csv", sep=";")
df = df.dropna()

# Définition du jeu de données
X = df.drop("is_genuine", axis=1)
y = df["is_genuine"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=0.20)

rf = RandomForestClassifier(criterion="gini", max_depth=4, max_features="auto", n_estimators=100)
rf.fit(X_train, y_train)

st.title('Bank notes analysis')
st.write("Select a csv file with the measures of bank notes to be checked.")

uploaded_file = st.file_uploader("Upload Files", type=['csv'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type,
                    "FileSize": uploaded_file.size}
    st.write(file_details)


# File manipulations
if uploaded_file is not None:
    try:
        df_verif = pd.read_csv(uploaded_file)
        st.write("Uploaded csv file:")
        st.write(df_verif)

        id = df_verif["id"]
        df_verif.drop("id", axis=1, inplace=True)

        # Checking file
        df11 = pd.DataFrame(rf.predict(df_verif))
        df11.columns = ["is_genuine"]
        df11["is_genuine"] = np.where(df11["is_genuine"] == 1, True, False)
        df12 = pd.DataFrame(rf.predict_proba(df_verif))
        df12.columns = ["proba_false (%)", "proba_true (%)"]
        df12["proba_false (%)"] = round(df12["proba_false (%)"] * 100, 4)
        df12["proba_true (%)"] = round(df12["proba_true (%)"] * 100, 4)
        df12["id"] = id
        st.subheader("Bank notes status and probability:")
        st.write(df11.join(df12))
    except:
        st.header("Your file doesn't fit the requirements.")
        df_mask = pd.read_csv("mask.csv")
        st.write("CSV file required with ',' as separator and variables as below:")
        st.write(df_mask)



