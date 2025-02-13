import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# Configuration page
st.set_page_config(page_title="Analyse ML", layout="wide")

# Initialisation de la session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Page d'accueil
def home_page():
    st.title("🙂 Bienvenue dans mon application d'analyse ML")
    st.write("""
    Cette application te permet d'explorer deux jeux de données  : Iris et Wine.
    - Pour Iris : Clustering avec KMeans (3 clusters)
    - Pour Wine : Classification avec régression logistique
    Vous pourrez visualiser les données et faire des prédictions.
    """)

    st.write("### Tu veux utiliser quelle base ?")
    dataset_choice = st.selectbox("", ["Sélectionner une base", "Iris", "Wine"])

    if dataset_choice == "Iris":
        st.session_state.page = 'iris'
        st.rerun()
    elif dataset_choice == "Wine":
        st.session_state.page = 'wine'
        st.rerun()


# Page Iris
def iris_page():
    st.markdown("<h1 style='text-align: center;'> Analyse du jeu de données Iris</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    
    # EDA
    st.subheader("Analyse exploratoire")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Aperçu des données")
        st.dataframe(data.head())
    with col2:
        st.write("Statistiques descriptives")
        st.dataframe(data.describe())
    
    # Visualisations
    st.subheader("Visualisations")
    
    # Distribution
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for idx, feature in enumerate(iris.feature_names):
        sns.histplot(data=data, x=feature, hue='target', ax=ax[idx//2, idx%2])
    st.pyplot(fig)
    
    # Matrice de corrélation
    st.write("Matrice de corrélation")
    corr = data.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig)
    
    # Activité
    activity = st.selectbox("Choisir l'activité:", ["Clustering", "Prédiction"])
    
    if activity == "Clustering":
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(data.drop('target', axis=1))
        
        fig = px.scatter_3d(data, x=iris.feature_names[0], 
                           y=iris.feature_names[1], 
                           z=iris.feature_names[2],
                           color=clusters)
        st.plotly_chart(fig)
        
    else:
        st.subheader("Prédiction")
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Sepal length:", 0.0, 10.0, 5.0)
            sepal_width = st.number_input("Sepal width:", 0.0, 10.0, 3.0)
        with col2:
            petal_length = st.number_input("Petal length:", 0.0, 10.0, 4.0)
            petal_width = st.number_input("Petal width:", 0.0, 10.0, 1.0)
        
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(data.drop('target', axis=1))
        prediction = kmeans.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        st.success(f"Cluster prédit: {prediction[0]}")

    if st.button("Retour à l'accueil"):
        st.session_state.page = 'home'
        st.rerun()

# Page Wine
def wine_page():
    st.markdown("<h1 style='text-align: center;'> Analyse du jeu de données Wine</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    wine = load_wine()
    data = pd.DataFrame(wine.data, columns=wine.feature_names)
    data['target'] = wine.target
    
    # EDA
    st.subheader("Analyse exploratoire")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Aperçu des données")
        st.dataframe(data.head())
    with col2:
        st.write("Statistiques descriptives")
        st.dataframe(data.describe())
    
    # Visualisations
    st.subheader("Visualisations")
    
    # Distribution des classes
    fig = px.pie(data, names='target', title='Distribution des classes de vin')
    st.plotly_chart(fig)
    
    # Boxplots
    fig = px.box(data, y=data.columns[:-1], title='Distribution des caractéristiques')
    st.plotly_chart(fig)
    
    # Activité
    activity = st.selectbox("Choisir l'activité:", ["Classification", "Prédiction"])
    
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2)
    
    if activity == "Classification":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.metric("Précision du modèle", f"{score:.2%}")
        
        fig = px.scatter_3d(data, x=wine.feature_names[0], 
                           y=wine.feature_names[1], 
                           z=wine.feature_names[2],
                           color=wine.target)
        st.plotly_chart(fig)
        
    else:
        st.subheader("Prédiction")
        features = []
        cols = st.columns(3)
        for i, name in enumerate(wine.feature_names):
            with cols[i % 3]:
                val = st.number_input(f"{name}:", 0.0, 100.0, 10.0)
                features.append(val)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        if st.button("Prédire"):
            prediction = model.predict([features])
            st.success(f"Classe de vin prédite: {prediction[0]}")

    if st.button("Retour à l'accueil"):
        st.session_state.page = 'home'
        st.rerun()

# Gestion de la navigation
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'iris':
    iris_page()
elif st.session_state.page == 'wine':
    wine_page()
