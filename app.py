import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="Segmentation Clients - Dynamique", layout="wide")

# Titre
st.title("🛒 Segmentation Clients Centre Commercial")
st.markdown("**Dashboard DYNAMIQUE K-means** | Dataset 2000 clients | k variable")

# Sidebar contrôles
st.sidebar.header("⚙️ Paramètres dynamiques")
k_clusters = st.sidebar.slider("**k clusters**", 2, 10, 5, 
                              help="Observez Silhouette changer en live!")
show_analysis = st.sidebar.checkbox("📊 Analyse K optimal", True)
show_3d = st.sidebar.checkbox("3D", False)
export = st.sidebar.button("💾 Excel")

# Charger données
@st.cache_data
def load_data():
    df = pd.read_csv('Mall-Customers.csv')
    st.info(f"✅ **{len(df)} clients** chargés")
    return df

df = load_data()

# Features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].copy()

# ANALYSE K OPTIMAL POUR ALGORITHME
@st.cache_data
def analyze_k_range(X, max_k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias, silhouettes = [], []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    k_optimal = np.argmax(silhouettes) + 2
    return {
        'k_range': range(2, max_k+1),
        'inertias': inertias, 'silhouettes': silhouettes,
        'k_optimal': k_optimal, 'best_score': max(silhouettes)
    }

if show_analysis:
    st.subheader("📊 Analyse automatique K optimal")
    results = analyze_k_range(X)
    
    # Graphique Elbow + Silhouette
    fig_analysis = go.Figure()
    fig_analysis.add_trace(go.Scatter(
        x=list(results['k_range']), y=results['inertias'],
        mode='lines+markers', name='Inertia (Elbow)', line=dict(color='blue')))
    fig_analysis.add_trace(go.Scatter(
        x=list(results['k_range']), y=results['silhouettes'],
        mode='lines+markers', name='Silhouette Score', line=dict(color='red')))
    fig_analysis.add_vline(x=results['k_optimal'], line_dash="dash", 
                          annotation_text=f"k optimal: {results['k_optimal']}", 
                          line_color="green")
    fig_analysis.update_layout(
        title="Elbow Method + Silhouette Score", height=400,
        xaxis_title="Nombre de clusters (k)")
    st.plotly_chart(fig_analysis, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("**k optimal**", results['k_optimal'])
        st.metric("Meilleur Silhouette", f"{results['best_score']:.3f}")
    with col2:
        st.info(f"**Votre k={k_clusters}** → Silhouette: {results['silhouettes'][k_clusters-2]:.3f}")
        if k_clusters == results['k_optimal']:
            st.success("✅ k choisi = k optimal !")

# CLUSTERING DYNAMIQUE (k variable)
st.subheader(f"🔄 Clustering live k={k_clusters}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Métriques live
silhouette_live = silhouette_score(X_scaled, df['Cluster'])
inertia_live = kmeans.inertia_

col1, col2, col3, col4 = st.columns(4)
col1.metric("Silhouette", f"{silhouette_live:.3f}")
col2.metric("Inertia", f"{inertia_live:.0f}")
col3.metric("Clusters", k_clusters)
col4.metric("Clients", f"{len(df):,}")

# Graphique principal DYNAMIQUE
st.subheader("📈 Segmentation dynamique")
fig_main = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                     color='Cluster', size='Age', hover_data=['Age', 'Gender'],
                     title=f'k={k_clusters} live | Silhouette {silhouette_live:.3f}',
                     labels={'Annual Income (k$)':'Revenu (k$)',
                            'Spending Score (1-100)':'Score dépense'})
st.plotly_chart(fig_main, use_container_width=True)

# Vue 3D
if show_3d:
    st.subheader("3D interactive")
    fig_3d = px.scatter_3d(df, 
                          x='Age', 
                          y='Annual Income (k$)', 
                          z='Spending Score (1-100)',
                          color='Cluster', 
                          size='Age',  # Colonne existante
                          title=f'3D k={k_clusters}')
    st.plotly_chart(fig_3d, use_container_width=True)

# STATS
st.subheader("📋 Statistiques par cluster (live)")
cluster_stats = df.groupby('Cluster')[features].agg(['mean', 'count']).round(1)
st.dataframe(cluster_stats, use_container_width=True)

# PROFILS
st.subheader("🎯 Profils auto-générés")

def generate_profile(cluster_data):
    age_m = cluster_data['Age'].mean()
    inc_m = cluster_data['Annual Income (k$)'].mean()
    spend_m = cluster_data['Spending Score (1-100)'].mean()
    size = len(cluster_data)
    pct = size/len(df)*100
    
    age_label = "Seniors" if age_m>50 else "Adultes" if age_m>35 else "Jeunes"
    spend_label = "Big Spenders" if spend_m>70 else "Économes" if spend_m<35 else "Moyens"
    income_label = "Riches" if inc_m>80 else "Modestes" if inc_m<50 else "Moyens"
    
    nom = f"{age_label} {spend_label} {income_label}"
    desc = f"{age_m:.0f} ans | {inc_m:.0f}k$ | Dépense {spend_m:.0f}"
    desc += f"\n{size} clients ({pct:.1f}%)"
    
    if spend_m > 70:
        strat = "VIP | Premium | Exclusivité"
    elif spend_m < 35:
        strat = "Activation | Offres ciblées"
    elif age_m < 30:
        strat = "Digital | Influenceurs | Promo"
    elif inc_m > 80:
        strat = "Luxe | Personnalisation"
    else:
        strat = "Fidélisation | Loyalty"
    
    return nom, desc, strat

# Affichage DYNAMIQUE
cols = st.columns(3)
for i, cid in enumerate(range(k_clusters)):
    cluster_data = df[df['Cluster']==cid]
    if len(cluster_data)>0:
        nom, desc, strat = generate_profile(cluster_data)
        with cols[i%3]:
            st.metric(f"C{cid}", f"{len(cluster_data)}")
            st.markdown(f"**{nom}**")
            st.caption(desc)
            st.caption(strat)

# Pie chart
fig_pie = px.pie(df, names='Cluster', title='Répartition dynamique')
st.plotly_chart(fig_pie, use_container_width=True)

# PRÉDICTION DYNAMIQUE
st.markdown("---")
st.subheader("🔮 Prédiction live")

col1,col2,col3,col4 = st.columns([1,1,1,1])
new_age = col1.slider("Âge", 18,70,35)
new_inc = col2.slider("Revenu k$",15,150,60)
new_spend = col3.slider("Score dépense",1,100,50)

if col4.button("**Prédire**", type="primary"):
    new_data = scaler.transform([[new_age, new_inc, new_spend]])
    pred_cluster = kmeans.predict(new_data)[0]
    cluster_pred = df[df['Cluster']==pred_cluster]
    nom, desc, strat = generate_profile(cluster_pred)
    
    colr1, colr2 = st.columns([1,2])
    with colr1:
        st.metric("Cluster", f"C{pred_cluster}")
        st.metric("Taille", f"{len(cluster_pred)}")
    with colr2:
        st.markdown(f"**{nom}**")
        st.markdown(desc)
        st.markdown(f"**{strat}**")
    
    fig_pred = px.scatter(cluster_pred, x='Annual Income (k$)', y='Spending Score (1-100)')
    fig_pred.add_scatter(x=[new_inc], y=[new_spend], mode='markers',
                        marker=dict(color='red', size=20),
                        name="Nouveau client")
    st.plotly_chart(fig_pred)
    st.success("✅ Prédiction terminée!")

# Export
if export:
    export_data = df.groupby('Cluster')[features].agg(['mean','count']).round(1)
    filename = f"clusters_k{k_clusters}.xlsx"
    export_data.to_excel(filename)
    st.success(f"✅ **{filename}**")

# Footer
st.markdown("---")
st.caption(f"**Live** k={k_clusters} | Silhouette {silhouette_live:.3f} | {len(df)} clients")
