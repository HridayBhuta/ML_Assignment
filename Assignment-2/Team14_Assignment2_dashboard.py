"""
Assignment 2
Team: 14
Members:
- Hriday Bhuta 2023A2PS0901H
- Naman Jindal 2023AAPS1064H
- Archit Diwane 2023A3PS1361H
- Aryan Agarwal 2023A5PS1039H
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0; padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Automated ML Pipeline Dashboard</p>', unsafe_allow_html=True)
st.divider()

CSV_DIR = "csv"

@st.cache_data(show_spinner="Loading raw CSV files...")
def load_raw_data():
    patients = pd.read_csv(f"{CSV_DIR}/patients.csv", on_bad_lines='skip')
    conditions = pd.read_csv(f"{CSV_DIR}/conditions.csv", on_bad_lines='skip')
    observations = pd.read_csv(f"{CSV_DIR}/observations.csv", on_bad_lines='skip')
    encounters = pd.read_csv(f"{CSV_DIR}/encounters.csv", on_bad_lines='skip')
    return patients, conditions, observations, encounters


@st.cache_data(show_spinner="Building feature matrix...")
def build_features(_patients, _conditions, _observations, _encounters, split_date_str):
    patients = _patients.copy()
    conditions = _conditions.copy()
    observations = _observations.copy()
    encounters = _encounters.copy()

    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'], errors='coerce')
    conditions['START'] = pd.to_datetime(conditions['START'], format='mixed',
                                         dayfirst=True, errors='coerce')
    observations['DATE'] = pd.to_datetime(observations['DATE'], errors='coerce', utc=True)
    observations['DATE'] = observations['DATE'].dt.tz_localize(None)
    encounters['START'] = pd.to_datetime(encounters['START'], errors='coerce', utc=True)
    encounters['START'] = encounters['START'].dt.tz_localize(None)

    split_ts = pd.Timestamp(split_date_str)

    cond_medical = conditions[conditions['DESCRIPTION'].str.contains('disorder', case=False, na=False)].copy()

    cond_medical = cond_medical.merge(
        encounters[['Id', 'START']].rename(columns={'Id': 'ENCOUNTER', 'START': 'ENC_DATE'}),
        on='ENCOUNTER', how='left'
    )

    cond_ds1 = cond_medical[cond_medical['ENC_DATE'] < split_ts]
    cond_ds2 = cond_medical[cond_medical['ENC_DATE'] >= split_ts]

    top_conditions = cond_medical['DESCRIPTION'].value_counts().head(8).index.tolist()

    def get_target(cond_df, top_conds):
        cond_df = cond_df[cond_df['DESCRIPTION'].isin(top_conds)]
        if len(cond_df) == 0:
            return pd.DataFrame(columns=['PATIENT', 'CONDITION'])
        target = cond_df.groupby('PATIENT')['DESCRIPTION'].agg(
            lambda x: x.value_counts().index[0]
        ).reset_index().rename(columns={'DESCRIPTION': 'CONDITION'})
        return target

    target_ds1 = get_target(cond_ds1, top_conditions)
    target_ds2 = get_target(cond_ds2, top_conditions)

    num_obs = observations[observations['TYPE'] == 'numeric'].copy()
    num_obs['VALUE'] = pd.to_numeric(num_obs['VALUE'], errors='coerce')
    num_obs = num_obs.dropna(subset=['VALUE'])
    top_obs = num_obs['DESCRIPTION'].value_counts().head(15).index.tolist()
    num_obs = num_obs[num_obs['DESCRIPTION'].isin(top_obs)]

    def build_obs_features(obs_df, patient_list):
        obs_subset = obs_df[obs_df['PATIENT'].isin(patient_list)]
        if len(obs_subset) == 0:
            return pd.DataFrame({'PATIENT': patient_list})

        agg_mean = obs_subset.pivot_table(index='PATIENT', columns='DESCRIPTION',
                                           values='VALUE', aggfunc='mean')
        agg_mean.columns = [f"{c}_mean" for c in agg_mean.columns]

        agg_std = obs_subset.pivot_table(index='PATIENT', columns='DESCRIPTION',
                                          values='VALUE', aggfunc='std')
        agg_std.columns = [f"{c}_std" for c in agg_std.columns]

        return agg_mean.join(agg_std, how='outer').reset_index()

    ref_date = pd.Timestamp('2025-01-01')
    patients['AGE'] = (ref_date - patients['BIRTHDATE']).dt.days / 365.25
    patients['IS_ALIVE'] = patients['DEATHDATE'].isna().astype(int)

    demo = patients[['Id', 'AGE', 'GENDER', 'RACE', 'ETHNICITY', 'INCOME', 'IS_ALIVE']].copy()
    demo = demo.rename(columns={'Id': 'PATIENT'})

    le_gender = LabelEncoder()
    demo['GENDER'] = le_gender.fit_transform(demo['GENDER'].fillna('Unknown'))
    le_race = LabelEncoder()
    demo['RACE'] = le_race.fit_transform(demo['RACE'].fillna('Unknown'))
    le_eth = LabelEncoder()
    demo['ETHNICITY'] = le_eth.fit_transform(demo['ETHNICITY'].fillna('Unknown'))

    enc_count = encounters.groupby('PATIENT').size().reset_index(name='NUM_ENCOUNTERS')
    enc_class = encounters.groupby(['PATIENT', 'ENCOUNTERCLASS']).size().unstack(fill_value=0)
    enc_class.columns = [f'ENC_{c}' for c in enc_class.columns]
    enc_class = enc_class.reset_index()

    def build_dataset(target_df, demo, num_obs, enc_count, enc_class):
        patient_list = target_df['PATIENT'].tolist()
        obs_feat = build_obs_features(num_obs, patient_list)

        df = target_df.merge(demo, on='PATIENT', how='left')
        df = df.merge(obs_feat, on='PATIENT', how='left')
        df = df.merge(enc_count, on='PATIENT', how='left')
        df = df.merge(enc_class, on='PATIENT', how='left')
        df = df.fillna(0)
        return df

    ds1 = build_dataset(target_ds1, demo, num_obs, enc_count, enc_class)
    ds2 = build_dataset(target_ds2, demo, num_obs, enc_count, enc_class)

    all_cols = sorted(set(ds1.columns) | set(ds2.columns))
    for col in all_cols:
        if col not in ds1.columns:
            ds1[col] = 0
        if col not in ds2.columns:
            ds2[col] = 0
    ds1 = ds1[all_cols]
    ds2 = ds2[all_cols]

    return ds1, ds2, le_gender, le_race, le_eth, top_obs, top_conditions


def get_feature_target(df):
    exclude = ['PATIENT', 'CONDITION']
    features = [c for c in df.columns if c not in exclude]
    X = df[features].astype(float).values
    y = df['CONDITION'].values
    return X, y, features

st.sidebar.header("Pipeline Configuration")
split_date = st.sidebar.date_input("Temporal Split Date", value=pd.Timestamp("2020-01-01"))
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
random_state = st.sidebar.number_input("Random State", value=42)

st.sidebar.subheader("Decision Tree")
dt_max_depth = st.sidebar.slider("DT Max Depth", 2, 30, 8)
dt_min_samples = st.sidebar.slider("DT Min Samples Split", 2, 20, 3)

st.sidebar.subheader("SVM")
svm_C = st.sidebar.select_slider("SVM C", options=[0.01, 0.1, 1, 10, 100], value=10)
svm_kernel = st.sidebar.selectbox("SVM Kernel", ["rbf", "linear", "poly"], index=0)

st.sidebar.subheader("Neural Network (MLP)")
mlp_hidden = st.sidebar.text_input("MLP Hidden Layers", "1024,512")
mlp_max_iter = st.sidebar.slider("MLP Max Iterations", 100, 2000, 1000)
mlp_alpha = st.sidebar.select_slider("MLP Alpha", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)

run_pipeline = st.sidebar.button("Run Full Pipeline", type="primary", use_container_width=True)

tabs = st.tabs([
    "Data & EDA",
    "Feature Engineering",
    "Model Training",
    "Evaluation & Comparison",
    "Continual Learning",
    "Model Interpretation"
])

patients, conditions, observations, encounters = load_raw_data()
ds1, ds2, le_gender, le_race, le_eth, top_obs, top_conditions = build_features(
    patients, conditions, observations, encounters, str(split_date)
)

with tabs[0]:
    st.header("Data Loading & Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(patients):,}")
    col2.metric("Total Conditions", f"{len(conditions):,}")
    col3.metric("Observations", f"{len(observations):,}")
    col4.metric("Encounters", f"{len(encounters):,}")

    st.subheader("Temporal Split Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Split Date", str(split_date))
    c2.metric("Dataset 1 (Historical)", f"{len(ds1)} patients")
    c3.metric("Dataset 2 (Current)", f"{len(ds2)} patients")

    st.subheader("Dataset 1 (Historical) - Sample")
    display_cols = [c for c in ds1.columns if c != 'PATIENT']
    st.dataframe(ds1[display_cols].head(15), width="stretch")

    # Class distribution comparison
    st.subheader("Class Distribution Comparison")
    fig_cls = make_subplots(rows=1, cols=2, subplot_titles=["Dataset 1 (Historical)", "Dataset 2 (Current)"])
    ds1_vc = ds1['CONDITION'].value_counts().head(10)
    ds2_vc = ds2['CONDITION'].value_counts().head(10)
    fig_cls.add_trace(go.Bar(x=ds1_vc.index, y=ds1_vc.values, marker_color='#667eea', name='DS1'), row=1, col=1)
    fig_cls.add_trace(go.Bar(x=ds2_vc.index, y=ds2_vc.values, marker_color='#764ba2', name='DS2'), row=1, col=2)
    fig_cls.update_layout(height=450, showlegend=False)
    fig_cls.update_xaxes(tickangle=45)
    st.plotly_chart(fig_cls, width="stretch")

    # Data drift analysis
    st.subheader("Data Drift Analysis")
    st.markdown("""
    **Temporal Data Drift:** By splitting data based on encounter date, we can observe how
    the distribution of conditions and patient features changes over time. This is critical for
    understanding whether models trained on historical data remain valid for current predictions.
    """)

    # Feature distributions
    st.subheader("Feature Distributions (Dataset 1 vs 2)")
    num_cols = [c for c in ds1.columns if c not in ['PATIENT', 'CONDITION']]
    sel_feat = st.selectbox("Select Feature", num_cols[:20], index=0)
    if sel_feat:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=ds1[sel_feat].astype(float), name='Dataset 1', opacity=0.7, marker_color='#667eea'))
        fig_dist.add_trace(go.Histogram(x=ds2[sel_feat].astype(float), name='Dataset 2', opacity=0.7, marker_color='#f093fb'))
        fig_dist.update_layout(barmode='overlay', title=f"Distribution of {sel_feat}", height=400)
        st.plotly_chart(fig_dist, width="stretch")

    # Descriptive stats
    st.subheader("Descriptive Statistics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write("**Dataset 1 (Historical)**")
        desc1 = ds1.drop(columns=['PATIENT', 'CONDITION'], errors='ignore').describe().T.round(2)
        st.dataframe(desc1, width="stretch")
    with col_s2:
        st.write("**Dataset 2 (Current)**")
        desc2 = ds2.drop(columns=['PATIENT', 'CONDITION'], errors='ignore').describe().T.round(2)
        st.dataframe(desc2, width="stretch")

    # Correlation heatmap
    st.subheader("Correlation Heatmap (Dataset 1)")
    feature_cols = [c for c in ds1.columns if c not in ['PATIENT', 'CONDITION']]
    corr = ds1[feature_cols].astype(float).corr()
    fig_corr = px.imshow(corr, text_auto=".1f", color_continuous_scale="RdBu_r",
                         title="Feature Correlation Matrix", height=1050)
    fig_corr.update_layout(width=1500, margin=dict(l=140, r=140, t=90, b=140))
    fig_corr.update_xaxes(tickangle=45)
    left_pad, center_col, right_pad = st.columns([1, 8, 1])
    with center_col:
        st.plotly_chart(fig_corr, width="stretch")

with tabs[1]:
    st.header("Feature Engineering & Representation")

    st.subheader("Feature Categories")
    st.markdown("""
    | Category | Features | Description |
    |----------|----------|-------------|
    | **Demographics** | AGE, GENDER, RACE, ETHNICITY, INCOME, IS_ALIVE | Patient demographic & socio-economic info |
    | **Vital Signs (Mean)** | Body Height_mean, Body Weight_mean, Heart rate_mean, ... | Mean of numeric clinical observations |
    | **Vital Signs (Std)** | Body Height_std, Body Weight_std, Heart rate_std, ... | Variability in clinical measurements |
    | **Encounter Features** | NUM_ENCOUNTERS, ENC_ambulatory, ENC_emergency, ... | Healthcare utilization patterns |
    """)

    st.subheader("Feature Engineering Steps")
    st.markdown("""
    1. **Observation Aggregation**: Numeric observations are pivoted and aggregated per patient using mean and standard deviation
    2. **Categorical Encoding**: Gender, Race, Ethnicity encoded using LabelEncoder
    3. **Missing Value Handling**: Missing values filled with 0 (patients without certain observations)
    4. **Target Variable**: Most frequent medical condition (disorder) diagnosed for each patient
    5. **Temporal Split**: Data split into Historical (DS1) and Current (DS2) based on encounter date
    """)

    X1, y1, feat_names = get_feature_target(ds1)
    X2, y2, _ = get_feature_target(ds2)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Features", len(feat_names))
    c2.metric("DS1 Samples", X1.shape[0])
    c3.metric("DS2 Samples", X2.shape[0])

    st.subheader("Feature Names")
    st.write(feat_names)

    # Target conditions
    st.subheader("Target Classes (Medical Conditions)")
    st.write(top_conditions)

with tabs[2]:
    st.header("Model Training & Hyperparameter Tuning")

    if run_pipeline or 'models' in st.session_state:
        if run_pipeline or 'models' not in st.session_state:
            with st.spinner("Training models..."):
                X1, y1, feat_names = get_feature_target(ds1)
                X2, y2, _ = get_feature_target(ds2)

                # Align classes
                all_classes = sorted(set(list(y1) + list(y2)))
                le_target = LabelEncoder()
                le_target.fit(all_classes)
                y1_enc = le_target.transform(y1)
                y2_enc = le_target.transform(y2)

                # Stratified split only if enough samples per class
                def safe_stratify(y, min_count=2):
                    counts = Counter(y)
                    return y if all(c >= min_count for c in counts.values()) else None

                X1_train, X1_test, y1_train, y1_test = train_test_split(
                    X1, y1_enc, test_size=test_size, random_state=random_state,
                    stratify=safe_stratify(y1_enc)
                )
                X2_train, X2_test, y2_train, y2_test = train_test_split(
                    X2, y2_enc, test_size=test_size, random_state=random_state,
                    stratify=safe_stratify(y2_enc)
                )

                # Scale features
                scaler = StandardScaler()
                X1_train_s = scaler.fit_transform(X1_train)
                X1_test_s = scaler.transform(X1_test)
                X2_train_s = scaler.transform(X2_train)
                X2_test_s = scaler.transform(X2_test)

                # Parse MLP hidden layers
                hidden = tuple(int(x.strip()) for x in mlp_hidden.split(','))

                # Determine if early stopping is feasible
                unique_train = len(set(y1_train))
                use_early_stop = len(X1_train) >= 10 and unique_train >= 2

                models = {
                    'Decision Tree': DecisionTreeClassifier(
                        max_depth=dt_max_depth, min_samples_split=dt_min_samples,
                        random_state=random_state
                    ),
                    'SVM': SVC(
                        C=svm_C, kernel=svm_kernel, probability=True,
                        random_state=random_state, max_iter=10000
                    ),
                    'Neural Network (MLP)': MLPClassifier(
                        hidden_layer_sizes=hidden, max_iter=mlp_max_iter,
                        alpha=mlp_alpha, random_state=random_state,
                        early_stopping=use_early_stop,
                        validation_fraction=0.15 if use_early_stop else 0.0
                    )
                }

                results = {}
                trained_models = {}

                for name, model in models.items():
                    st.write(f"Training **{name}**...")
                    t0 = time.time()

                    use_scaled = (name != 'Decision Tree')
                    Xtr = X1_train_s if use_scaled else X1_train
                    Xte1 = X1_test_s if use_scaled else X1_test
                    Xte2 = X2_test_s if use_scaled else X2_test

                    model.fit(Xtr, y1_train)
                    train_pred = model.predict(Xtr)
                    ds1_pred = model.predict(Xte1)
                    ds2_pred = model.predict(Xte2)
                    ds1_proba = model.predict_proba(Xte1)
                    ds2_proba = model.predict_proba(Xte2)
                    elapsed = time.time() - t0

                    results[name] = {
                        'train_acc': accuracy_score(y1_train, train_pred),
                        'ds1_acc': accuracy_score(y1_test, ds1_pred),
                        'ds1_prec': precision_score(y1_test, ds1_pred, average='weighted', zero_division=0),
                        'ds1_rec': recall_score(y1_test, ds1_pred, average='weighted', zero_division=0),
                        'ds1_f1': f1_score(y1_test, ds1_pred, average='weighted', zero_division=0),
                        'ds2_acc': accuracy_score(y2_test, ds2_pred),
                        'ds2_prec': precision_score(y2_test, ds2_pred, average='weighted', zero_division=0),
                        'ds2_rec': recall_score(y2_test, ds2_pred, average='weighted', zero_division=0),
                        'ds2_f1': f1_score(y2_test, ds2_pred, average='weighted', zero_division=0),
                        'ds1_cm': confusion_matrix(y1_test, ds1_pred),
                        'ds2_cm': confusion_matrix(y2_test, ds2_pred),
                        'ds1_proba': ds1_proba,
                        'ds2_proba': ds2_proba,
                        'train_time': elapsed,
                    }
                    trained_models[name] = model

                st.session_state['models'] = trained_models
                st.session_state['results'] = results
                st.session_state['le_target'] = le_target
                st.session_state['scaler'] = scaler
                st.session_state['feat_names'] = feat_names
                st.session_state['X1_train'] = X1_train
                st.session_state['X1_test'] = X1_test
                st.session_state['X1_train_s'] = X1_train_s
                st.session_state['X1_test_s'] = X1_test_s
                st.session_state['X2_train'] = X2_train
                st.session_state['X2_test'] = X2_test
                st.session_state['X2_train_s'] = X2_train_s
                st.session_state['X2_test_s'] = X2_test_s
                st.session_state['y1_train'] = y1_train
                st.session_state['y1_test'] = y1_test
                st.session_state['y2_train'] = y2_train
                st.session_state['y2_test'] = y2_test

        results = st.session_state['results']
        st.success("All models trained successfully!")

        # Summary table
        summary = pd.DataFrame({
            name: {
                'Train Accuracy': f"{r['train_acc']:.3f}",
                'DS1 Test Accuracy': f"{r['ds1_acc']:.3f}",
                'DS1 Precision': f"{r['ds1_prec']:.3f}",
                'DS1 Recall': f"{r['ds1_rec']:.3f}",
                'DS1 F1': f"{r['ds1_f1']:.3f}",
                'DS2 Test Accuracy': f"{r['ds2_acc']:.3f}",
                'DS2 Precision': f"{r['ds2_prec']:.3f}",
                'DS2 Recall': f"{r['ds2_rec']:.3f}",
                'DS2 F1': f"{r['ds2_f1']:.3f}",
                'Training Time (s)': f"{r['train_time']:.2f}",
            } for name, r in results.items()
        })
        st.dataframe(summary, width="stretch")
    else:
        st.info("Click **Run Full Pipeline** in the sidebar to train models.")

with tabs[3]:
    st.header("Model Evaluation & Cross-Dataset Comparison")

    if 'results' in st.session_state:
        results = st.session_state['results']
        le_target = st.session_state['le_target']

        # Bar chart comparison
        st.subheader("Performance Comparison")
        metrics_df = []
        for name, r in results.items():
            metrics_df.append({'Model': name, 'Metric': 'DS1 Accuracy', 'Value': r['ds1_acc']})
            metrics_df.append({'Model': name, 'Metric': 'DS2 Accuracy', 'Value': r['ds2_acc']})
            metrics_df.append({'Model': name, 'Metric': 'DS1 F1', 'Value': r['ds1_f1']})
            metrics_df.append({'Model': name, 'Metric': 'DS2 F1', 'Value': r['ds2_f1']})
        metrics_df = pd.DataFrame(metrics_df)
        fig_bar = px.bar(metrics_df, x='Model', y='Value', color='Metric', barmode='group',
                         title="Model Performance: Dataset 1 vs Dataset 2",
                         color_discrete_sequence=['#667eea', '#f093fb', '#43e97b', '#fa709a'])
        fig_bar.update_layout(height=450)
        st.plotly_chart(fig_bar, width="stretch")

        # Train vs Test accuracy (overfitting analysis)
        st.subheader("Overfitting Analysis (Train vs Test)")
        overfit_data = []
        for name, r in results.items():
            overfit_data.append({'Model': name, 'Set': 'Train', 'Accuracy': r['train_acc']})
            overfit_data.append({'Model': name, 'Set': 'DS1 Test', 'Accuracy': r['ds1_acc']})
            overfit_data.append({'Model': name, 'Set': 'DS2 Test', 'Accuracy': r['ds2_acc']})
        overfit_df = pd.DataFrame(overfit_data)
        fig_over = px.bar(overfit_df, x='Model', y='Accuracy', color='Set', barmode='group',
                          title="Train vs Test Accuracy - Bias-Variance Analysis",
                          color_discrete_sequence=['#667eea', '#43e97b', '#fa709a'])
        st.plotly_chart(fig_over, width="stretch")

        st.markdown("""
        **Bias-Variance Trade-off Analysis:**
        - **Decision Tree**: High train accuracy but lower test → **overfitting** (high variance, low bias)
        - **SVM**: Moderate gap → balanced bias-variance depending on C and kernel
        - **MLP**: Gap depends on architecture complexity and regularization (alpha)
        - **DS2 performance drop** indicates **temporal data drift**
        """)

        # Confusion Matrices
        st.subheader("Confusion Matrices")
        sel_model = st.selectbox("Select Model", list(results.keys()))
        c1, c2 = st.columns(2)
        with c1:
            fig_cm1 = px.imshow(results[sel_model]['ds1_cm'], text_auto=True,
                                title=f"{sel_model} DS1 Test", color_continuous_scale='Blues')
            st.plotly_chart(fig_cm1, width="stretch")
        with c2:
            fig_cm2 = px.imshow(results[sel_model]['ds2_cm'], text_auto=True,
                                title=f"{sel_model} DS2 Test", color_continuous_scale='Purples')
            st.plotly_chart(fig_cm2, width="stretch")

        # ROC Curves
        st.subheader("ROC Curves (One-vs-Rest)")
        y1_test = st.session_state['y1_test']
        n_classes = len(le_target.classes_)

        fig_roc = make_subplots(rows=1, cols=len(results),
                                subplot_titles=list(results.keys()))
        colors = px.colors.qualitative.Set2
        for idx, (name, r) in enumerate(results.items()):
            proba = r['ds1_proba']
            for i in range(min(n_classes, 5)):
                if i < proba.shape[1]:
                    y_bin = (y1_test == i).astype(int)
                    if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
                        auc_val = auc(fpr, tpr)
                        fig_roc.add_trace(
                            go.Scatter(x=fpr, y=tpr, name=f"Class {i} (AUC={auc_val:.2f})",
                                       line=dict(color=colors[i % len(colors)]),
                                       showlegend=(idx == 0)),
                            row=1, col=idx + 1
                        )
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'),
                                         showlegend=False), row=1, col=idx + 1)
        fig_roc.update_layout(height=450, title_text="ROC Curves (DS1 Test Set)")
        st.plotly_chart(fig_roc, width="stretch")
    else:
        st.info("Train models first.")

with tabs[4]:
    st.header("Continual Learning Implementation")

    if 'models' in st.session_state:
        st.markdown("""
        **Strategy:** Fine-tune models trained on Dataset 1 using Dataset 2's training data.
        - **Decision Tree**: Retrain with combined DS1+DS2 train data (trees can't be incrementally updated)
        - **SVM**: Retrain on combined data using same hyperparameters
        - **MLP**: Use `warm_start=True` to continue training on DS2 data (true continual learning)
        """)

        if st.button("Run Continual Learning", type="primary"):
            with st.spinner("Performing continual learning..."):
                scaler = st.session_state['scaler']
                X1_train = st.session_state['X1_train']
                y1_train = st.session_state['y1_train']
                X2_train = st.session_state['X2_train']
                X2_test = st.session_state['X2_test']
                X2_train_s = st.session_state['X2_train_s']
                X2_test_s = st.session_state['X2_test_s']
                y2_train = st.session_state['y2_train']
                y2_test = st.session_state['y2_test']

                X_combined = np.vstack([X1_train, X2_train])
                y_combined = np.concatenate([y1_train, y2_train])
                X_combined_s = scaler.transform(X_combined)

                cl_results = {}

                # Decision Tree – retrain on combined data
                dt_cl = DecisionTreeClassifier(
                    max_depth=dt_max_depth, min_samples_split=dt_min_samples,
                    random_state=random_state
                )
                dt_cl.fit(X_combined, y_combined)
                dt_pred = dt_cl.predict(X2_test)
                cl_results['Decision Tree (CL)'] = {
                    'acc': accuracy_score(y2_test, dt_pred),
                    'f1': f1_score(y2_test, dt_pred, average='weighted', zero_division=0),
                    'prec': precision_score(y2_test, dt_pred, average='weighted', zero_division=0),
                    'rec': recall_score(y2_test, dt_pred, average='weighted', zero_division=0),
                }

                # SVM – retrain on combined data
                svm_cl = SVC(C=svm_C, kernel=svm_kernel, probability=True,
                             random_state=random_state, max_iter=10000)
                svm_cl.fit(X_combined_s, y_combined)
                svm_pred = svm_cl.predict(X2_test_s)
                cl_results['SVM (CL)'] = {
                    'acc': accuracy_score(y2_test, svm_pred),
                    'f1': f1_score(y2_test, svm_pred, average='weighted', zero_division=0),
                    'prec': precision_score(y2_test, svm_pred, average='weighted', zero_division=0),
                    'rec': recall_score(y2_test, svm_pred, average='weighted', zero_division=0),
                }

                # MLP – warm start continual learning
                # warm_start requires the same classes in each fit() call,
                # so we first fit on DS1, then continue on the combined data
                hidden = tuple(int(x.strip()) for x in mlp_hidden.split(','))
                mlp_cl = MLPClassifier(
                    hidden_layer_sizes=hidden, max_iter=mlp_max_iter,
                    alpha=mlp_alpha, random_state=random_state, warm_start=True
                )
                X1_train_s = st.session_state['X1_train_s']
                mlp_cl.fit(X1_train_s, y1_train)  # Train on DS1
                mlp_cl.fit(X_combined_s, y_combined)  # Continue on combined data
                mlp_pred = mlp_cl.predict(X2_test_s)
                cl_results['Neural Network (CL)'] = {
                    'acc': accuracy_score(y2_test, mlp_pred),
                    'f1': f1_score(y2_test, mlp_pred, average='weighted', zero_division=0),
                    'prec': precision_score(y2_test, mlp_pred, average='weighted', zero_division=0),
                    'rec': recall_score(y2_test, mlp_pred, average='weighted', zero_division=0),
                }

                st.session_state['cl_results'] = cl_results

        if 'cl_results' in st.session_state:
            cl_results = st.session_state['cl_results']
            results = st.session_state['results']
            st.success("Continual learning complete!")

            # Comparison table
            compare_data = []
            model_names = ['Decision Tree', 'SVM', 'Neural Network (MLP)']
            cl_names = ['Decision Tree (CL)', 'SVM (CL)', 'Neural Network (CL)']
            for orig, cl in zip(model_names, cl_names):
                compare_data.append({
                    'Model': orig,
                    'DS2 Acc (Original)': f"{results[orig]['ds2_acc']:.3f}",
                    'DS2 Acc (Continual)': f"{cl_results[cl]['acc']:.3f}",
                    'DS2 F1 (Original)': f"{results[orig]['ds2_f1']:.3f}",
                    'DS2 F1 (Continual)': f"{cl_results[cl]['f1']:.3f}",
                    'Improvement (Acc)': f"{cl_results[cl]['acc'] - results[orig]['ds2_acc']:+.3f}",
                })
            st.dataframe(pd.DataFrame(compare_data), width="stretch")

            # Visual comparison
            fig_cl = go.Figure()
            for orig, cl in zip(model_names, cl_names):
                fig_cl.add_trace(go.Bar(name=f"{orig} (Before)", x=[orig],
                                        y=[results[orig]['ds2_acc']], marker_color='#fa709a'))
                fig_cl.add_trace(go.Bar(name=f"{orig} (After CL)", x=[orig],
                                        y=[cl_results[cl]['acc']], marker_color='#43e97b'))
            fig_cl.update_layout(barmode='group',
                                  title="Continual Learning: DS2 Accuracy Before vs After",
                                  height=450)
            st.plotly_chart(fig_cl, width="stretch")

            st.markdown("""
            **Continual Learning Analysis:**
            - Incorporating historical data (DS1) alongside current data (DS2) should improve generalization
            - MLP with warm_start is a true continual learning approach it preserves learned weights from DS1
            - Decision Tree and SVM are retrained from scratch on combined data (pseudo-continual learning)
            - Improvement indicates successful knowledge transfer across temporal distributions
            """)
    else:
        st.info("Train models first.")

with tabs[5]:
    st.header("Model Interpretation & Feature Importance")

    if 'models' in st.session_state:
        models = st.session_state['models']
        feat_names = st.session_state['feat_names']

        # Decision Tree feature importance
        st.subheader("Decision Tree Feature Importance")
        dt_model = models['Decision Tree']
        dt_imp = pd.DataFrame({
            'Feature': feat_names,
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)

        fig_imp = px.bar(dt_imp, x='Importance', y='Feature', orientation='h',
                         title="Decision Tree Feature Importance (Top 15)",
                         color='Importance', color_continuous_scale='Viridis')
        fig_imp.update_layout(height=500, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_imp, width="stretch")

        # Decision Tree visualization
        st.subheader("Decision Tree Visualization (limited depth)")
        fig_tree, ax_tree = plt.subplots(figsize=(20, 8))
        plot_tree(dt_model, max_depth=3, feature_names=feat_names,
                  filled=True, rounded=True, fontsize=7, ax=ax_tree)
        st.pyplot(fig_tree)

        # MLP permutation importance
        st.subheader("MLP Permutation Importance")
        if st.button("Compute MLP Permutation Importance"):
            with st.spinner("Computing permutation importance..."):
                X1_test_s = st.session_state['X1_test_s']
                y1_test = st.session_state['y1_test']
                perm_imp = permutation_importance(
                    models['Neural Network (MLP)'], X1_test_s, y1_test,
                    n_repeats=10, random_state=random_state, scoring='accuracy'
                )
                perm_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Importance': perm_imp.importances_mean
                }).sort_values('Importance', ascending=False).head(15)

                fig_perm = px.bar(perm_df, x='Importance', y='Feature', orientation='h',
                                 title="MLP Permutation Importance (Top 15)",
                                 color='Importance', color_continuous_scale='Magma')
                fig_perm.update_layout(height=500, yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig_perm, width="stretch")

        # Model complexity discussion
        st.subheader("Model Complexity & Generalization Analysis")
        st.markdown("""
        ### Bias-Variance Trade-off

        | Model | Complexity | Bias | Variance | Comments |
        |-------|-----------|------|----------|----------|
        | **Decision Tree** | High (controlled by max_depth) | Low | High | Prone to overfitting; pruning reduces variance |
        | **SVM** | Medium (controlled by C, kernel) | Medium | Medium | RBF kernel captures non-linear boundaries |
        | **MLP** | High (controlled by layers, alpha) | Low | High | Regularization and early stopping help |

        ### Impact of Temporal Shift
        - Models trained on historical data may underperform on current data due to **data drift**
        - Feature distributions and condition prevalence change over time
        - Continual learning helps adapt to new distributions while retaining historical knowledge

        ### Feature Representation Effects
        - Aggregating observations (mean, std) captures central tendency and variability
        - Standardization is critical for SVM and MLP performance
        - Encounter-based features capture healthcare utilization patterns
        - Different feature representations may benefit different models differently
        """)
    else:
        st.info("Train models first.")

st.markdown("---")
st.markdown("Team 14 ML Assignment-2")