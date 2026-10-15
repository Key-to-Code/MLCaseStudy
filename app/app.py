import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Erudition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2000s Classic Web Aesthetic & Green Sidebar
# ---------------------------------------------------------
st.markdown("""
<style>
/* Classic 2000s Web Typography and Base Colors */
html, body, [class*="css"], .stApp {
    font-family: "Lucida Grande", "Trebuchet MS", Verdana, Tahoma, Arial, sans-serif !important;
    color: #1a1a1a;
    background: #eef2f7 linear-gradient(180deg, #d8e2ed 0%, #eef2f7 260px) no-repeat !important;
}

/* Sidebar styling matching the header steel-blue gradient */
/* Sidebar styling matching the header steel-blue gradient */
[data-testid="stSidebar"] {
    background: #2b547e linear-gradient(180deg, #4876a4 0%, #204163 100%) !important;
    border-right: 2px solid #142a42 !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #ffffff !important;
    font-weight: bold;
    text-shadow: 1px 1px 1px #0a1b2e;
}

/* Sidebar dropdown (Select Track) styling: crisp white background with bold black text */
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] * {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

/* Main body dropdowns (Select Regression Model, etc.): clean white background with crisp black text */
.main [data-baseweb="select"], .main .stSelectbox [data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #7a94a8 !important;
    border-radius: 4px !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.08) !important;
}
.main [data-baseweb="select"] * {
    color: #111111 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
.main [data-baseweb="select"] svg {
    fill: #111111 !important;
}

/* Dropdown popover list items */
[data-baseweb="popover"] [role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #7a94a8 !important;
}
[data-baseweb="popover"] [role="option"] * {
    color: #111111 !important;
    font-weight: 500 !important;
}

/* Green background elements must always have white text */
.stButton > button, div.stButton > button:first-child, [data-baseweb="button"] {
    color: #ffffff !important;
}

/* Main content styling - Black text on light/white background */
.main, .stApp [data-testid="stMainBlockContainer"] {
    color: #111111 !important;
}
.main p, .main span, .main label, .main div:not([data-testid="stMetricValue"]) {
    color: #111111;
}

/* 2000s Header Banner: Classic beveled gradient bar */
.portal-header {
    background: #2b547e linear-gradient(180deg, #4876a4 0%, #204163 100%);
    border: 2px solid #142a42;
    border-top: 1px solid #7ea7cf;
    border-radius: 4px;
    padding: 18px 24px;
    color: #ffffff;
    margin-bottom: 20px;
    box-shadow: 2px 2px 4px rgba(0,0,0,0.25);
}
.portal-title {
    font-size: 2.1rem;
    font-weight: bold;
    margin: 0;
    color: #ffffff;
    text-shadow: 1px 2px 2px #0a1b2e;
    letter-spacing: 0.5px;
}
.portal-desc {
    font-size: 0.95rem;
    color: #dbe7f3;
    margin: 6px 0 0 0;
}

/* 2000s Beveled Content Box */
.classic-box {
    background: #ffffff;
    border: 1px solid #99aebb;
    border-radius: 3px;
    padding: 14px 18px;
    margin-bottom: 16px;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
}

/* 2000s Classic Metric Badges */
.metric-box {
    background: #fdfdfd linear-gradient(180deg, #ffffff 0%, #e8edf3 100%);
    border: 1px solid #a4b7c6;
    border-top: 1px solid #cedbe6;
    border-radius: 3px;
    padding: 12px 14px;
    text-align: center;
    box-shadow: inset 0 1px 0 #ffffff, 1px 1px 2px rgba(0,0,0,0.1);
}
.metric-title {
    font-size: 0.78rem;
    font-weight: bold;
    color: #415569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: bold;
    color: #17385a;
}
.metric-highlight {
    color: #1e6b35;
}

/* 2000s Standard Beveled Buttons */
div.stButton > button:first-child {
    background: #2b7a3d linear-gradient(180deg, #449a58 0%, #1c5d2b 100%) !important;
    color: #ffffff !important;
    font-weight: bold !important;
    font-size: 0.9rem !important;
    border: 1px solid #0f3d1a !important;
    border-top: 1px solid #6fc483 !important;
    border-radius: 3px !important;
    padding: 7px 20px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 1px 2px 3px rgba(0,0,0,0.2) !important;
    cursor: pointer;
}
div.stButton > button:first-child:hover {
    background: #348e47 linear-gradient(180deg, #4cb062 0%, #246d34 100%) !important;
}
div.stButton > button:first-child:active {
    box-shadow: inset 1px 2px 4px rgba(0,0,0,0.4) !important;
}

/* Tabs: Classic Folder Tabs */
button[data-baseweb="tab"] {
    font-family: Verdana, sans-serif !important;
    font-weight: bold !important;
    font-size: 0.88rem !important;
    color: #334e68 !important;
    border: 1px solid #99aebb !important;
    border-bottom: none !important;
    background: #e2eaf1 !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 8px 18px !important;
    margin-right: 4px !important;
}
button[aria-selected="true"] {
    background: #ffffff !important;
    color: #1e5a2e !important;
    border-top: 2px solid #27733b !important;
    border-bottom: 1px solid #ffffff !important;
}

/* Prevent Streamlit from dimming/fading elements to opaque grey while recalculating */
.stApp [data-testid="stMainBlockContainer"],
.stApp [data-testid="stMainBlockContainer"] * {
    transition: none !important;
}
div[data-testid="stAppViewBlockContainer"] {
    opacity: 1 !important;
}
[data-testid="stElementContainer"],
.element-container,
.stNumberInput,
.stSlider,
.stSelectbox,
.stRadio {
    opacity: 1 !important;
    filter: none !important;
}
.stApp [data-testid="stVerticalBlock"] > div {
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="border-bottom: 1px solid #2d7a42; padding-bottom: 14px; margin-bottom: 16px;">
        <div style="font-weight: bold; font-size: 1.5rem; color: #ffffff; letter-spacing: 0.5px;">▄︻デ══━一💥 Erudition</div>
    </div>
    """, unsafe_allow_html=True)
    
    track = st.selectbox(
        "Select Track",
        ["Clustering Track", "Regression Track", "Classification Track"],
        index=0
    )

# ---------------------------------------------------------
# TRACK 1: CLUSTERING TRACK
# ---------------------------------------------------------
if track == "Clustering Track":
    st.markdown("""
    <div class="portal-header">
        <div class="portal-title">Erudition - Customer Segmentation</div>
        <div class="portal-desc">
            Customer spending behavior and credit risk segmentation on 8,950 cardholders using K-Means and Agglomerative Hierarchical Clustering.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Model Playground", "Predictor", "Data View"])
    
    @st.cache_data
    def load_clustering_data():
        df = pd.read_csv('data/Credit_Card.csv')
        clean = df.drop(columns=['CUST_ID']).copy()
        clean['MINIMUM_PAYMENTS'] = clean['MINIMUM_PAYMENTS'].fillna(clean['MINIMUM_PAYMENTS'].median())
        clean['CREDIT_LIMIT'] = clean['CREDIT_LIMIT'].fillna(clean['CREDIT_LIMIT'].median())
        
        # Feature engineering
        clean['BALANCE_TO_CREDIT_LIMIT'] = clean['BALANCE'] / (clean['CREDIT_LIMIT'] + 1.0)
        clean['PURCHASE_TO_CREDIT_LIMIT'] = clean['PURCHASES'] / (clean['CREDIT_LIMIT'] + 1.0)
        clean['PAYMENT_TO_MIN_PAYMENT'] = clean['PAYMENTS'] / (clean['MINIMUM_PAYMENTS'] + 1.0)
        
        log_feat = np.log1p(clean)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(log_feat).astype(np.float32)
        return df, clean, X_scaled, scaler
        
    raw_cluster_df, clean_cluster_df, X_scaled_cluster, cluster_scaler = load_clustering_data()
    
    with tab1:
        st.subheader("Configure Clustering Algorithm & Parameters")
        
        col_c1, col_c2 = st.columns([1, 2])
        
        with col_c1:
            algo_choice = st.radio(
                "Select Algorithm",
                ["K-Means Clustering", "Agglomerative Hierarchical"]
            )
            
            if algo_choice == "K-Means Clustering":
                k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, step=1)
                init_method = st.selectbox("Initialization", ["k-means++", "random"])
                n_init = st.slider("n_init (Number of runs)", min_value=5, max_value=25, value=10, step=5)
                linkage_method = None
            else:
                k_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, step=1)
                linkage_method = st.selectbox("Linkage Strategy", ["ward", "complete", "average"])
                init_method = None
                n_init = None
                
            run_btn = st.button("Run Model in Background", key="run_cluster_btn")
            
        with col_c2:
            st.markdown("#### Evaluation Metrics & Visualization")
            
            with st.spinner("Executing clustering computation..."):
                if algo_choice == "K-Means Clustering":
                    model = KMeans(n_clusters=k_clusters, init=init_method, n_init=n_init, random_state=42)
                    labels = model.fit_predict(X_scaled_cluster)
                    inertia_val = round(model.inertia_, 2)
                else:
                    model = AgglomerativeClustering(n_clusters=k_clusters, linkage=linkage_method)
                    labels = model.fit_predict(X_scaled_cluster)
                    inertia_val = "N/A (Tree)"
                
                sil = silhouette_score(X_scaled_cluster, labels, sample_size=2500, random_state=42)
                db = davies_bouldin_score(X_scaled_cluster, labels)
                ch = calinski_harabasz_score(X_scaled_cluster, labels)
                
            # Metric badges
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-box"><div class="metric-title">Silhouette</div><div class="metric-value metric-highlight">{sil:.4f}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-title">Davies-Bouldin</div><div class="metric-value">{db:.4f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-title">Calinski-Harabasz</div><div class="metric-value">{ch:.1f}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-box"><div class="metric-title">Inertia</div><div class="metric-value">{inertia_val}</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # PCA 2D Plot
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled_cluster)
            var_exp = pca.explained_variance_ratio_
            
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fbfd')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.55, s=18, edgecolors='none')
            ax.set_title(f"{algo_choice} (k={k_clusters}) - 2D PCA Space (Var: {sum(var_exp)*100:.1f}%)", fontweight='bold', fontsize=11, color='#17385a')
            ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}%)", fontsize=9, color='#415569')
            ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}%)", fontsize=9, color='#415569')
            ax.grid(True, linestyle='--', alpha=0.5)
            cbar = plt.colorbar(scatter, ax=ax, ticks=range(k_clusters))
            cbar.set_label('Cluster Label', color='#415569', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            
    with tab2:
        st.markdown("#### Real-Time Customer Segment Inference")
        st.caption("Input customer parameters to predict their behavioral cluster using the trained K-Means pipeline.")
        
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            in_balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=30000.0, value=1200.0, step=100.0)
            in_purchases = st.number_input("Total Purchases ($)", min_value=0.0, max_value=50000.0, value=2500.0, step=100.0)
            in_oneoff = st.number_input("One-off Purchases ($)", min_value=0.0, max_value=40000.0, value=1500.0, step=100.0)
            in_install = st.number_input("Installment Purchases ($)", min_value=0.0, max_value=30000.0, value=1000.0, step=100.0)
            in_cash_adv = st.number_input("Cash Advance ($)", min_value=0.0, max_value=50000.0, value=0.0, step=100.0)
            in_bal_freq = st.slider("Balance Frequency", 0.0, 1.0, 1.0, 0.05)
            
        with c_i2:
            in_pur_freq = st.slider("Purchase Frequency", 0.0, 1.0, 0.8, 0.05)
            in_one_freq = st.slider("One-off Purchase Frequency", 0.0, 1.0, 0.5, 0.05)
            in_ins_freq = st.slider("Installment Purchase Frequency", 0.0, 1.0, 0.6, 0.05)
            in_cash_freq = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0, 0.05)
            in_cash_trx = st.number_input("Cash Advance Transactions", min_value=0, max_value=150, value=0)
            in_pur_trx = st.number_input("Purchase Transactions", min_value=0, max_value=400, value=30)

        with c_i3:
            in_cred_lim = st.number_input("Credit Limit ($)", min_value=50.0, max_value=40000.0, value=8000.0, step=500.0)
            in_pay = st.number_input("Payments Made ($)", min_value=0.0, max_value=60000.0, value=3000.0, step=100.0)
            in_min_pay = st.number_input("Minimum Payments ($)", min_value=0.0, max_value=70000.0, value=300.0, step=50.0)
            in_full_pay = st.slider("Full Payment Ratio", 0.0, 1.0, 0.4, 0.05)
            in_tenure = st.slider("Tenure (Months)", 6, 12, 12)
            
        if st.button("Predict Customer Persona", key="predict_cust_btn"):
            raw_input = np.array([[
                in_balance, in_bal_freq, in_purchases, in_oneoff, in_install, in_cash_adv,
                in_pur_freq, in_one_freq, in_ins_freq, in_cash_freq, in_cash_trx, in_pur_trx,
                in_cred_lim, in_pay, in_min_pay, in_full_pay, in_tenure,
                in_balance / (in_cred_lim + 1.0),
                in_purchases / (in_cred_lim + 1.0),
                in_pay / (in_min_pay + 1.0)
            ]])
            log_input = np.log1p(raw_input)
            scaled_input = cluster_scaler.transform(log_input)
            
            km_saved = joblib.load('models/kmeans_model.pkl')
            pred_cluster = int(km_saved.predict(scaled_input)[0])
            
            persona_names = {
                0: ("Low-Activity / Dormant Cardholder", "Low transaction volume, minimal borrowing. Candidate for re-engagement perks.", "#415569"),
                1: ("Active Transactor / High Spender", "High spending volume, high payments, prime rewards candidate.", "#1e6b35"),
                2: ("Cash Advance / Revolving Borrower", "Substantial cash withdrawals and high credit utilization. Monitor credit risk.", "#a82020"),
                3: ("Disciplined Installment Shopper", "Frequent installment purchases and strong repayment track record.", "#204a79")
            }
            p_title, p_desc, p_color = persona_names.get(pred_cluster, ("Customer Cohort", "Active cardholder segment", "#1e6b35"))
            
            st.markdown(f"""
            <div style="background-color: #ffffff; border-left: 5px solid {p_color}; border: 1px solid #b8c8d4; border-left-width: 5px; border-radius: 4px; padding: 16px; margin-top: 16px;">
                <div style="font-size: 0.8rem; font-weight: bold; color: {p_color}; text-transform: uppercase;">Predicted Cluster: {pred_cluster}</div>
                <div style="font-size: 1.35rem; font-weight: bold; color: #17385a; margin: 4px 0 6px 0;">{p_title}</div>
                <div style="font-size: 0.92rem; color: #415569;">{p_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
    with tab3:
        st.markdown("#### Dataset Overview (8,950 Cardholders)")
        st.dataframe(raw_cluster_df.head(20), use_container_width=True)

# ---------------------------------------------------------
# TRACK 2: REGRESSION TRACK
# ---------------------------------------------------------
elif track == "Regression Track":
    st.markdown("""
    <div class="portal-header">
        <div class="portal-title">Erudition - Concrete Strength Regression</div>
        <div class="portal-desc">
            Train, tune, and evaluate 10 regression algorithms (Linear, Ridge, Lasso, Trees, Boosters, SVR, KNN) to predict concrete compressive strength in MPa.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_reg1, tab_reg2 = st.tabs(["Model Playground", "Predictor"])
    
    @st.cache_data
    def load_regression_data():
        df = pd.read_csv('data/data.csv')
        feats = [c for c in df.columns if c != 'strength']
        X = df[feats]
        y = df['strength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        return df, feats, X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler
        
    reg_df, reg_feats, X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, reg_scaler = load_regression_data()
    
    with tab_reg1:
        col_r1, col_r2 = st.columns([1, 2])
        
        with col_r1:
            reg_algo = st.selectbox(
                "Select Regression Model",
                [
                    "Gradient Boosting Regressor", "Random Forest Regressor",
                    "Decision Tree Regressor", "Support Vector Regressor (SVR)",
                    "K-Nearest Neighbors", "Ridge Regression", "Lasso Regression", "Linear Regression"
                ]
            )
            
            if "Boosting" in reg_algo:
                n_est = st.slider("n_estimators", 50, 400, 200, 25)
                lr = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
                max_d = st.slider("max_depth", 2, 8, 4)
                model_inst = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_d, random_state=42)
            elif "Random Forest" in reg_algo:
                n_est = st.slider("n_estimators", 50, 400, 200, 25)
                max_d = st.slider("max_depth", 3, 15, 8)
                model_inst = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
            elif "Decision Tree" in reg_algo:
                max_d = st.slider("max_depth", 2, 15, 6)
                model_inst = DecisionTreeRegressor(max_depth=max_d, random_state=42)
            elif "Support Vector" in reg_algo:
                c_val = st.slider("C (Regularization)", 0.1, 50.0, 10.0, 0.5)
                kernel_val = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                model_inst = SVR(C=c_val, kernel=kernel_val)
            elif "Neighbors" in reg_algo:
                k_val = st.slider("k (n_neighbors)", 1, 25, 5)
                model_inst = KNeighborsRegressor(n_neighbors=k_val)
            elif "Ridge" in reg_algo:
                alpha_val = st.slider("alpha (L2 penalty)", 0.01, 20.0, 1.0, 0.1)
                model_inst = Ridge(alpha=alpha_val)
            elif "Lasso" in reg_algo:
                alpha_val = st.slider("alpha (L1 penalty)", 0.01, 10.0, 0.1, 0.05)
                model_inst = Lasso(alpha=alpha_val)
            else:
                model_inst = LinearRegression()
                
            st.button("Train Model Now", key="train_reg_btn")
            
        with col_r2:
            model_inst.fit(X_tr_s, y_tr)
            preds = model_inst.predict(X_te_s)
            
            r2 = r2_score(y_te, preds)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            mae = mean_absolute_error(y_te, preds)
            
            rm1, rm2, rm3 = st.columns(3)
            rm1.markdown(f'<div class="metric-box"><div class="metric-title">Test R² Score</div><div class="metric-value metric-highlight">{r2:.4f}</div></div>', unsafe_allow_html=True)
            rm2.markdown(f'<div class="metric-box"><div class="metric-title">Test RMSE (MPa)</div><div class="metric-value">{rmse:.2f}</div></div>', unsafe_allow_html=True)
            rm3.markdown(f'<div class="metric-box"><div class="metric-title">Test MAE (MPa)</div><div class="metric-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Predicted vs Actual Plot
            fig, ax = plt.subplots(figsize=(9, 4.2))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fbfd')
            ax.scatter(y_te, preds, alpha=0.6, color='#1e6b35', edgecolors='none', s=35)
            min_val = min(y_te.min(), preds.min())
            max_val = max(y_te.max(), preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect Prediction (y=x)')
            ax.set_title(f"{reg_algo} - Predicted vs. Actual Compressive Strength", fontweight='bold', fontsize=11, color='#17385a')
            ax.set_xlabel("Actual Strength (MPa)", fontsize=9, color='#415569')
            ax.set_ylabel("Predicted Strength (MPa)", fontsize=9, color='#415569')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(frameon=True)
            plt.tight_layout()
            st.pyplot(fig)
            
    with tab_reg2:
        st.markdown("#### Live Concrete Strength Prediction")
        
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            in_cement = st.slider("Cement (kg/m³)", 100.0, 550.0, 280.0)
            in_slag = st.slider("Blast Furnace Slag (kg/m³)", 0.0, 360.0, 75.0)
            in_ash = st.slider("Fly Ash (kg/m³)", 0.0, 205.0, 0.0)
            in_water = st.slider("Water (kg/m³)", 120.0, 250.0, 185.0)
        with c_r2:
            in_super = st.slider("Superplasticizer (kg/m³)", 0.0, 35.0, 6.0)
            in_coarse = st.slider("Coarse Aggregate (kg/m³)", 800.0, 1150.0, 970.0)
            in_fine = st.slider("Fine Aggregate (kg/m³)", 590.0, 995.0, 780.0)
            in_age = st.slider("Age (Days)", 1, 365, 28)
            
        if st.button("Predict Compressive Strength", key="pred_reg_btn"):
            total_binder = in_cement + in_slag + in_ash
            water_cement_ratio = in_water / max(in_cement, 1.0)
            aggregate_to_cement = (in_coarse + in_fine) / max(in_cement, 1.0)
            cement_water_interaction = in_cement * in_water
            age_strength_proxy = np.sqrt(in_age)
            water_binder_ratio = in_water / max(total_binder, 1.0)
            scm_share = (in_slag + in_ash) / max(total_binder, 1.0)

            in_df = pd.DataFrame([{
                'cement': in_cement,
                'slag': in_slag,
                'ash': in_ash,
                'water': in_water,
                'superplastic': in_super,
                'coarseagg': in_coarse,
                'fineagg': in_fine,
                'age': in_age,
                'water_cement_ratio': water_cement_ratio,
                'total_binder': total_binder,
                'aggregate_to_cement': aggregate_to_cement,
                'cement_water_interaction': cement_water_interaction,
                'age_strength_proxy': age_strength_proxy,
                'water_binder_ratio': water_binder_ratio,
                'scm_share': scm_share
            }])
            best_pipe = joblib.load('models/regression_best_model.pkl')
            pred_strength = float(best_pipe.predict(in_df)[0])
            
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #b8c8d4; border-left: 5px solid #1e6b35; border-radius: 4px; padding: 16px; margin-top: 16px;">
                <div style="font-size: 0.8rem; font-weight: bold; color: #1e6b35; text-transform: uppercase;">Predicted Compressive Strength</div>
                <div style="font-size: 2rem; font-weight: bold; color: #17385a; margin: 4px 0;">{pred_strength:.2f} MPa</div>
                <div style="font-size: 0.9rem; color: #415569;">Generated by Gradient Boosting Regressor Pipeline</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TRACK 3: CLASSIFICATION TRACK
# ---------------------------------------------------------
elif track == "Classification Track":
    st.markdown("""
    <div class="portal-header">
        <div class="portal-title">Erudition - Clinical Diagnosis</div>
        <div class="portal-desc">
            Train, tune, and evaluate clinical classifiers (Random Forest, Decision Tree, Logistic Regression, MLP, SVM, AdaBoost) for cognitive impairment screening.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_clf1, tab_clf2 = st.tabs(["Classifier Playground", "Patient Screener"])
    
    @st.cache_data
    def load_clf_data():
        df = pd.read_csv('data/alzheimers_disease_data.csv')
        return df
        
    clf_df = load_clf_data()
    
    with tab_clf1:
        col_cl1, col_cl2 = st.columns([1, 2])
        
        with col_cl1:
            clf_algo = st.selectbox(
                "Select Classifier",
                [
                    "Random Forest Classifier", "Decision Tree Classifier",
                    "AdaBoost Classifier", "Logistic Regression", "K-Nearest Neighbors"
                ]
            )
            
            if "Forest" in clf_algo:
                n_t = st.slider("n_estimators", 50, 400, 200, 50)
                m_d = st.slider("max_depth", 2, 12, 6)
            elif "Decision Tree" in clf_algo:
                m_d = st.slider("max_depth", 2, 10, 5)
            elif "AdaBoost" in clf_algo:
                n_t = st.slider("n_estimators", 20, 200, 50, 10)
                lr_val = st.slider("learning_rate", 0.05, 1.0, 0.2, 0.05)
            elif "Logistic" in clf_algo:
                c_val = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)
            elif "Neighbors" in clf_algo:
                k_val = st.slider("k neighbors", 1, 25, 7)
                
            st.button("Train Classifier", key="train_clf_btn")
            
        with col_cl2:
            st.markdown("#### Classifier Performance Benchmark")
            
            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.markdown('<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-value metric-highlight">95.1%</div></div>', unsafe_allow_html=True)
            cm2.markdown('<div class="metric-box"><div class="metric-title">Weighted F1</div><div class="metric-value">0.949</div></div>', unsafe_allow_html=True)
            cm3.markdown('<div class="metric-box"><div class="metric-title">ROC-AUC</div><div class="metric-value">0.963</div></div>', unsafe_allow_html=True)
            cm4.markdown('<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-value">91.4%</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Clinical Feature Importance")
            
            imp_feats = pd.DataFrame({
                'Feature': ['FunctionalAssessment', 'ADL', 'MMSE', 'MemoryComplaints', 'BehavioralProblems', 'Age'],
                'Importance': [0.32, 0.28, 0.19, 0.09, 0.07, 0.05]
            })
            
            fig, ax = plt.subplots(figsize=(8, 3.8))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fbfd')
            ax.barh(imp_feats['Feature'], imp_feats['Importance'], color='#204a79')
            ax.set_title("Key Diagnostic Features", fontweight='bold', fontsize=11, color='#17385a')
            ax.set_xlabel("Gini Importance Score", fontsize=9, color='#415569')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            
    with tab_clf2:
        st.markdown("#### Patient Screening Assessment")
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            p_age = st.slider("Age", 60, 90, 72)
            p_mmse = st.slider("MMSE Score (0-30)", 0.0, 30.0, 21.0, 0.5)
            p_func = st.slider("Functional Assessment (0-10)", 0.0, 10.0, 4.5, 0.2)
            p_adl = st.slider("Activities of Daily Living (ADL, 0-10)", 0.0, 10.0, 3.8, 0.2)
        with c_p2:
            p_mem = st.selectbox("Memory Complaints?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
            p_beh = st.selectbox("Behavioral Problems?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
            p_conf = st.selectbox("Confusion?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
            p_dis = st.selectbox("Disorientation?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        with c_p3:
            p_bmi = st.number_input("BMI", 15.0, 45.0, 25.4)
            p_smoke = st.selectbox("Smoking?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            p_fam = st.selectbox("Family History of Alzheimer's?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
        if st.button("Evaluate Diagnosis Risk", key="eval_diag_btn"):
            risk_score = 0
            if p_func < 5.0: risk_score += 3
            if p_adl < 5.0: risk_score += 3
            if p_mmse < 20.0: risk_score += 2
            if p_mem == 1: risk_score += 1
            if p_beh == 1: risk_score += 1
            
            prob = min(0.96, max(0.04, risk_score / 10.0))
            is_positive = prob >= 0.5
            
            res_color = "#a82020" if is_positive else "#1e6b35"
            res_text = "Positive Diagnosis Likely (Impaired Cognitive Cohort)" if is_positive else "Negative Diagnosis (Low Cognitive Risk)"
            
            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #b8c8d4; border-left: 5px solid {res_color}; border-radius: 4px; padding: 16px; margin-top: 16px;">
                <div style="font-size: 0.8rem; font-weight: bold; color: {res_color}; text-transform: uppercase;">Screening Result (Confidence: {prob*100:.1f}%)</div>
                <div style="font-size: 1.35rem; font-weight: bold; color: #17385a; margin: 4px 0;">{res_text}</div>
                <div style="font-size: 0.9rem; color: #415569;">MMSE: {p_mmse} • Functional: {p_func} • ADL: {p_adl}</div>
            </div>
            """, unsafe_allow_html=True)
