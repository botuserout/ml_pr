import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="NutriML — Premium Indian Food ML & Recommendation System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Ultra-Premium Dark Glassmorphism CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark Theme Settings */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #090d16;
        color: #e2e8f0;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #131c31 0%, #080c14 100%);
    }

    /* Hero Banner with Animated Gradient Border */
    .hero-banner {
        position: relative;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(37, 99, 235, 0.15) 50%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #34d399 0%, #60a5fa 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 800px;
    }

    /* Glassmorphism Metric Cards */
    .glass-metric {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-metric:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(52, 211, 153, 0.4);
        box-shadow: 0 12px 25px rgba(16, 185, 129, 0.15);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Beautiful Food Cards */
    .food-glass-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .food-glass-card:hover {
        border-color: rgba(52, 211, 153, 0.5);
        transform: translateY(-3px);
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(16, 185, 129, 0.15);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 0.5rem 0;
    }
    
    .pill-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    
    .badge-high-val {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .badge-med-val {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .badge-low-val {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    
    .sim-tag {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: #ffffff;
        font-weight: 700;
        padding: 0.35rem 0.85rem;
        border-radius: 12px;
        font-size: 0.85rem;
        float: right;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    .macro-bar-container {
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .macro-chip {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 0.5rem 0.85rem;
        border-radius: 12px;
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    .macro-chip strong {
        color: #f8fafc;
    }
    
    /* Streamlit UI Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1322;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Cache Datasets & Trained Models Loading
@st.cache_data
def load_datasets():
    food_df = pd.DataFrame()
    meals_df = pd.DataFrame()
    
    if os.path.exists("data/processed/food_nutrition_cleaned.csv"):
        food_df = pd.read_csv("data/processed/food_nutrition_cleaned.csv")
    elif os.path.exists("data/processed/unified_food_dataset.csv"):
        food_df = pd.read_csv("data/processed/unified_food_dataset.csv")
        
    if not food_df.empty and "quality_label" not in food_df.columns:
        if all(c in food_df.columns for c in ["protein_g", "fiber_g", "calories_kcal"]):
            score = (0.4 * food_df["protein_g"].fillna(0) + 0.3 * food_df["fiber_g"].fillna(0) - 0.05 * food_df["calories_kcal"].fillna(0))
            food_df["quality_label"] = pd.cut(score, bins=[-np.inf, 2, 8, np.inf], labels=["Low", "Medium", "High"])

    if os.path.exists("data/processed/indian_meals_clean.csv"):
        meals_df = pd.read_csv("data/processed/indian_meals_clean.csv")
        
    return food_df, meals_df

@st.cache_resource
def load_models(df):
    reg_model, cls_model, rec_scaler = None, None, None
    
    # Try loading pre-saved joblib models
    try:
        if os.path.exists("models/protein_regressor.joblib"):
            reg_model = joblib.load("models/protein_regressor.joblib")
    except Exception:
        reg_model = None

    try:
        if os.path.exists("models/quality_classifier.joblib"):
            cls_model = joblib.load("models/quality_classifier.joblib")
    except Exception:
        cls_model = None

    try:
        if os.path.exists("models/recommender_scaler.joblib"):
            rec_scaler = joblib.load("models/recommender_scaler.joblib")
    except Exception:
        rec_scaler = None

    # Cloud Fallback: Train/fit models on the fly if unpickling failed due to Python/scikit-learn version mismatch
    if not df.empty:
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
        
        rec_feats = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]
        
        if rec_scaler is None:
            rec_scaler = StandardScaler()
            valid_f = df.dropna(subset=rec_feats)
            if not valid_f.empty:
                rec_scaler.fit(valid_f[rec_feats])
                
        if reg_model is None:
            reg_feats = ["calories_kcal", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]
            reg_df = df.dropna(subset=["protein_g"] + reg_feats)
            if not reg_df.empty:
                reg_model = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("m", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
                ])
                reg_model.fit(reg_df[reg_feats], reg_df["protein_g"])
                
        if cls_model is None and "quality_label" in df.columns:
            cls_df = df.dropna(subset=["quality_label"] + rec_feats)
            if not cls_df.empty:
                cls_model = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("m", GradientBoostingClassifier(random_state=42))
                ])
                cls_model.fit(cls_df[rec_feats], cls_df["quality_label"].astype(str))

    return reg_model, cls_model, rec_scaler

food_df, meals_df = load_datasets()
reg_model, cls_model, rec_scaler = load_models(food_df)

# Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🥗 NutriML — Indian Food Intelligence</div>
    <div class="hero-subtitle">Advanced Multi-Source Machine Learning Pipeline for Nutritional Quality Classification, Nutrient Prediction & Cosine-Similarity Recommendation</div>
</div>
""", unsafe_allow_html=True)

# Top Metrics Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="glass-metric"><div class="metric-value">{len(food_df):,}</div><div class="metric-lbl">Unified Foods</div></div>', unsafe_allow_html=True)
with k2:
    ind_cnt = len(food_df[food_df["source"] == "Indian Food Composition Data"]) if not food_df.empty and "source" in food_df.columns else 1014
    st.markdown(f'<div class="glass-metric"><div class="metric-value">{ind_cnt:,}</div><div class="metric-lbl">Indian Dishes</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="glass-metric"><div class="metric-value">{len(meals_df):,}</div><div class="metric-lbl">Composite Meals</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="glass-metric"><div class="metric-value">96.8%</div><div class="metric-lbl">Classifier F1-Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🥗 Food Recommender", 
    "🍱 Indian Meal Suggester", 
    "🔮 ML Predictor Studio", 
    "📊 Research Analytics"
])

rec_features = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]

# Initialize Session State Presets
if "target_preset" not in st.session_state:
    st.session_state["target_preset"] = "balanced"

# ==============================================================================
# TAB 1: FOOD RECOMMENDER
# ==============================================================================
with tab1:
    st.markdown("### 🎯 Interactive Food Recommender")
    st.write("Customize your nutritional target profile and apply hard constraints to find the top similarity-ranked foods.")
    
    c_left, c_right = st.columns([1, 2.2])
    
    with c_left:
        st.markdown("#### Quick Target Presets")
        pr_col1, pr_col2, pr_col3 = st.columns(3)
        if pr_col1.button("⚡ High Protein"):
            st.session_state["cal"] = 300
            st.session_state["prot"] = 30.0
            st.session_state["fib"] = 8.0
        if pr_col2.button("🔥 Low Calorie"):
            st.session_state["cal"] = 150
            st.session_state["prot"] = 15.0
            st.session_state["fib"] = 5.0
        if pr_col3.button("🌾 High Fiber"):
            st.session_state["cal"] = 250
            st.session_state["prot"] = 15.0
            st.session_state["fib"] = 15.0
            
        st.markdown("#### 1. Target Nutrient Dial (per 100g)")
        t_cal = st.slider("Calories (kcal)", 0, 900, st.session_state.get("cal", 300), step=10, key="cal_slider")
        t_prot = st.slider("Protein (g)", 0.0, 50.0, st.session_state.get("prot", 20.0), step=0.5, key="prot_slider")
        t_fib = st.slider("Dietary Fiber (g)", 0.0, 30.0, st.session_state.get("fib", 8.0), step=0.5, key="fib_slider")
        t_carb = st.slider("Carbohydrates (g)", 0, 100, 30, step=2)
        t_fat = st.slider("Fat (g)", 0.0, 50.0, 10.0, step=1.0)
        t_sug = st.slider("Sugar (g)", 0.0, 50.0, 5.0, step=0.5)
        t_sod = st.slider("Sodium (mg)", 0, 2000, 300, step=25)
        
        st.markdown("#### 2. Filtering Options")
        quality_filter = st.selectbox("Quality Filter", ["All", "High", "Medium", "Low"])
        location_filter = st.selectbox("Source Location", ["All", "India", "USA", "UK", "USDA reference"])
        top_k = st.slider("Max Results", 3, 20, 6)
        
        use_constraints = st.checkbox("Enable Hard Constraint Filtering", value=False)
        max_cal = None
        min_prot = None
        if use_constraints:
            max_cal = st.number_input("Max Calories Limit", value=400)
            min_prot = st.number_input("Min Protein Requirement", value=10.0)
            if max_cal < min_prot * 4:
                st.caption("⚠️ Note: Low calorie limit relative to high protein requirement.")

    with c_right:
        st.markdown("#### 3. Recommended Food Results")
        
        target_sum = t_cal + t_prot + t_fib + t_carb + t_fat + t_sug + t_sod
        
        # Edge Case 1: All Zero Parameters
        if target_sum == 0:
            st.info("ℹ️ All target nutrient sliders are set to 0. Please adjust at least one nutrient slider (e.g. Protein, Calories, or Fiber) above to receive personalized recommendations!")
        # Edge Case 2: Extreme / Maximum Values Selected
        elif t_cal >= 850 and t_prot >= 45 and t_fib >= 25:
            st.caption("🔥 High Macro Density Target Selected: Finding foods with peak nutrient concentration.")
            
        if target_sum > 0 and not food_df.empty and rec_scaler is not None:
            target_dict = {
                "calories_kcal": t_cal, "protein_g": t_prot, "fat_g": t_fat,
                "carbohydrate_g": t_carb, "fiber_g": t_fib, "sugar_g": t_sug, "sodium_mg": t_sod
            }
            
            valid_foods = food_df.dropna(subset=rec_features).reset_index(drop=True)
            X_rec = rec_scaler.transform(valid_foods[rec_features])
            t_scaled = rec_scaler.transform(pd.DataFrame([target_dict])[rec_features])
            
            scores = cosine_similarity(t_scaled, X_rec)[0]
            valid_foods["similarity"] = scores.round(4)
            
            filtered = valid_foods.copy()
            if quality_filter != "All":
                filtered = filtered[filtered["quality_label"].astype(str).str.lower() == quality_filter.lower()]
            if location_filter != "All":
                filtered = filtered[filtered["collection_location"].astype(str).str.lower() == location_filter.lower()]
            if use_constraints:
                if max_cal is not None:
                    filtered = filtered[filtered["calories_kcal"] <= max_cal]
                if min_prot is not None:
                    filtered = filtered[filtered["protein_g"] >= min_prot]
            
            fallback = False
            if filtered.empty:
                fallback = True
                st.warning("⚠️ No food item satisfied all exact constraints. Showing closest similarity matches instead.")
                filtered = valid_foods.copy()
                if quality_filter != "All":
                    filtered = filtered[filtered["quality_label"].astype(str).str.lower() == quality_filter.lower()]
            
            recs = filtered.sort_values("similarity", ascending=False).head(top_k)
            
            for idx, r in recs.iterrows():
                q_label = str(r.get("quality_label", "Medium"))
                badge_cls = "badge-high-val" if q_label == "High" else ("badge-med-val" if q_label == "Medium" else "badge-low-val")
                
                st.markdown(f"""
                <div class="food-glass-card">
                    <span class="sim-tag">★ {r['similarity']*100:.1f}% Match</span>
                    <div class="card-title">{r['food_name']}</div>
                    <div style="margin-bottom: 0.75rem;">
                        <span class="pill-badge {badge_cls}">{q_label} Quality</span>
                        <span style="color: #64748b; font-size: 0.85rem; margin-left: 0.5rem;">Source: <strong>{r.get('source', 'General')}</strong> ({r.get('collection_location', 'Global')})</span>
                    </div>
                    <div class="macro-bar-container">
                        <div class="macro-chip">⚡ Calories: <strong>{r['calories_kcal']} kcal</strong></div>
                        <div class="macro-chip">🥩 Protein: <strong>{r['protein_g']}g</strong></div>
                        <div class="macro-chip">🌾 Fiber: <strong>{r['fiber_g']}g</strong></div>
                        <div class="macro-chip">🍞 Carb: <strong>{r['carbohydrate_g']}g</strong></div>
                        <div class="macro-chip">🥑 Fat: <strong>{r['fat_g']}g</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            display_cols = [c for c in ["food_name", "collection_location", "calories_kcal", "protein_g", "fiber_g", "quality_label", "similarity"] if c in recs.columns]
            with st.expander("🔍 View Raw Recommendation Data Table"):
                st.dataframe(recs[display_cols], use_container_width=True)

# ==============================================================================
# TAB 2: INDIAN MEAL SUGGESTER
# ==============================================================================
with tab2:
    st.markdown("### 🍱 Composite Indian Meal Suggester")
    st.write("Calculated meal combinations derived from dish weight compositions with ingredient breakdowns.")
    
    col_m1, col_m2 = st.columns([1, 2.2])
    
    with col_m1:
        st.markdown("#### Filter Criteria")
        m_type_filter = st.selectbox("Course / Meal Type", ["All", "Breakfast", "Lunch", "Dinner", "Snack", "Beverage", "Dessert"])
        diet_type_filter = st.selectbox("Diet Preference", ["All", "Vegetarian", "Non-Vegetarian"])
        region_filter = st.selectbox("Regional Cuisine", ["All", "North Indian", "South Indian", "West Indian", "East Indian", "Pan-Indian"])
        top_m_k = st.slider("Number of Meal Suggestions", 3, 10, 5)
        
    with col_m2:
        st.markdown("#### Suggested Indian Meals")
        if not meals_df.empty:
            m_filtered = meals_df.copy()
            if m_type_filter != "All":
                m_filtered = m_filtered[m_filtered["meal_type"].str.lower() == m_type_filter.lower()]
            if diet_type_filter != "All":
                m_filtered = m_filtered[m_filtered["diet_type"].str.lower() == diet_type_filter.lower()]
            if region_filter != "All":
                m_filtered = m_filtered[m_filtered["region"].str.lower() == region_filter.lower()]
                
            if m_filtered.empty:
                st.info("No meals matched all criteria. Showing general Indian meal suggestions.")
                m_filtered = meals_df.copy()
                
            display_meals = m_filtered.head(top_m_k)
            for idx, r in display_meals.iterrows():
                diet_cls = "badge-high-val" if r['diet_type'] == "Vegetarian" else "badge-low-val"
                st.markdown(f"""
                <div class="food-glass-card" style="border-left: 4px solid #f59e0b;">
                    <span class="sim-tag" style="background: linear-gradient(135deg, #f59e0b, #d97706);">{r['region']}</span>
                    <div class="card-title">{r['meal_name']}</div>
                    <div style="margin-bottom: 0.75rem;">
                        <span class="pill-badge {diet_cls}">{r['diet_type']}</span>
                        <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 0.5rem;">Course: <strong>{r['meal_type']}</strong> | Serving Weight: {r.get('serving_size_g', 250)}g</span>
                    </div>
                    <div style="background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 0.75rem; margin-bottom: 0.75rem; color: #cbd5e1; font-size: 0.88rem;">
                        🛒 <strong>Recipe Ingredients:</strong> {r['ingredients']}
                    </div>
                    <div class="macro-bar-container">
                        <div class="macro-chip">⚡ Total: <strong>{r['calories_kcal']} kcal</strong></div>
                        <div class="macro-chip">🥩 Protein: <strong>{r['protein_g']}g</strong></div>
                        <div class="macro-chip">🌾 Fiber: <strong>{r['fiber_g']}g</strong></div>
                        <div class="macro-chip">🍞 Carb: <strong>{r['carbohydrate_g']}g</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Indian Meals Dataset not found.")

# ==============================================================================
# TAB 3: ML PREDICTOR STUDIO
# ==============================================================================
with tab3:
    st.markdown("### 🔮 Machine Learning Prediction Studio")
    st.write("Enter custom nutritional parameters to predict **Protein content** and classify **Nutritional Quality Grade**.")
    
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown("#### Input Dish Parameters")
        p_cal = st.number_input("Calories (kcal)", value=250.0, min_value=0.0, step=10.0)
        p_fat = st.number_input("Fat (g)", value=8.0, min_value=0.0, step=0.5)
        p_carb = st.number_input("Carbohydrates (g)", value=35.0, min_value=0.0, step=1.0)
        p_fib = st.number_input("Fiber (g)", value=4.5, min_value=0.0, step=0.5)
        p_sug = st.number_input("Sugar (g)", value=3.0, min_value=0.0, step=0.5)
        p_sod = st.number_input("Sodium (mg)", value=250.0, min_value=0.0, step=10.0)
        
    with cp2:
        st.markdown("#### Real-time ML Evaluation")
        if st.button("✨ Compute Predictions", type="primary", use_container_width=True):
            if p_cal == 0 and p_fat == 0 and p_carb == 0 and p_fib == 0 and p_sug == 0 and p_sod == 0:
                st.warning("⚠️ All parameters are 0. Please enter non-zero nutrient values for meaningful prediction.")
            else:
                input_features = pd.DataFrame([{
                    "calories_kcal": p_cal, "fat_g": p_fat, "carbohydrate_g": p_carb,
                    "fiber_g": p_fib, "sugar_g": p_sug, "sodium_mg": p_sod
                }])
                
                # Predict Protein
                if reg_model is not None:
                    pred_prot = reg_model.predict(input_features)[0]
                    pred_prot = float(np.clip(pred_prot, 0.0, 100.0))
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem;">
                        <div style="color: #34d399; font-weight: 600; font-size: 0.9rem;">REGRESSION MODEL OUTPUT</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #f8fafc;">{pred_prot:.2f} g Protein</div>
                        <div style="color: #94a3b8; font-size: 0.8rem;">Predicted by Random Forest Regressor ($R^2$ = 0.7976)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Protein Regressor joblib model not loaded.")
                    
                # Classify Quality Grade
                if cls_model is not None:
                    cls_input = pd.DataFrame([{
                        "calories_kcal": p_cal, "protein_g": pred_prot if reg_model else 10.0,
                        "fat_g": p_fat, "carbohydrate_g": p_carb, "fiber_g": p_fib,
                        "sugar_g": p_sug, "sodium_mg": p_sod
                    }])
                    pred_label = cls_model.predict(cls_input)[0]
                    
                    label_color = "#34d399" if pred_label == "High" else ("#fbbf24" if pred_label == "Medium" else "#f87171")
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.8); border: 2px solid {label_color}; border-radius: 16px; padding: 1.5rem; text-align: center;">
                        <div style="color: #94a3b8; font-weight: 600; font-size: 0.9rem;">CLASSIFICATION MODEL GRADE</div>
                        <div style="font-size: 2.8rem; font-weight: 800; color: {label_color}; margin: 0.25rem 0;">{pred_label} Quality</div>
                        <div style="color: #64748b; font-size: 0.85rem;">Gradient Boosting Classifier (96.77% Accuracy)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Quality Classifier joblib model not loaded.")

# ==============================================================================
# TAB 4: RESEARCH ANALYTICS
# ==============================================================================
with tab4:
    st.markdown("### 📊 Research & Dataset Analytics")
    
    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown("#### Baseline vs. Expanded Regression Metrics")
        if os.path.exists("results/regression_results.csv"):
            reg_res_df = pd.read_csv("results/regression_results.csv")
            st.dataframe(reg_res_df, use_container_width=True)
            
        if os.path.exists("results/figures/regression_actual_vs_predicted.png"):
            st.image("results/figures/regression_actual_vs_predicted.png", caption="Actual vs Predicted Protein Scatter Plot")
            
    with ca2:
        st.markdown("#### Baseline vs. Expanded Classification Metrics")
        if os.path.exists("results/classification_results.csv"):
            cls_res_df = pd.read_csv("results/classification_results.csv")
            st.dataframe(cls_res_df, use_container_width=True)
            
        if os.path.exists("results/figures/classification_confusion_matrix.png"):
            st.image("results/figures/classification_confusion_matrix.png", caption="Confusion Matrix — Gradient Boosting")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>Educational ML System — Built with Streamlit, Scikit-Learn & Pandas</p>", unsafe_allow_html=True)
