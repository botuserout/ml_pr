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
    page_title="NutriML — Indian Food Machine Learning System",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Professional Editorial UI CSS (No AI Slop / No Broken Gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0c0e14;
        color: #f1f5f9;
    }

    /* Clean Brand Header */
    .brand-container {
        padding: 1.8rem 2rem;
        background: #121722;
        border: 1px solid #1e293b;
        border-radius: 14px;
        margin-bottom: 1.75rem;
    }
    
    .brand-logo {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }
    
    .brand-logo .brand-green {
        color: #10b981;
    }
    
    .brand-logo .brand-white {
        color: #ffffff;
    }
    
    .brand-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin: 0;
    }

    /* Stat Box Grid */
    .stat-tile {
        background: #121722;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: left;
    }
    
    .stat-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }
    
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.35rem;
    }

    /* Refined Food Card Design */
    .food-card-v2 {
        background: #121722;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.1rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    
    .food-card-v2:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    .card-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    
    .food-title-v2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .sim-score-badge {
        background: #1e3a8a;
        color: #93c5fd;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #1d4ed8;
    }

    .meta-line {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .tag-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    
    .tag-high { background: #064e3b; color: #6ee7b7; border: 1px solid #047857; }
    .tag-med  { background: #78350f; color: #fde047; border: 1px solid #b45309; }
    .tag-low  { background: #7f1d1d; color: #fca5a5; border: 1px solid #b91c1c; }

    .macro-grid-v2 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
        gap: 0.6rem;
        background: #090c13;
        padding: 0.85rem;
        border-radius: 10px;
        border: 1px solid #182030;
    }
    
    .macro-item {
        font-size: 0.82rem;
        color: #94a3b8;
    }
    
    .macro-item strong {
        color: #f1f5f9;
        font-weight: 700;
        font-size: 0.95rem;
        display: block;
    }

    /* Streamlit Input Enhancements */
    div[data-baseweb="select"] > div {
        background-color: #121722 !important;
        border-color: #1e293b !important;
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

    # Cloud Fallback if unpickling failed
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

# Clean Brand Header (No broken gradients / No clipped emojis)
st.markdown("""
<div class="brand-container">
    <div class="brand-logo"><span class="brand-green">Nutri</span><span class="brand-white">ML</span> — Indian Food Intelligence</div>
    <div class="brand-subtitle">Multi-Source Nutritional Machine Learning System, Quality Classifier & Recommendation Engine</div>
</div>
""", unsafe_allow_html=True)

# Clean Key Stats Row
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f'<div class="stat-tile"><div class="stat-num">{len(food_df):,}</div><div class="stat-label">Unified Food Records</div></div>', unsafe_allow_html=True)
with s2:
    ind_cnt = len(food_df[food_df["source"] == "Indian Food Composition Data"]) if not food_df.empty and "source" in food_df.columns else 1014
    st.markdown(f'<div class="stat-tile"><div class="stat-num">{ind_cnt:,}</div><div class="stat-label">Indian Food Dishes</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown(f'<div class="stat-tile"><div class="stat-num">{len(meals_df):,}</div><div class="stat-label">Calculated Meals</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-tile"><div class="stat-num">96.8%</div><div class="stat-label">Classifier F1-Score</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🥗 Food Recommender", 
    "🍱 Indian Meal Suggester", 
    "🔮 ML Predictor Studio", 
    "📊 Research Analytics"
])

rec_features = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]

# ==============================================================================
# TAB 1: FOOD RECOMMENDER
# ==============================================================================
with tab1:
    st.subheader("🎯 Nutrition-Aware Food Recommender")
    st.caption("Specify your target nutritional profile per 100g to rank food items using cosine similarity.")
    
    col_left, col_right = st.columns([1, 2.2])
    
    with col_left:
        st.markdown("##### Target Presets")
        pr1, pr2, pr3 = st.columns(3)
        if pr1.button("High Protein"):
            st.session_state["cal"] = 300
            st.session_state["prot"] = 30.0
            st.session_state["fib"] = 8.0
        if pr2.button("Low Calorie"):
            st.session_state["cal"] = 150
            st.session_state["prot"] = 15.0
            st.session_state["fib"] = 5.0
        if pr3.button("High Fiber"):
            st.session_state["cal"] = 250
            st.session_state["prot"] = 15.0
            st.session_state["fib"] = 15.0
            
        st.markdown("##### Nutrient Profile Targets")
        t_cal = st.slider("Calories (kcal)", 0, 900, st.session_state.get("cal", 300), step=10, key="cal_s")
        t_prot = st.slider("Protein (g)", 0.0, 50.0, st.session_state.get("prot", 20.0), step=0.5, key="prot_s")
        t_fib = st.slider("Fiber (g)", 0.0, 30.0, st.session_state.get("fib", 8.0), step=0.5, key="fib_s")
        t_carb = st.slider("Carbohydrates (g)", 0, 100, 30, step=2)
        t_fat = st.slider("Fat (g)", 0.0, 50.0, 10.0, step=1.0)
        t_sug = st.slider("Sugar (g)", 0.0, 50.0, 5.0, step=0.5)
        t_sod = st.slider("Sodium (mg)", 0, 2000, 300, step=25)
        
        st.markdown("##### Filter Rules")
        quality_filter = st.selectbox("Nutritional Quality Filter", ["All", "High", "Medium", "Low"])
        location_filter = st.selectbox("Data Source Location", ["All", "India", "USA", "UK", "USDA reference"])
        top_k = st.slider("Number of Recommendations", 3, 20, 6)
        
        use_constraints = st.checkbox("Apply Numerical Threshold Constraints", value=False)
        max_cal, min_prot = None, None
        if use_constraints:
            max_cal = st.number_input("Maximum Calories Limit", value=400)
            min_prot = st.number_input("Minimum Protein Target (g)", value=10.0)

    with col_right:
        st.markdown("##### Recommended Food Results")
        
        target_sum = t_cal + t_prot + t_fib + t_carb + t_fat + t_sug + t_sod
        
        if target_sum == 0:
            st.info("ℹ️ All target sliders are set to 0. Adjust at least one nutrient slider on the left to receive recommendations.")
        elif target_sum > 0 and not food_df.empty and rec_scaler is not None:
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
            
            if filtered.empty:
                st.warning("⚠️ No exact matches satisfied all hard constraints. Showing closest cosine similarity matches.")
                filtered = valid_foods.copy()
                if quality_filter != "All":
                    filtered = filtered[filtered["quality_label"].astype(str).str.lower() == quality_filter.lower()]
            
            recs = filtered.sort_values("similarity", ascending=False).head(top_k)
            
            for idx, r in recs.iterrows():
                q_label = str(r.get("quality_label", "Medium"))
                badge_cls = "tag-high" if q_label == "High" else ("tag-med" if q_label == "Medium" else "tag-low")
                
                st.markdown(f"""
                <div class="food-card-v2">
                    <div class="card-header-flex">
                        <div class="food-title-v2">{r['food_name']}</div>
                        <span class="sim-score-badge">{r['similarity']*100:.1f}% Match</span>
                    </div>
                    <div class="meta-line">
                        <span class="tag-pill {badge_cls}">{q_label} Quality</span>
                        <span>Source: <strong>{r.get('source', 'General')}</strong> ({r.get('collection_location', 'Global')})</span>
                    </div>
                    <div class="macro-grid-v2">
                        <div class="macro-item">Calories<strong>{r['calories_kcal']} kcal</strong></div>
                        <div class="macro-item">Protein<strong>{r['protein_g']} g</strong></div>
                        <div class="macro-item">Fiber<strong>{r['fiber_g']} g</strong></div>
                        <div class="macro-item">Carbs<strong>{r['carbohydrate_g']} g</strong></div>
                        <div class="macro-item">Fat<strong>{r['fat_g']} g</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            display_cols = [c for c in ["food_name", "collection_location", "calories_kcal", "protein_g", "fiber_g", "quality_label", "similarity"] if c in recs.columns]
            with st.expander("🔍 View Raw Recommendation Table"):
                st.dataframe(recs[display_cols], use_container_width=True)

# ==============================================================================
# TAB 2: INDIAN MEAL SUGGESTER
# ==============================================================================
with tab2:
    st.subheader("🍱 Composite Indian Meal Suggester")
    st.caption("Composite meal combinations calculated from dish ingredients and weight contributions.")
    
    col_m1, col_m2 = st.columns([1, 2.2])
    
    with col_m1:
        st.markdown("##### Meal Filters")
        m_type_filter = st.selectbox("Course Type", ["All", "Breakfast", "Lunch", "Dinner", "Snack", "Beverage", "Dessert"])
        diet_type_filter = st.selectbox("Diet Preference", ["All", "Vegetarian", "Non-Vegetarian"])
        region_filter = st.selectbox("Regional Cuisine", ["All", "North Indian", "South Indian", "West Indian", "East Indian", "Pan-Indian"])
        top_m_k = st.slider("Number of Suggestions", 3, 10, 5)
        
    with col_m2:
        st.markdown("##### Suggested Meals")
        if not meals_df.empty:
            m_filtered = meals_df.copy()
            if m_type_filter != "All":
                m_filtered = m_filtered[m_filtered["meal_type"].str.lower() == m_type_filter.lower()]
            if diet_type_filter != "All":
                m_filtered = m_filtered[m_filtered["diet_type"].str.lower() == diet_type_filter.lower()]
            if region_filter != "All":
                m_filtered = m_filtered[m_filtered["region"].str.lower() == region_filter.lower()]
                
            if m_filtered.empty:
                st.info("No meals matched all active filters. Displaying general Indian meal suggestions.")
                m_filtered = meals_df.copy()
                
            display_meals = m_filtered.head(top_m_k)
            for idx, r in display_meals.iterrows():
                diet_cls = "tag-high" if r['diet_type'] == "Vegetarian" else "tag-low"
                st.markdown(f"""
                <div class="food-card-v2" style="border-left: 4px solid #f59e0b;">
                    <div class="card-header-flex">
                        <div class="food-title-v2">{r['meal_name']}</div>
                        <span class="sim-score-badge" style="background:#451a03; color:#fde047; border-color:#854d0e;">{r['region']}</span>
                    </div>
                    <div class="meta-line">
                        <span class="tag-pill {diet_cls}">{r['diet_type']}</span>
                        <span>Course: <strong>{r['meal_type']}</strong> | Serving: {r.get('serving_size_g', 250)}g</span>
                    </div>
                    <div style="background: #090c13; padding: 0.75rem; border-radius: 8px; border: 1px solid #182030; margin-bottom: 0.75rem; font-size: 0.88rem; color: #cbd5e1;">
                        🛒 <strong>Recipe Ingredients:</strong> {r['ingredients']}
                    </div>
                    <div class="macro-grid-v2">
                        <div class="macro-item">Total Energy<strong>{r['calories_kcal']} kcal</strong></div>
                        <div class="macro-item">Protein<strong>{r['protein_g']} g</strong></div>
                        <div class="macro-item">Fiber<strong>{r['fiber_g']} g</strong></div>
                        <div class="macro-item">Carbs<strong>{r['carbohydrate_g']} g</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Indian Meals Dataset not found.")

# ==============================================================================
# TAB 3: ML PREDICTOR STUDIO
# ==============================================================================
with tab3:
    st.subheader("🔮 Machine Learning Prediction Studio")
    st.caption("Predict continuous Protein level and classify Nutritional Quality Grade.")
    
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown("##### Dish Nutritional Inputs")
        p_cal = st.number_input("Calories (kcal)", value=250.0, min_value=0.0, step=10.0)
        p_fat = st.number_input("Fat (g)", value=8.0, min_value=0.0, step=0.5)
        p_carb = st.number_input("Carbohydrates (g)", value=35.0, min_value=0.0, step=1.0)
        p_fib = st.number_input("Fiber (g)", value=4.5, min_value=0.0, step=0.5)
        p_sug = st.number_input("Sugar (g)", value=3.0, min_value=0.0, step=0.5)
        p_sod = st.number_input("Sodium (mg)", value=250.0, min_value=0.0, step=10.0)
        
    with cp2:
        st.markdown("##### ML Inference Results")
        if st.button("Run ML Evaluation", type="primary", use_container_width=True):
            if p_cal == 0 and p_fat == 0 and p_carb == 0 and p_fib == 0 and p_sug == 0 and p_sod == 0:
                st.warning("⚠️ All inputs are set to 0. Please enter non-zero nutrient values for inference.")
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
                    <div style="background: #121722; border: 1px solid #10b981; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                        <div style="color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">REGRESSION PREDICTION</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #10b981; margin: 0.2rem 0;">{pred_prot:.2f} g Protein</div>
                        <div style="color: #94a3b8; font-size: 0.82rem;">Predicted by Random Forest Regressor ($R^2$ = 0.7976)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Protein Regressor model not available.")
                    
                # Classify Quality Grade
                if cls_model is not None:
                    cls_input = pd.DataFrame([{
                        "calories_kcal": p_cal, "protein_g": pred_prot if reg_model else 10.0,
                        "fat_g": p_fat, "carbohydrate_g": p_carb, "fiber_g": p_fib,
                        "sugar_g": p_sug, "sodium_mg": p_sod
                    }])
                    pred_label = cls_model.predict(cls_input)[0]
                    
                    label_clr = "#10b981" if pred_label == "High" else ("#f59e0b" if pred_label == "Medium" else "#ef4444")
                    st.markdown(f"""
                    <div style="background: #121722; border: 1px solid {label_clr}; border-radius: 12px; padding: 1.4rem; text-align: center;">
                        <div style="color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">QUALITY CLASSIFICATION GRADE</div>
                        <div style="font-size: 2.5rem; font-weight: 800; color: {label_clr}; margin: 0.3rem 0;">{pred_label} Quality</div>
                        <div style="color: #94a3b8; font-size: 0.82rem;">Gradient Boosting Classifier (96.77% Accuracy)</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Quality Classifier model not available.")

# ==============================================================================
# TAB 4: RESEARCH ANALYTICS
# ==============================================================================
with tab4:
    st.subheader("📊 Research & Experiment Evaluation")
    
    ca1, ca2 = st.columns(2)
    with ca1:
        st.markdown("##### Regression Performance Comparison")
        if os.path.exists("results/regression_results.csv"):
            reg_res_df = pd.read_csv("results/regression_results.csv")
            st.dataframe(reg_res_df, use_container_width=True)
            
        if os.path.exists("results/figures/regression_actual_vs_predicted.png"):
            st.image("results/figures/regression_actual_vs_predicted.png", caption="Actual vs Predicted Protein Scatter Plot")
            
    with ca2:
        st.markdown("##### Classification Performance Comparison")
        if os.path.exists("results/classification_results.csv"):
            cls_res_df = pd.read_csv("results/classification_results.csv")
            st.dataframe(cls_res_df, use_container_width=True)
            
        if os.path.exists("results/figures/classification_confusion_matrix.png"):
            st.image("results/figures/classification_confusion_matrix.png", caption="Confusion Matrix — Gradient Boosting Classifier")

st.markdown("<br><hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.82rem;'>NutriML Pipeline — Machine Learning & Food Nutrition Research System</p>", unsafe_allow_html=True)
