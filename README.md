# Machine Learning-Based Indian Food Nutrition, Quality Classification & Recommendation System

## 📌 Project Overview
This project presents a multi-source machine learning framework that integrates international food composition data (**USDA FoodData Central**, **Open Food Facts**) with a curated **Indian Food Composition & Composite Meal Dataset**.

The system performs:
1. **Multi-Source Data Acquisition & Schema Standardization**
2. **Data Preparation & Provenance Profiling**
3. **Nutritional Quality Classification** (Low, Medium, High)
4. **Nutrient Prediction Regression Models** (Predicting `protein_g` & `calories_kcal`)
5. **Baseline vs. Expanded ML Model Comparison** (Demonstrating empirical value of Indian food data integration)
6. **Constraint-Based Food & Meal Recommendation Engine** (Cosine similarity with hard constraints & natural language explanations)

---

## 🏗️ System Architecture

```text
  General International Food Data (USDA + Open Food Facts)
                         +
    Indian Food Data (Indian_Food_Nutrition_Processed)
                         +
    Calculated Composite Indian Meal Dataset (~200 records)
                         │
                         ▼
        Multi-Source Data Integration Layer
                         │
                         ▼
        Data Cleaning, Unit Standardization & Profiling
                         │
                         ▼
         Feature Engineering & Quality Scoring
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Regression    Classification  Recommendation
     Predicting    Quality Label   Food & Meal
     Nutrients     (Low/Med/High)  Suggestions
```

---

## 📁 Repository Structure
```text
ml_pr/
├── food_nutrition_ml_project_multisource_complete.py  # Main pipeline script
├── Indian_Food_Nutrition_Processed.csv               # Raw Indian food composition dataset
├── requirements.txt                                  # Dependency manifest
├── README.md                                         # Project documentation
├── tests/
│   └── test_pipeline.py                              # Automated unit tests
├── data/
│   ├── raw/                                          # Raw ingested datasets
│   ├── interim/                                      # Intermediate normalized data
│   └── processed/                                    # Processed datasets
│       ├── indian_foods_clean.csv
│       ├── indian_meals_clean.csv
│       ├── food_nutrition_cleaned.csv
│       └── unified_food_dataset.csv
└── results/                                          # Analytical reports & ML outputs
    ├── data_quality_report.csv
    ├── ingredient_matching_report.csv
    ├── regression_results.csv
    ├── classification_results.csv
    └── example_recommendations.csv
```

---

## 📊 Ingested Datasets
- **USDA SR Legacy**: Standard reference database for nutritional baseline.
- **Open Food Facts**: Multi-country barcode dataset (India, USA, UK).
- **Indian Food Dataset**: 1,000+ representative Indian dishes covering North, South, East, West, and Pan-Indian cuisines with regional and dietary labels.
- **Composite Indian Meal Dataset**: 200 calculated composite meal recipes (e.g., *Dal Rice*, *Paneer Paratha with Curd*) computed using exact ingredient weights and nutrient aggregation formulas.

---

## 🚀 How to Run in Google Colab & Host on Streamlit

### Option A: Running in Google Colab
1. Upload `food_nutrition_ml_project_multisource_complete.py` and `Indian_Food_Nutrition_Processed.csv` to your Google Colab environment.
2. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
3. Run the complete pipeline script:
   ```bash
   !python food_nutrition_ml_project_multisource_complete.py
   ```

### Option B: Running Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### Option C: Hosting Recommender on Streamlit
Import the trained recommendation functions in your Streamlit app (`app.py`):
```python
from food_nutrition_ml_project_multisource_complete import recommend_foods, recommend_meals

# Interface controls
recs = recommend_foods(target_dict, top_k=10, min_protein=15, max_calories=500)
st.dataframe(recs)
```

---

## ⚠️ Limitations & Ethical Disclaimer
- **Educational Framework**: The nutritional quality score (0–100) is an educational scoring system and does not constitute medical or clinical advice.
- **Data Provenance**: Nutritional composition can vary by crop variety, cooking method, brand, and regional preparation.
- **No FSSAI Endorsement**: Claims are purely research-driven and not officially endorsed by government regulatory bodies.
