# ==============================================================================
# SECTION 1: Project Overview & Imports
# ==============================================================================
# # Food Nutrition Machine Learning Project — Multi-Source & Indian Food Pipeline
# ## Multi-Source Data Acquisition, Data Preparation, Indian Food Integration, Meal Construction, Nutrient Prediction, Nutritional-Quality Classification, and Nutrition-Aware Food & Meal Recommendation
#
# **Project type:** Educational Machine Learning / Data Preparation and Analysis Project.
#
# This script extends the original multi-source food nutrition pipeline (USDA + Open Food Facts) by adding an Indian food composition dataset, an Indian recipe/meal calculation framework, extended EDA, baseline vs. expanded model comparison, model persistence (.joblib), plot exports (.png), and an upgraded recommendation engine with nutritional constraints and explanations.

import io
import os
import re
import time
import zipfile
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_STATE = 42
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 80)

# Create standard directory structure
os.makedirs("data/raw/indian_foods", exist_ok=True)
os.makedirs("data/raw/external_sources", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

print("Environment, directories, and models folder ready.")

# ==============================================================================
# SECTION 2: Data Source Configuration
# ==============================================================================
OFF_BASE_URL = "https://world.openfoodfacts.org/api/v2/search"

OFF_FIELDS = ",".join([
    "code", "product_name", "brands", "categories", "countries",
    "countries_tags_en", "food_groups_tags_en", "nutriments",
    "nutrition_grades", "nova_group", "serving_size", "quantity"
])

OFF_HEADERS = {
    "User-Agent": "FoodNutritionMLStudentProject/1.0 (educational project)"
}

OFF_PAGES_PER_COUNTRY = 3
OFF_PAGE_SIZE = 50

COUNTRIES = {
    "India": "india",
    "USA": "united-states",
    "UK": "united-kingdom"
}

# ==============================================================================
# SECTION 3: Open Food Facts Data Acquisition
# ==============================================================================
def collect_openfoodfacts_country(country_name, country_tag, pages=3, page_size=50, sleep_seconds=2):
    records = []
    for page in range(1, pages + 1):
        params = {
            "countries_tags_en": country_tag,
            "page": page,
            "page_size": page_size,
            "fields": OFF_FIELDS
        }
        try:
            r = requests.get(OFF_BASE_URL, params=params, headers=OFF_HEADERS, timeout=30)
            r.raise_for_status()
            payload = r.json()
            products = payload.get("products", [])

            for p in products:
                nutr = p.get("nutriments") or {}
                records.append({
                    "food_id": p.get("code"),
                    "food_name": p.get("product_name"),
                    "brand": p.get("brands"),
                    "categories": p.get("categories"),
                    "countries": p.get("countries"),
                    "collection_location": country_name,
                    "calories_kcal": nutr.get("energy-kcal_100g"),
                    "protein_g": nutr.get("proteins_100g"),
                    "carbohydrate_g": nutr.get("carbohydrates_100g"),
                    "fat_g": nutr.get("fat_100g"),
                    "fiber_g": nutr.get("fiber_100g"),
                    "sugar_g": nutr.get("sugars_100g"),
                    "sodium_mg": nutr.get("sodium_100g"),
                    "saturated_fat_g": nutr.get("saturated-fat_100g"),
                    "salt_g": nutr.get("salt_100g"),
                    "nutriscore": p.get("nutrition_grades"),
                    "nova_group": p.get("nova_group"),
                    "serving_size": p.get("serving_size"),
                    "quantity": p.get("quantity"),
                    "source": "Open Food Facts"
                })
            print(f"OFF {country_name}: page {page} -> {len(products)} items")
            if page < pages:
                time.sleep(sleep_seconds)
        except Exception as e:
            print(f"OFF Collection warning for {country_name} page {page}: {e}")
            break

    return pd.DataFrame(records)

off_file_path = "data/raw/openfoodfacts_multilocation.csv"
if os.path.exists(off_file_path):
    print("Loading cached Open Food Facts data...")
    off_raw = pd.read_csv(off_file_path)
else:
    print("Collecting fresh Open Food Facts data...")
    off_frames = []
    for c_name, c_tag in COUNTRIES.items():
        frame = collect_openfoodfacts_country(c_name, c_tag, pages=OFF_PAGES_PER_COUNTRY, page_size=OFF_PAGE_SIZE)
        off_frames.append(frame)
    off_raw = pd.concat(off_frames, ignore_index=True)
    off_raw.to_csv(off_file_path, index=False)

print("Open Food Facts raw shape:", off_raw.shape)

# ==============================================================================
# SECTION 4: USDA SR Legacy Data Acquisition
# ==============================================================================
usda_standard_path = "data/raw/usda_standardized.csv"
if os.path.exists(usda_standard_path):
    print("Loading cached USDA data...")
    usda_standard = pd.read_csv(usda_standard_path)
else:
    print("Downloading USDA SR Legacy dataset...")
    USDA_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"
    response = requests.get(USDA_URL, timeout=120)
    response.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(response.content))
    names = z.namelist()

    def find_file(part):
        matches = [n for n in names if n.lower().endswith("/" + part.lower()) or n.lower() == part.lower()]
        return matches[0]

    food_file = find_file("food.csv")
    nutrient_file = find_file("food_nutrient.csv")
    lookup_file = find_file("nutrient.csv")

    usda_food = pd.read_csv(z.open(food_file), low_memory=False)
    usda_food_nutrient = pd.read_csv(z.open(nutrient_file), low_memory=False)
    usda_nutrient_lookup = pd.read_csv(z.open(lookup_file), low_memory=False)

    merged_usda = usda_food_nutrient.merge(
        usda_nutrient_lookup[["id", "name"]],
        left_on="nutrient_id",
        right_on="id",
        how="left"
    )
    wide_usda = merged_usda.pivot_table(
        index="fdc_id",
        columns="name",
        values="amount",
        aggfunc="mean"
    ).reset_index()

    usda_data = usda_food[["fdc_id", "description"]].merge(wide_usda, on="fdc_id", how="inner")

    def find_col(columns, patterns):
        for col in columns:
            low = str(col).lower()
            if all(p in low for p in patterns):
                return col
        return None

    usda_map = {
        "calories_kcal": find_col(usda_data.columns, ["energy", "kcal"]),
        "protein_g": find_col(usda_data.columns, ["protein"]),
        "carbohydrate_g": find_col(usda_data.columns, ["carbohydrate"]),
        "fat_g": find_col(usda_data.columns, ["total", "lipid"]),
        "fiber_g": find_col(usda_data.columns, ["fiber"]),
        "sugar_g": find_col(usda_data.columns, ["sugars"]),
        "sodium_mg": find_col(usda_data.columns, ["sodium"]),
        "saturated_fat_g": find_col(usda_data.columns, ["fatty", "saturated"])
    }

    usda_standard = pd.DataFrame({
        "food_id": usda_data["fdc_id"].astype(str),
        "food_name": usda_data["description"],
        "brand": np.nan,
        "categories": np.nan,
        "countries": np.nan,
        "collection_location": "USDA reference",
        "source": "USDA FoodData Central SR Legacy"
    })
    for std_col, src_col in usda_map.items():
        if src_col:
            usda_standard[std_col] = pd.to_numeric(usda_data[src_col], errors="coerce")
        else:
            usda_standard[std_col] = np.nan

    usda_standard["nutriscore"] = np.nan
    usda_standard["nova_group"] = np.nan
    usda_standard["serving_size"] = np.nan
    usda_standard["quantity"] = np.nan
    usda_standard["salt_g"] = np.nan
    usda_standard.to_csv(usda_standard_path, index=False)

print("USDA Standardized shape:", usda_standard.shape)

# ==============================================================================
# SECTION 5: Indian Food Composition Dataset Processing
# ==============================================================================
indian_raw_file = "Indian_Food_Nutrition_Processed.csv"
if os.path.exists(indian_raw_file):
    print("Loading local Indian Food Nutrition dataset...")
    ind_df = pd.read_csv(indian_raw_file)

    # Column mapping to unified schema
    ind_mapped = pd.DataFrame({
        "food_id": ["IND_FOOD_" + str(i + 1) for i in range(len(ind_df))],
        "food_name": ind_df["Dish Name"],
        "brand": np.nan,
        "categories": "Indian Food",
        "countries": "India",
        "collection_location": "India",
        "calories_kcal": pd.to_numeric(ind_df["Calories (kcal)"], errors="coerce"),
        "protein_g": pd.to_numeric(ind_df["Protein (g)"], errors="coerce"),
        "carbohydrate_g": pd.to_numeric(ind_df["Carbohydrates (g)"], errors="coerce"),
        "fat_g": pd.to_numeric(ind_df["Fats (g)"], errors="coerce"),
        "fiber_g": pd.to_numeric(ind_df["Fibre (g)"], errors="coerce"),
        "sugar_g": pd.to_numeric(ind_df["Free Sugar (g)"], errors="coerce"),
        "sodium_mg": pd.to_numeric(ind_df["Sodium (mg)"], errors="coerce"),
        "saturated_fat_g": np.nan,
        "salt_g": np.nan,
        "nutriscore": np.nan,
        "nova_group": np.nan,
        "serving_size": "100 g",
        "quantity": "100 g",
        "source": "Indian Food Composition Data"
    })

    # Deduplicate and basic cleaning
    ind_mapped["food_name"] = ind_mapped["food_name"].astype("string").str.strip()
    ind_mapped = ind_mapped.dropna(subset=["food_name"]).drop_duplicates(subset=["food_name"])

    # Categorize region & diet_type
    def infer_region(name):
        n = str(name).lower()
        if any(x in n for x in ["dosa", "idli", "sambar", "rasam", "uttapam", "avial", "thoran", "payasam", "appam", "vada"]):
            return "South Indian"
        elif any(x in n for x in ["parantha", "paratha", "rajma", "chole", "kadhi", "paneer", "dal makhani", "naan", "bhatura", "saag"]):
            return "North Indian"
        elif any(x in n for x in ["dhokla", "thepla", "poha", "pav", "khichdi", "gatte", "puranpoli", "modak"]):
            return "West Indian"
        elif any(x in n for x in ["machher", "bengali", "momos", "thukpa", "rasgulla"]):
            return "East Indian"
        return "Pan-Indian"

    def infer_diet(name):
        n = str(name).lower()
        if any(x in n for x in ["chicken", "mutton", "egg", "fish", "keema", "meat", "pork", "salami", "bacon", "prawn"]):
            return "Non-Vegetarian"
        return "Vegetarian"

    ind_mapped["region"] = ind_mapped["food_name"].apply(infer_region)
    ind_mapped["diet_type"] = ind_mapped["food_name"].apply(infer_diet)

    ind_mapped.to_csv("data/processed/indian_foods_clean.csv", index=False)
    print("Indian Foods Cleaned shape:", ind_mapped.shape)
else:
    print("WARNING: Indian_Food_Nutrition_Processed.csv not found!")
    ind_mapped = pd.DataFrame()

# ==============================================================================
# SECTION 6: Combine Sources into Unified Dataset
# ==============================================================================
COMMON_COLS = [
    "food_id", "food_name", "brand", "categories", "countries",
    "collection_location", "calories_kcal", "protein_g", "carbohydrate_g",
    "fat_g", "fiber_g", "sugar_g", "sodium_mg", "saturated_fat_g", "salt_g",
    "nutriscore", "nova_group", "serving_size", "quantity", "source"
]

for col in COMMON_COLS:
    if col not in off_raw.columns:
        off_raw[col] = np.nan
    if col not in usda_standard.columns:
        usda_standard[col] = np.nan
    if not ind_mapped.empty and col not in ind_mapped.columns:
        ind_mapped[col] = np.nan

all_frames = [off_raw[COMMON_COLS], usda_standard[COMMON_COLS]]
if not ind_mapped.empty:
    all_frames.append(ind_mapped[COMMON_COLS])

raw_combined = pd.concat(all_frames, ignore_index=True)
raw_combined.to_csv("data/raw/food_master_raw.csv", index=False)
print("Combined raw dataset shape:", raw_combined.shape)

# ==============================================================================
# SECTION 7: Data Cleaning & Standard Preprocessing Pipeline
# ==============================================================================
clean_df = raw_combined.copy()

# Text standardization
text_fields = ["food_name", "brand", "categories", "countries", "collection_location", "source"]
for tf in text_fields:
    clean_df[tf] = clean_df[tf].astype("string").str.strip()

clean_df["food_name"] = clean_df["food_name"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
clean_df = clean_df.dropna(subset=["food_name"]).copy()

# Numeric fields validation
num_cols = ["calories_kcal", "protein_g", "carbohydrate_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
for nc in num_cols:
    clean_df[nc] = pd.to_numeric(clean_df[nc], errors="coerce")
    clean_df.loc[clean_df[nc] < 0, nc] = np.nan

# Exact & source-aware barcode deduplication
clean_df = clean_df.drop_duplicates()

off_part = clean_df[clean_df["source"] == "Open Food Facts"].drop_duplicates(subset=["food_id"], keep="first")
non_off_part = clean_df[clean_df["source"] != "Open Food Facts"]
clean_df = pd.concat([off_part, non_off_part], ignore_index=True)

# Completeness Filter (at least 5 of 7 core nutrients)
clean_df["core_nutrient_count"] = clean_df[num_cols].notna().sum(axis=1)
prepared = clean_df[clean_df["core_nutrient_count"] >= 5].copy()

prepared.to_csv("data/processed/unified_food_dataset.csv", index=False)
print(f"Prepared unified dataset shape: {prepared.shape} (from {len(raw_combined)} raw records)")

# Save Data Source Counts Figure
plt.figure(figsize=(8, 4))
prepared["source"].value_counts().plot(kind="bar", color=["#2b5c8f", "#d95f02", "#7570b3"])
plt.title("Record Distribution by Data Source")
plt.ylabel("Number of Clean Records")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/figures/data_quality_source_counts.png")
plt.close()

# ==============================================================================
# SECTION 8: Data Quality Report Generation
# ==============================================================================
def generate_data_quality_report(raw_df, clean_df, final_df):
    report_rows = []
    sources = raw_df["source"].unique()

    for s in sources:
        raw_sub = raw_df[raw_df["source"] == s]
        clean_sub = clean_df[clean_df["source"] == s]
        final_sub = final_df[final_df["source"] == s]

        raw_cnt = len(raw_sub)
        clean_cnt = len(clean_sub)
        final_cnt = len(final_sub)
        dups_removed = raw_cnt - clean_cnt
        excluded_cnt = clean_cnt - final_cnt
        missing_core_pct = (raw_sub[num_cols].isna().sum().sum() / (raw_cnt * len(num_cols))) * 100

        report_rows.append({
            "source": s,
            "raw_record_count": raw_cnt,
            "clean_record_count": clean_cnt,
            "duplicates_removed": dups_removed,
            "excluded_rows": excluded_cnt,
            "final_record_count": final_cnt,
            "missing_core_nutrient_pct": round(missing_core_pct, 2)
        })

    dq_report = pd.DataFrame(report_rows)
    dq_report.to_csv("results/data_quality_report.csv", index=False)
    print("\nData Quality Report:")
    print(dq_report.to_string(index=False))
    return dq_report

generate_data_quality_report(raw_combined, clean_df, prepared)

# ==============================================================================
# SECTION 9: Indian Meal Construction & Ingredient Matching
# ==============================================================================
def match_recipe_ingredient(raw_ingredient_name, food_df):
    raw_clean = str(raw_ingredient_name).lower().strip()
    if not raw_clean:
        return None, "unmatched", 0.0

    exact = food_df[food_df["food_name"].str.lower() == raw_clean]
    if not exact.empty:
        return exact.iloc[0]["food_id"], "exact", 1.0

    tokens = [t for t in re.split(r"\W+", raw_clean) if len(t) > 2]
    best_match = None
    best_score = 0

    for idx, row in food_df.iterrows():
        fname = str(row["food_name"]).lower()
        score = sum(1 for t in tokens if t in fname)
        if score > best_score:
            best_score = score
            best_match = row["food_id"]

    if best_score >= 2:
        return best_match, "high", 0.85
    elif best_score == 1:
        return best_match, "medium", 0.60
    else:
        return None, "unmatched", 0.0

if not ind_mapped.empty:
    ind_food_lookup = ind_mapped.set_index("food_id")
    meal_records = []
    matching_log = []

    base_dishes = ind_mapped.to_dict("records")
    meal_types = ["Breakfast", "Lunch", "Dinner", "Snack", "Beverage", "Dessert"]

    np.random.seed(RANDOM_STATE)
    for i in range(1, 201):
        primary_dish = base_dishes[(i - 1) % len(base_dishes)]
        secondary_dish = base_dishes[(i * 7) % len(base_dishes)]

        meal_id = f"IND_MEAL_{i:03d}"
        meal_name = f"{primary_dish['food_name']} with {secondary_dish['food_name']}"
        m_type = meal_types[i % len(meal_types)]
        reg = primary_dish.get("region", "Pan-Indian")
        diet = "Non-Vegetarian" if "Non-Vegetarian" in [primary_dish.get("diet_type"), secondary_dish.get("diet_type")] else "Vegetarian"

        w1, w2 = 150.0, 100.0

        c_kcal = ((w1 / 100.0) * (primary_dish["calories_kcal"] or 0)) + ((w2 / 100.0) * (secondary_dish["calories_kcal"] or 0))
        c_prot = ((w1 / 100.0) * (primary_dish["protein_g"] or 0)) + ((w2 / 100.0) * (secondary_dish["protein_g"] or 0))
        c_carb = ((w1 / 100.0) * (primary_dish["carbohydrate_g"] or 0)) + ((w2 / 100.0) * (secondary_dish["carbohydrate_g"] or 0))
        c_fat  = ((w1 / 100.0) * (primary_dish["fat_g"] or 0)) + ((w2 / 100.0) * (secondary_dish["fat_g"] or 0))
        c_fib  = ((w1 / 100.0) * (primary_dish["fiber_g"] or 0)) + ((w2 / 100.0) * (secondary_dish["fiber_g"] or 0))
        c_sug  = ((w1 / 100.0) * (primary_dish["sugar_g"] or 0)) + ((w2 / 100.0) * (secondary_dish["sugar_g"] or 0))
        c_sod  = ((w1 / 100.0) * (primary_dish["sodium_mg"] or 0)) + ((w2 / 100.0) * (secondary_dish["sodium_mg"] or 0))

        f1_id, conf1, score1 = match_recipe_ingredient(primary_dish["food_name"], ind_mapped)
        f2_id, conf2, score2 = match_recipe_ingredient(secondary_dish["food_name"], ind_mapped)

        matching_log.append({
            "meal_id": meal_id, "raw_ingredient": primary_dish["food_name"],
            "matched_food_id": f1_id, "confidence": conf1, "score": score1
        })
        matching_log.append({
            "meal_id": meal_id, "raw_ingredient": secondary_dish["food_name"],
            "matched_food_id": f2_id, "confidence": conf2, "score": score2
        })

        meal_records.append({
            "meal_id": meal_id,
            "meal_name": meal_name,
            "meal_type": m_type,
            "region": reg,
            "cuisine": f"{reg} Cuisine",
            "diet_type": diet,
            "ingredients": f"{primary_dish['food_name']} (150g), {secondary_dish['food_name']} (100g)",
            "serving_size_g": 250,
            "calories_kcal": round(c_kcal, 2),
            "protein_g": round(c_prot, 2),
            "carbohydrate_g": round(c_carb, 2),
            "fat_g": round(c_fat, 2),
            "fiber_g": round(c_fib, 2),
            "sugar_g": round(c_sug, 2),
            "sodium_mg": round(c_sod, 2),
            "source_name": "Calculated from Indian Food Composition Data",
            "source_type": "Composite Recipe Engine",
            "retrieved_date": "2026-09-01"
        })

    meals_df = pd.DataFrame(meal_records)
    meals_df.to_csv("data/processed/indian_meals_clean.csv", index=False)

    match_df = pd.DataFrame(matching_log)
    match_df.to_csv("results/ingredient_matching_report.csv", index=False)
    print(f"Generated Indian Meals Dataset: {meals_df.shape[0]} rows saved to data/processed/indian_meals_clean.csv")

# ==============================================================================
# SECTION 10: Feature Engineering & Quality Scoring
# ==============================================================================
df = prepared.copy()

df["protein_per_100_kcal"] = df["protein_g"] / df["calories_kcal"].replace(0, np.nan) * 100
df["fiber_per_100_kcal"]   = df["fiber_g"] / df["calories_kcal"].replace(0, np.nan) * 100
df["sugar_ratio"]          = df["sugar_g"] / df["carbohydrate_g"].replace(0, np.nan)
df["fat_ratio"]            = df["fat_g"] / (df["protein_g"] + df["carbohydrate_g"] + df["fat_g"]).replace(0, np.nan)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

quality_cols = ["calories_kcal", "protein_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
quality_df = df.dropna(subset=quality_cols).copy()

for c in quality_cols:
    lo = quality_df[c].quantile(0.01)
    hi = quality_df[c].quantile(0.99)
    quality_df[c + "_norm"] = 0.5 if hi == lo else ((quality_df[c] - lo) / (hi - lo)).clip(0, 1)

quality_df["quality_score_raw"] = (
    0.35 * quality_df["protein_g_norm"] +
    0.25 * quality_df["fiber_g_norm"] -
    0.15 * quality_df["sugar_g_norm"] -
    0.10 * quality_df["sodium_mg_norm"] -
    0.10 * quality_df["fat_g_norm"] -
    0.05 * quality_df["calories_kcal_norm"]
)

score_min = quality_df["quality_score_raw"].min()
score_max = quality_df["quality_score_raw"].max()
quality_df["quality_score"] = ((quality_df["quality_score_raw"] - score_min) / (score_max - score_min) * 100).round(2)

quality_df["quality_label"] = pd.cut(
    quality_df["quality_score"],
    bins=[-np.inf, 33.33, 66.67, np.inf],
    labels=["Low", "Medium", "High"]
)

df = df.merge(quality_df[["food_id", "quality_score", "quality_label"]], on="food_id", how="left")

final_columns = [c for c in df.columns if not c.endswith("_norm")]
final_dataset = df[final_columns].copy()
final_dataset.to_csv("data/processed/food_nutrition_cleaned.csv", index=False)

# ==============================================================================
# SECTION 11: Machine Learning Baseline vs. Expanded Comparison & Model Saving
# ==============================================================================
reg_features = ["calories_kcal", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]
cls_features = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]

def run_regression_experiments(dataset, exp_name):
    reg_df = dataset.dropna(subset=["protein_g"] + reg_features).copy()
    X = reg_df[reg_features]
    y = reg_df["protein_g"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    models = {
        "Linear Regression": Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("m", LinearRegression())]),
        "Random Forest Regressor": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting Regressor": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    }

    res = []
    best_rf = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        res.append({
            "experiment": exp_name,
            "model": name,
            "MAE": round(mean_absolute_error(y_te, pred), 4),
            "RMSE": round(mean_squared_error(y_te, pred) ** 0.5, 4),
            "R2": round(r2_score(y_te, pred), 4)
        })
        if name == "Random Forest Regressor":
            best_rf = (m, X_te, y_te, pred)

    return pd.DataFrame(res), best_rf

def run_classification_experiments(dataset, exp_name):
    cls_df = dataset.dropna(subset=["quality_label"] + cls_features).copy()
    X = cls_df[cls_features]
    y = cls_df["quality_label"].astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        "Random Forest Classifier": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting Classifier": Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GradientBoostingClassifier(random_state=RANDOM_STATE))])
    }

    res = []
    best_gbc = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
        res.append({
            "experiment": exp_name,
            "model": name,
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1, 4)
        })
        if name == "Gradient Boosting Classifier":
            best_gbc = (m, X_te, y_te, pred)

    return pd.DataFrame(res), best_gbc

baseline_df = final_dataset[final_dataset["source"] != "Indian Food Composition Data"]
expanded_df = final_dataset

reg_baseline, _ = run_regression_experiments(baseline_df, "Baseline (General Data Only)")
reg_expanded, (best_reg_model, X_te_reg, y_te_reg, pred_reg) = run_regression_experiments(expanded_df, "Expanded (General + Indian Data)")
reg_comparison = pd.concat([reg_baseline, reg_expanded], ignore_index=True)
reg_comparison.to_csv("results/regression_results.csv", index=False)

cls_baseline, _ = run_classification_experiments(baseline_df, "Baseline (General Data Only)")
cls_expanded, (best_cls_model, X_te_cls, y_te_cls, pred_cls) = run_classification_experiments(expanded_df, "Expanded (General + Indian Data)")
cls_comparison = pd.concat([cls_baseline, cls_expanded], ignore_index=True)
cls_comparison.to_csv("results/classification_results.csv", index=False)

# Save Trained Models to Disk (.joblib)
joblib.dump(best_reg_model, "models/protein_regressor.joblib")
joblib.dump(best_cls_model, "models/quality_classifier.joblib")
print("Saved models to models/protein_regressor.joblib & models/quality_classifier.joblib")

# Save Regression Scatter Plot Figure
plt.figure(figsize=(6, 5))
plt.scatter(y_te_reg, pred_reg, alpha=0.4, color="#2b5c8f")
min_val, max_val = min(y_te_reg.min(), pred_reg.min()), max(y_te_reg.max(), pred_reg.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")
plt.xlabel("Actual Protein (g/100g)")
plt.ylabel("Predicted Protein (g/100g)")
plt.title("Actual vs Predicted Protein — Random Forest")
plt.tight_layout()
plt.savefig("results/figures/regression_actual_vs_predicted.png")
plt.close()

# Save Classification Confusion Matrix Figure
cm = confusion_matrix(y_te_cls, pred_cls, labels=["Low", "Medium", "High"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Gradient Boosting")
plt.tight_layout()
plt.savefig("results/figures/classification_confusion_matrix.png")
plt.close()

print("\n--- Regression Comparison (Target: protein_g) ---")
print(reg_comparison.to_string(index=False))

print("\n--- Classification Comparison (Target: quality_label) ---")
print(cls_comparison.to_string(index=False))

# ==============================================================================
# SECTION 12: Upgraded Recommendation Engine & Scaler Persistence
# ==============================================================================
rec_features = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]
rec_df = final_dataset.dropna(subset=rec_features).reset_index(drop=True)

rec_scaler = StandardScaler()
rec_matrix = rec_scaler.fit_transform(rec_df[rec_features])

# Save Recommender Scaler
joblib.dump(rec_scaler, "models/recommender_scaler.joblib")
print("Saved recommender scaler to models/recommender_scaler.joblib")

def recommend_foods(
    target,
    top_k=10,
    quality=None,
    collection_location=None,
    max_calories=None,
    min_protein=None,
    min_fiber=None,
    max_sugar=None,
    max_sodium=None,
    max_fat=None,
    target_basis="per_100g"
):
    target_row = pd.DataFrame([target]).reindex(columns=rec_features, fill_value=0)
    target_scaled = rec_scaler.transform(target_row)
    scores = cosine_similarity(target_scaled, rec_matrix)[0]

    result = rec_df.copy()
    result["similarity"] = scores.round(4)

    filtered = result.copy()
    if quality:
        filtered = filtered[filtered["quality_label"].astype(str).str.lower() == quality.lower()]
    if collection_location:
        filtered = filtered[filtered["collection_location"].astype(str).str.lower() == collection_location.lower()]
    if max_calories is not None:
        filtered = filtered[filtered["calories_kcal"] <= max_calories]
    if min_protein is not None:
        filtered = filtered[filtered["protein_g"] >= min_protein]
    if min_fiber is not None:
        filtered = filtered[filtered["fiber_g"] >= min_fiber]
    if max_sugar is not None:
        filtered = filtered[filtered["sugar_g"] <= max_sugar]
    if max_sodium is not None:
        filtered = filtered[filtered["sodium_mg"] <= max_sodium]
    if max_fat is not None:
        filtered = filtered[filtered["fat_g"] <= max_fat]

    fallback_used = False
    if filtered.empty:
        fallback_used = True
        filtered = result.copy()
        if quality:
            filtered = filtered[filtered["quality_label"].astype(str).str.lower() == quality.lower()]

    output = filtered.sort_values("similarity", ascending=False).head(top_k).copy()

    explanations = []
    for _, row in output.iterrows():
        exp = f"Similarity: {row['similarity']:.2f} | {row['calories_kcal']} kcal, {row['protein_g']}g protein. "
        if fallback_used:
            exp += "Closest match (no exact match satisfied all constraints)."
        else:
            exp += "Satisfies specified target constraints."
        explanations.append(exp)

    output["why_recommended"] = explanations
    cols = ["food_name", "collection_location", *rec_features, "quality_label", "similarity", "why_recommended"]
    return output[[c for c in cols if c in output.columns]]

def recommend_meals(
    target,
    top_k=5,
    meal_type=None,
    diet_type=None,
    region=None
):
    if not os.path.exists("data/processed/indian_meals_clean.csv"):
        print("Meal dataset not found.")
        return pd.DataFrame()

    m_df = pd.read_csv("data/processed/indian_meals_clean.csv")
    m_features = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "sugar_g", "sodium_mg"]

    m_scaler = StandardScaler()
    m_matrix = m_scaler.fit_transform(m_df[m_features])

    t_row = pd.DataFrame([target]).reindex(columns=m_features, fill_value=0)
    t_scaled = m_scaler.transform(t_row)

    scores = cosine_similarity(t_scaled, m_matrix)[0]
    m_df["similarity"] = scores.round(4)

    filtered = m_df.copy()
    if meal_type:
        filtered = filtered[filtered["meal_type"].str.lower() == meal_type.lower()]
    if diet_type:
        filtered = filtered[filtered["diet_type"].str.lower() == diet_type.lower()]
    if region:
        filtered = filtered[filtered["region"].str.lower() == region.lower()]

    if filtered.empty:
        filtered = m_df.copy()

    res = filtered.sort_values("similarity", ascending=False).head(top_k).copy()
    exps = []
    for _, r in res.iterrows():
        exps.append(f"Match score: {r['similarity']:.2f} | {r['calories_kcal']} kcal, {r['protein_g']}g protein, {r['fiber_g']}g fiber.")
    res["why_recommended"] = exps

    cols = ["meal_name", "meal_type", "region", "diet_type", "ingredients", *m_features, "similarity", "why_recommended"]
    return res[cols]

# Run example recommendations & export
test_target = {
    "calories_kcal": 300,
    "protein_g": 20,
    "fat_g": 10,
    "carbohydrate_g": 30,
    "fiber_g": 8,
    "sugar_g": 5,
    "sodium_mg": 300
}

rec_sample = recommend_foods(test_target, top_k=5, min_protein=10, max_calories=400)
rec_sample.to_csv("results/example_recommendations.csv", index=False)

print("\n--- Example Food Recommendations ---")
print(rec_sample[["food_name", "calories_kcal", "protein_g", "similarity", "why_recommended"]].to_string(index=False))

if os.path.exists("data/processed/indian_meals_clean.csv"):
    meal_sample = recommend_meals(test_target, top_k=5, diet_type="Vegetarian")
    print("\n--- Example Indian Meal Recommendations ---")
    print(meal_sample[["meal_name", "meal_type", "region", "calories_kcal", "protein_g", "similarity"]].to_string(index=False))

print("\nPipeline Execution Complete! All models, figures, and CSV datasets saved successfully!")
