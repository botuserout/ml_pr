# Antigravity AI Agent — Complete Implementation Brief

## Project
**A Machine Learning-Based System for Food Nutrient Prediction, Nutritional Quality Classification, and Food Recommendation**

## Main Goal
Upgrade the existing project so it has a stronger **Indian-food focus**, a reproducible **multi-source data-preparation pipeline**, and an improved **nutrition-aware recommendation system**.

> **Important decision:** Do NOT build the project around extracting data from the uploaded ICMR-NIN IFCT 2017 PDF. Do not blindly scrape the 584-page PDF or manually transcribe its tables. Treat it only as background/reference material. Do not invent values from it.

---

## 1. Role

Act as a **senior data engineer + machine-learning engineer + research-project developer**.

This is an already partially implemented student research project.

**Do not rewrite the whole project from scratch.** First inspect the repository and extend the working implementation.

Preserve existing:
- datasets
- preprocessing
- EDA
- ML models
- evaluation
- quality-score logic
- recommendation functions
- notebook structure
- working UI/code

Only refactor when there is a clear technical reason.

---

## 2. Existing Project Objective

The project contains three connected tasks:

### A. Nutrient Prediction
Predict selected food nutrients using supervised ML.

Primary target:
- `protein_g`

Optional secondary targets:
- `calories_kcal`
- `fiber_g`

Evaluation:
- MAE
- RMSE
- R²

### B. Nutritional Quality Classification
Classify foods into:
- Low
- Medium
- High

The scoring/labeling rules must remain transparent and reproducible. Do not present the labels as medical truth.

Evaluation:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### C. Food Recommendation
Use content-based nutritional similarity to recommend foods matching a target profile.

Existing fields include variants of:
```text
food_name
brand
collection_location
calories_kcal
protein_g
fat_g
carbohydrate_g
fiber_g
sugar_g
sodium_mg
quality_label
quality_score
similarity
```

Inspect the actual schema before changing anything.

---

# 3. First Task — Audit the Existing Repository

Before editing anything, inspect:

- notebooks
- Python scripts
- CSV/Excel/JSON datasets
- preprocessing
- feature engineering
- ML training
- train/test split
- evaluation
- recommendation functions
- quality score
- README
- requirements
- existing UI

Search specifically for:

```text
recommend_foods
interactive_recommender
quality_score
quality_label
preprocessing
feature engineering
model training
model evaluation
data loading
```

Create an internal implementation map.

Do not duplicate functionality that already exists.

---

# 4. New Project Direction

Upgrade the system to:

```text
General / International Food Data
             +
Indian Food Data
             +
Indian Recipe / Meal Data
             ↓
       Data Integration
             ↓
       Data Preparation
             ↓
      Feature Engineering
             ↓
 ┌───────────┼────────────┐
 ↓           ↓            ↓
Regression  Classification  Recommendation
 ↓           ↓            ↓
Nutrient    Quality        Food / Meal
Prediction  Classification Suggestions
```

The Indian component must be reproducible and source-traceable.

---

# 5. Data Architecture

Prefer:

```text
data/
├── raw/
│   ├── indian_foods/
│   ├── indian_recipes/
│   └── external_sources/
├── interim/
│   └── normalized/
└── processed/
    ├── indian_foods_clean.csv
    ├── indian_meals_clean.csv
    └── unified_food_dataset.csv
```

Adapt this to the existing repository rather than creating unnecessary duplicate structures.

---

# 6. Data Source Strategy

Use legitimate, accessible, reproducible sources.

## Priority 1 — Indian institutional sources

Where structured or reliably extractable data is actually available, consider:

- ICMR
- ICMR-NIN
- Government of India
- FSSAI
- other reputable public Indian nutrition resources

## Priority 2 — established food databases

Consider:

- Open Food Facts
- USDA FoodData Central
- other reputable public food-composition databases

## Priority 3 — recipe datasets

For recipe/meal structure, use a publicly accessible dataset containing, where possible:

- recipe name
- ingredients
- quantities
- cuisine/region
- meal/course type
- diet type

Kaggle/GitHub datasets may be used when their usage rights permit it.

### Critical provenance rule

Never claim data came from FSSAI/ICMR-NIN unless the actual records came from that source.

Retain:

```text
source_name
source_url
source_type
collection_method
retrieved_date
```

for imported records where practical.

---

# 7. IFCT 2017 PDF — Do Not Use as an Automated Main Dataset

The uploaded IFCT 2017 document is an official Indian Food Composition Tables publication. Its preface says it provides nutritional information for **528 key foods and 151 discrete food components**, with data from regional composite samples across six geographical regions.

However, do not build the automated pipeline by blindly extracting all PDF tables.

Avoid:
- broken table extraction
- repeated headers
- merged cells
- continuation-table errors
- OCR errors
- incorrect column alignment
- footnotes becoming records
- missing values being converted to zero
- nutrient values being assigned to the wrong food

If a structured public source derived from the same material is legitimately available and its provenance/usage permits it, evaluate it separately. Otherwise leave IFCT out of the automated ingestion pipeline.

---

# 8. Indian Food Dataset

Create:

```text
indian_foods_clean.csv
```

Preferred schema:

```text
food_id
food_name
normalized_food_name
regional_name
food_group
region
diet_type

calories_kcal
protein_g
fat_g
carbohydrate_g
fiber_g
sugar_g
sodium_mg

source_name
source_url
source_type
serving_basis
retrieved_date
```

Prefer **per 100 g** when the source explicitly provides it.

If the source is per serving:
- preserve original serving size
- convert only when scientifically defensible
- never silently assume one serving = 100 g

---

# 9. Representative Indian Foods

Build a representative dataset.

Include categories such as:

### Grains
- rice
- wheat
- atta
- ragi
- jowar
- bajra
- maize
- oats

### Pulses / legumes
- moong dal
- masoor dal
- toor dal
- chana dal
- urad dal
- rajma
- chickpeas
- black chana
- soybean

### Dairy
- milk
- curd
- yogurt
- paneer
- buttermilk

### Vegetables
- potato
- tomato
- onion
- spinach
- cauliflower
- carrot
- peas
- okra
- brinjal
- bottle gourd
- cabbage
- beans

### Fruits
- banana
- apple
- mango
- guava
- papaya
- orange
- pomegranate

### Nuts/seeds
- peanuts
- almonds
- cashews
- sesame
- flaxseed
- sunflower seeds
- makhana

### Common ingredients
- mustard oil
- groundnut oil
- coconut
- ghee
- jaggery
- common spices when reliable nutrition data exists

Also include representative packaged Indian foods where reliable nutrition-label data is available.

---

# 10. Indian Meal / Recipe Dataset

Create:

```text
indian_meals_clean.csv
```

Preferred schema:

```text
meal_id
meal_name
meal_type
region
cuisine
diet_type
ingredients
serving_size_g
source_name
source_url
source_type
retrieved_date
```

Meal types:

```text
Breakfast
Lunch
Dinner
Snack
Beverage
Dessert
```

Possible regions/cuisines:

```text
North Indian
South Indian
West Indian
East Indian
Northeast Indian
Gujarati
Punjabi
Maharashtrian
Bengali
Rajasthani
Pan-Indian
```

Only assign a region when the source supports it.

---

# 11. Food Data vs Meal Data

Keep these separate.

Food-level examples:

```text
Rice
Dal
Paneer
Curd
```

Meal-level examples:

```text
Dal Rice
Khichdi
Poha
Idli + Sambar
Thepla + Curd
Rajma Rice
Dal + Roti + Sabzi
```

---

# 12. Meal Nutrition Calculation

When ingredient quantities are available, calculate meal nutrition from the food-level dataset.

For every nutrient:

```text
meal_nutrient =
SUM((ingredient_weight_g / 100) * nutrient_per_100g)
```

Example:

```text
Dal Rice
Rice = 100 g
Dal  = 60 g
Oil  = 5 g
```

Calculate the contribution of each ingredient and sum them.

Generate:

```text
calories_kcal
protein_g
fat_g
carbohydrate_g
fiber_g
sugar_g
sodium_mg
```

If an ingredient cannot be matched confidently:
- do not guess
- mark it unmatched
- report it
- exclude/flag the meal depending on the calculation mode

Generate an ingredient-matching report.

---

# 13. Ingredient Normalization

Create:

```text
raw_ingredient_name
normalized_ingredient_name
matched_food_id
match_confidence
```

Confidence:

```text
exact
high
medium
low
unmatched
```

Automatically use only exact/high-confidence mappings.

Do not merge terms merely because they sound similar.

---

# 14. Data Cleaning

Implement a reproducible pipeline.

Handle:

### Missing values

Distinguish:

```text
missing
not reported
zero
trace
not applicable
```

Never convert every missing value to zero.

### Duplicates

Use combinations of:

```text
normalized_food_name
brand
source
nutrition values
```

to identify potential duplicates.

### Units

Standardize:

```text
energy → kcal
protein → g
fat → g
carbohydrate → g
fiber → g
sugar → g
sodium → mg
```

### Invalid values

Flag:
- negative nutrients
- impossible percentages
- extreme outliers
- suspicious decimal placement

Do not silently delete questionable records.

---

# 15. Responsible Data Acquisition

Before scraping any website:

1. Check for an official API.
2. Check for downloadable datasets.
3. Check robots.txt and terms where applicable.
4. Prefer APIs/downloads over scraping.
5. Use reasonable delays.
6. Cache downloads.
7. Save raw responses before transformation.
8. Never bypass CAPTCHA, authentication, rate limits, or access controls.

If a source blocks automation, stop using that acquisition method and find a legitimate alternative.

---

# 16. Reproducible Data Pipeline

Create functions/scripts similar to:

```text
collect_sources()
load_raw_data()
normalize_columns()
normalize_food_names()
standardize_units()
handle_missing_values()
remove_duplicates()
validate_nutrition_ranges()
match_recipe_ingredients()
calculate_meal_nutrition()
save_processed_data()
generate_data_quality_report()
```

Prefer:

```text
raw → interim → processed
```

Do not manually edit the final CSV.

---

# 17. Data Quality Report

Generate:

```text
source
raw_record_count
clean_record_count
duplicates_removed
missing_values
unmatched_ingredients
invalid_values
excluded_rows
final_record_count
```

Also calculate:

```text
missing percentage by column
duplicate percentage
source distribution
region distribution
meal-type distribution
```

Save as CSV/JSON/Markdown.

---

# 18. Extend Existing EDA

Include:

- nutrient distributions
- correlation matrix
- quality-label distribution
- Indian vs non-Indian comparisons where compatible
- regional distribution
- outlier analysis
- missing-value analysis

Compare Indian/non-Indian data only when units, definitions, and populations are sufficiently compatible.

Avoid unsupported health claims.

---

# 19. Quality Score

Inspect the existing quality-score code first.

Document:
- formula
- nutrients used
- normalization
- thresholds
- Low/Medium/High mapping

Do not replace it automatically.

Make it consistent for Indian foods.

Do not claim:

```text
"FSSAI defines this exact score"
```

unless an official source actually defines it.

---

# 20. ML Integration

First establish the existing baseline.

Then compare:

```text
Baseline:
Existing dataset

Expanded:
Existing + Indian data
```

Avoid data leakage.

Investigate duplicates across train/test.

## Regression

Possible models:

```text
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
```

Metrics:

```text
MAE
RMSE
R²
```

## Classification

Possible models:

```text
Logistic Regression
Decision Tree
Random Forest
```

Metrics:

```text
Accuracy
Precision
Recall
F1
Confusion Matrix
```

Use stratification for classification where appropriate.

---

# 21. Recommendation Engine

Preserve the existing interactive recommender.

It currently accepts:

```text
Calories
Protein
Fat
Carbohydrates
Fiber
Sugar
Sodium
Quality filter
Number of recommendations
```

Improve the backend without unnecessarily changing the interface.

Use:

```text
[
    calories_kcal,
    protein_g,
    fat_g,
    carbohydrate_g,
    fiber_g,
    sugar_g,
    sodium_mg
]
```

Standardize features before cosine similarity.

Recommended pipeline:

```text
StandardScaler
→ scaled food vectors
→ scaled target vector
→ cosine similarity
```

Do not calculate raw similarity across mixed kcal/g/mg features.

---

# 22. Recommendation Constraints

Support optional constraints:

```text
maximum calories
minimum protein
minimum fiber
maximum sugar
maximum sodium
maximum fat
```

If no item satisfies every constraint:

```text
No exact matches found.
Showing the closest nutritionally similar foods instead.
```

Never crash.

---

# 23. Indian Meal Recommendation

If meal nutrition is successfully calculated, add:

```text
recommend_meals(...)
```

Example target:

```text
Calories <= 500
Protein >= 20 g
Fiber >= 6 g
Sugar <= 10 g
```

Potential output:

```text
Besan Chilla + Curd
Moong Dal Khichdi + Curd
Rajma Rice
Dal + Roti + Vegetable
```

These are recommendations matching a computational target, not medical prescriptions.

---

# 24. Recommendation Explanation

Each recommendation should explain why it was selected.

Example:

```text
1. Besan Chilla + Curd

Similarity: 0.91
Calories: 410 kcal
Protein: 21 g
Fiber: 7 g

Why recommended:
- close to requested calorie range
- satisfies protein constraint
- high similarity to target nutrient profile
```

---

# 25. Target-Basis Problem

The current interface says the target is **per 100 g**, but values such as:

```text
1500 kcal
120 g protein
70 g fiber
```

are more naturally interpreted as daily targets.

Do not silently change the existing behavior.

Add a target-basis option:

```text
Per 100 g
Per serving
Per day
```

Keep the research prototype default as appropriate to the existing implementation, but validate unusual inputs and warn users.

---

# 26. Optional Daily Diet Mode

If time permits, add:

```text
Daily Diet Recommendation
```

Inputs:

```text
daily calorie target
protein target
fiber target
maximum sugar
maximum sodium
diet type
region preference
```

Output:

```text
Breakfast
Lunch
Snack
Dinner
```

This is optional. Prioritize data correctness and model evaluation over UI features.

---

# 27. FSSAI / Indian Guidance Layer

If FSSAI or Indian institutional guidance is incorporated:

Use it as a **reference/guidance layer**, not as a fabricated food-composition database.

Possible fields:

```text
guideline_source
guideline_topic
reference_nutrient
reference_value
unit
population_group
```

Only include values that can be traced to the actual source.

Do not imply FSSAI endorses the ML model.

---

# 28. Research Contribution

The upgraded research story should be:

> A multi-source machine-learning framework that integrates general food nutrition data with Indian food and meal information to support nutrient prediction, transparent nutritional-quality classification, and nutrition-aware food recommendation.

A useful experiment:

```text
Baseline:
general food dataset

vs

Expanded:
general + Indian food dataset
```

Compare recommendation relevance.

Possible metrics:

```text
Top-k relevance
constraint satisfaction rate
average nutrient distance
cosine similarity
```

This provides an empirical way to discuss whether Indian-context data improves recommendation usefulness.

---

# 29. Recommendation Evaluation

Do not claim collaborative filtering accuracy without user-rating data.

Use:

### Constraint satisfaction rate
Percentage of recommendations satisfying requested constraints.

### Nutrient distance
Distance between target nutrient vector and recommendation nutrient vector.

### Cosine similarity
Same similarity measure used by the recommender.

### Top-k inspection
Qualitative review of top recommendations.

Clearly document the absence of real user-rating/history data.

---

# 30. Tests

Add tests for:

```text
data loading
column validation
unit conversion
missing-value handling
quality score
similarity
recommendation
ingredient matching
meal nutrition calculation
```

Examples:

```text
test_meal_nutrition()
test_standardization()
test_missing_values()
test_recommendation_returns_k()
test_quality_labels_are_valid()
```

---

# 31. Error Handling

Handle:

- failed downloads
- empty datasets
- invalid numeric input
- invalid quality filter
- no recommendation
- unmatched ingredient
- malformed source data

Use clear messages.

---

# 32. Required Notebook Structure

Prefer:

```text
1. Project Title
2. Problem Statement
3. Objectives
4. Imports
5. Data Sources
6. Data Acquisition
7. Raw Dataset Inspection
8. Data Cleaning
9. Data Integration
10. Missing Value Analysis
11. Duplicate Analysis
12. Feature Engineering
13. Exploratory Data Analysis
14. Nutritional Quality Score
15. Regression Models
16. Regression Evaluation
17. Classification Models
18. Classification Evaluation
19. Indian Food Integration
20. Indian Meal Construction
21. Recommendation System
22. Recommendation Evaluation
23. Example Recommendations
24. Limitations
25. Future Work
26. Conclusion
```

---

# 33. Required Visualizations

At minimum:

1. Nutrient distributions
2. Correlation heatmap
3. Quality-label distribution
4. Model comparison
5. Confusion matrix
6. Actual vs predicted regression plot
7. Indian vs non-Indian comparison
8. Regional distribution
9. Recommendation results
10. Missing-data visualization

Use clear units and readable labels.

---

# 34. Output Files

Produce, where feasible:

```text
indian_foods_raw.csv
indian_foods_clean.csv
indian_meals_raw.csv
indian_meals_clean.csv
unified_food_dataset.csv
data_quality_report.csv
ingredient_matching_report.csv
```

If a requested file cannot be generated because reliable data is unavailable, document the reason.

Never fabricate rows to satisfy a record-count target.

---

# 35. README Update

Update the README with:

- project overview
- architecture
- data sources
- acquisition methods
- preprocessing
- feature engineering
- ML models
- recommendation system
- Indian-food integration
- limitations
- reproducibility
- ethical disclaimer

Include source attribution.

---

# 36. Research-Paper Support

After implementation, produce a research-ready summary containing actual values for:

### Dataset
```text
number of sources
raw records
cleaned records
Indian foods
Indian meals
```

### Preprocessing
```text
missing values
duplicates
unit normalization
ingredient matching
feature engineering
```

### ML
```text
best regression model
MAE
RMSE
R²

best classifier
accuracy
precision
recall
F1
```

### Recommendation
```text
similarity method
scaling method
constraint filtering
Top-k evaluation
```

### Indian contribution
Explain exactly how the Indian data affected the system.

**Never invent metrics or record counts.**

---

# 37. Reproducibility

A fresh user should be able to run the project.

Provide/update:

```text
requirements.txt
```

Document:

```text
Python version
packages
data acquisition
preprocessing
training
recommendation execution
```

Never hard-code API keys.

Use environment variables.

---

# 38. Data Versioning

Store metadata:

```text
dataset_version
collection_date
source_name
source_url
record_count
```

This makes the research experiment reproducible.

---

# 39. Research Integrity Rules

Never:

- fabricate nutrition values
- fabricate sources
- fabricate model metrics
- fabricate record counts
- claim FSSAI endorsement
- claim ICMR validation of the ML model
- claim medical accuracy
- hide preprocessing decisions
- silently delete inconvenient data
- copy restricted data without checking usage rights

If something cannot be verified:

```text
mark it unknown
```

and document the limitation.

---

# 40. Definition of Done

## Data
- [ ] Existing dataset preserved
- [ ] Indian food dataset added
- [ ] Indian recipe/meal dataset added where feasible
- [ ] Source provenance recorded
- [ ] Units standardized
- [ ] Missing values documented
- [ ] Duplicates handled
- [ ] Data-quality report generated

## ML
- [ ] Existing regression works
- [ ] Existing classification works
- [ ] Indian data integrated without leakage
- [ ] Metrics generated
- [ ] Baseline vs expanded comparison performed where possible

## Recommendation
- [ ] Existing recommender still works
- [ ] Scaling before similarity
- [ ] Nutritional constraints
- [ ] No-match fallback
- [ ] Indian foods recommendable
- [ ] Indian meals recommendable where nutrition is available
- [ ] Explanations shown

## Engineering
- [ ] Tests added
- [ ] Error handling
- [ ] README updated
- [ ] Requirements updated
- [ ] No secrets
- [ ] No unnecessary duplicated code

## Research
- [ ] Sources documented
- [ ] Methodology documented
- [ ] Results reproducible
- [ ] Limitations documented
- [ ] Indian-context contribution clearly stated

---

# 41. Final Instruction to Antigravity

Start by **auditing the existing repository**.

Then:

1. Identify the current data pipeline.
2. Identify the current ML pipeline.
3. Identify the current recommender.
4. Identify the best integration point for Indian food data.
5. Select legitimate accessible data sources.
6. Build raw → interim → processed ingestion.
7. Build the Indian food dataset.
8. Build the Indian meal dataset where reliable ingredient/quantity data exists.
9. Integrate with the existing project.
10. Extend EDA.
11. Re-run and compare models.
12. Upgrade recommendation logic.
13. Add tests.
14. Update documentation.
15. Produce a final implementation report with **actual** record counts, sources, and metrics.

## Non-negotiable rules

**Do not use the IFCT 2017 PDF as a blindly scraped dataset.**

**Do not fabricate missing nutritional values.**

**Do not fabricate research results.**

**Do not replace working components unnecessarily.**

**Prioritize data quality, reproducibility, source attribution, and research validity over dataset size.**

The final project should look like one coherent research prototype:
**multi-source food data → data preparation → ML → Indian food integration → nutrition-aware recommendation.**
