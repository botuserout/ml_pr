import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from food_nutrition_ml_project_multisource_complete import (
    recommend_foods,
    recommend_meals,
    match_recipe_ingredient
)

def test_recommendation_returns_k():
    target = {
        "calories_kcal": 300,
        "protein_g": 20,
        "fat_g": 10,
        "carbohydrate_g": 30,
        "fiber_g": 8,
        "sugar_g": 5,
        "sodium_mg": 300
    }
    recs = recommend_foods(target, top_k=5)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= 5
    assert "similarity" in recs.columns
    assert "why_recommended" in recs.columns

def test_quality_labels_are_valid():
    df_path = "data/processed/food_nutrition_cleaned.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        if "quality_label" in df.columns:
            valid_labels = {"Low", "Medium", "High", np.nan}
            unique_labels = set(df["quality_label"].unique())
            assert unique_labels.issubset(valid_labels)

def test_ingredient_matching():
    food_data = pd.DataFrame([
        {"food_id": "IND_001", "food_name": "Paneer"},
        {"food_id": "IND_002", "food_name": "Butter Chicken"},
        {"food_id": "IND_003", "food_name": "Moong Dal"}
    ])
    f_id, conf, score = match_recipe_ingredient("Paneer", food_data)
    assert f_id == "IND_001"
    assert conf == "exact"
    assert score == 1.0

def test_constraint_filtering():
    target = {
        "calories_kcal": 300, "protein_g": 25, "fat_g": 5,
        "carbohydrate_g": 20, "fiber_g": 10, "sugar_g": 2, "sodium_mg": 200
    }
    recs = recommend_foods(target, top_k=3, max_calories=500, min_protein=5)
    assert not recs.empty
    if "why_recommended" in recs.columns:
        assert isinstance(recs.iloc[0]["why_recommended"], str)

def test_indian_meals_dataset_structure():
    meals_path = "data/processed/indian_meals_clean.csv"
    if os.path.exists(meals_path):
        m_df = pd.read_csv(meals_path)
        req_cols = ["meal_id", "meal_name", "calories_kcal", "protein_g", "ingredients"]
        for col in req_cols:
            assert col in m_df.columns
