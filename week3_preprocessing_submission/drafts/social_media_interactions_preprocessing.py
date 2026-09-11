import pandas as pd
import numpy as np


# ============================================================
# SOCIAL MEDIA INTERACTIONS DATA PREPROCESSING
# ============================================================

# 1. LOAD DATASET

interactions = pd.read_csv(
    "social_media_interactions_contaminated.csv"
)


# 2. INITIAL INSPECTION

print("===== INITIAL INSPECTION =====")

print("\nFirst 5 rows:")
print(interactions.head())

print("\nDataset shape:")
print(interactions.shape)

print("\nData types:")
print(interactions.dtypes)


# 3. CHECK MISSING VALUES

print("\n===== MISSING VALUES =====")

print(interactions.isnull().sum())


# 4. CHECK DUPLICATES

print("\n===== DUPLICATES =====")

print(
    "Number of duplicate rows:",
    interactions.duplicated().sum()
)

print("\nSample duplicate rows:")

print(
    interactions[
        interactions.duplicated(keep=False)
    ].head(20)
)


# 5. REMOVE DUPLICATES

interactions_cleaned = (
    interactions.drop_duplicates().copy()
)

print("\n===== AFTER REMOVING DUPLICATES =====")

print("New shape:")
print(interactions_cleaned.shape)

print(
    "Remaining duplicates:",
    interactions_cleaned.duplicated().sum()
)


# 6. CLEAN INTERACTION DATE

interactions_cleaned["InteractionDate"] = pd.to_datetime(
    interactions_cleaned["InteractionDate"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)


# 7. HANDLE MISSING PLATFORM VALUES

interactions_cleaned["Platform"] = (
    interactions_cleaned["Platform"]
    .fillna("Unknown")
)


# 8. HANDLE MISSING SENTIMENT VALUES

interactions_cleaned["Sentiment"] = (
    interactions_cleaned["Sentiment"]
    .fillna("Unknown")
)

interactions_cleaned["Sentiment"] = (
    interactions_cleaned["Sentiment"]
    .replace("None", "Unknown")
)


# 9. FINAL VALIDATION

print("\n===== FINAL VALIDATION =====")

print("\nFinal shape:")
print(interactions_cleaned.shape)

print("\nFinal data types:")
print(interactions_cleaned.dtypes)

print("\nFinal missing values:")
print(interactions_cleaned.isnull().sum())

print("\nFinal duplicate count:")
print(interactions_cleaned.duplicated().sum())

print("\nCleaned data preview:")
print(interactions_cleaned.head())


# 10. SAVE CLEANED DATASET

interactions_cleaned.to_csv(
    "social_media_interactions_preprocessed.csv",
    index=False,
    date_format="%Y-%m-%d"
)

print(
    "\nPreprocessed dataset saved successfully!"
)