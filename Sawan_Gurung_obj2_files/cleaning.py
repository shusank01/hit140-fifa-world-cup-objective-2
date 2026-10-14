
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent
data_folder = folder / "initial_2.2_data_sets"
output_folder = folder / "cleaned_data"
output_folder.mkdir(exist_ok=True)

# Load the datasets
age_data = pd.read_csv(data_folder / "avg_age.csv")
league_data = pd.read_csv(data_folder / "top5_league_players_by_country.csv")

# Clean average age data
age_data.columns = age_data.columns.str.strip()
age_data["team"] = age_data["team"].astype("string").str.strip()
age_data["squad_avg_age"] = pd.to_numeric(
    age_data["squad_avg_age"], errors="coerce"
)

age_data = age_data.dropna(subset=["team", "squad_avg_age"])
age_data = age_data.drop_duplicates(subset="team")
age_data = age_data[age_data["squad_avg_age"].between(15, 60)]
age_data = age_data.sort_values("team")

age_data.to_csv(output_folder / "avg_age_cleaned.csv", index=False)


# Clean league player data
league_data.columns = (
    league_data.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

league_data["country"] = league_data["country"].astype("string").str.strip()
league_data["code"] = league_data["code"].astype("string").str.strip()

player_columns = [
    "english_premier_league",
    "german_bundesliga",
    "spanish_la_liga",
    "french_ligue_1",
    "italian_serie_a"
]

number_columns = ["rank"] + player_columns + ["total_players"]

for column in number_columns:
    league_data[column] = pd.to_numeric(
        league_data[column], errors="coerce"
    )

league_data = league_data.dropna(
    subset=["country", "code"] + number_columns
)

league_data = league_data.drop_duplicates(subset="country")
league_data = league_data.drop_duplicates(subset="code")

valid_counts = (
    (league_data[player_columns] >= 0).all(axis=1)
    & (league_data[player_columns] % 1 == 0).all(axis=1)
)

league_data = league_data[valid_counts]

league_data = league_data[
    (league_data["rank"] >= 1)
    & (league_data["rank"] % 1 == 0)
    & (league_data["total_players"] >= 0)
    & (league_data["total_players"] % 1 == 0)
].copy()

league_data["total_players"] = league_data[player_columns].sum(axis=1)
league_data[number_columns] = league_data[number_columns].astype(int)

league_data = league_data.sort_values(["rank", "country"])

league_data.to_csv(
    output_folder / "top5_league_players_by_country_cleaned.csv",
    index=False
)

print("Data cleaning completed.")
print("Average age dataset:", age_data.shape)
print("League player dataset:", league_data.shape)
print("Cleaned files saved in:", output_folder)
