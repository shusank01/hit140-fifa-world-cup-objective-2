import math
import csv
import pandas as pd

data = pd.read_csv('data.csv')

# Drop rows with NaN values in the "Score" column and reset the index
df = data.dropna(subset=["Score"]).reset_index(drop=True)

print(df.shape)

# Separate country codes from team names
#Home looks like "Mexico mx", Away looks like "za South Africa"

df["team_a"] = df["Home"].str.rsplit(" ", n=1).str[0]   # splitting once from the right, keeping the left part
df["team_b"] = df["Away"].str.split(" ", n=1).str[1]    #splitting after first space, keeping right side
 
#    Uses an en dash (–) and regex to extract the goal pattern and splits them. 
#    so shootout games become draws. Scores include extra time.
goals = df["Score"].str.extract(r"(\d+)–(\d+)").astype(int)
df["goals_a"] = goals[0]
df["goals_b"] = goals[1]
 
# 4. Parse the date
df["date"] = pd.to_datetime(df["Date"])
 
# Building  new columns
df["goal_diff"] = df["goals_a"] - df["goals_b"]          # Team A minus Team B
df["knockout"] = (df["Round"] != "Group stage").astype(int) #var 1
df["match_id"] = range(1, len(df) + 1)
df = df.rename(columns={"Round": "round"})
 
#var 2
clean_set = df[["match_id", "date", "round", "team_a", "team_b",
            "goals_a", "goals_b", "goal_diff", "knockout"]].copy()

w22 = pd.read_csv("fifa_2022_stats.csv", header=1)             
q22 = set(w22["Squad"].str.split(" ", n=1).str[1])  # drop the country code at first

clean_set["qualified_2022_diff"] = clean_set["team_a"].isin(q22).astype(int) - clean_set["team_b"].isin(q22).astype(int)

#var 3
hosts = {"USA", "Mexico", "Canada"}
clean_set["hosting_difference"] = clean_set["team_a"].isin(hosts).astype(int) - clean_set["team_b"].isin(hosts).astype(int)

#var 4
age = pd.read_csv("avg_age.csv")  
m = age.set_index("team")["squad_avg_age"]
clean_set["avg_age_difference"] = clean_set["team_a"].map(m) - clean_set["team_b"].map(m)

#var 5

points = pd.read_csv("fifa_rank_var.csv")
m = points.set_index("team")["fifa_points"]
clean_set["fifa_point_difference"] = clean_set["team_a"].map(m) - clean_set["team_b"].map(m)

#var 6 & 7

clean = clean_set.sort_values(["date", "match_id"]).reset_index(drop=True) # latest date first so that we can ignore the recent match

INITIAL = 1 #base value as the starting point for averages (can't start with 0)
scored_total = {}
conceded_total = {}
games_played = {}

sc_a, sc_b, co_a, co_b = [], [], [], []

for i, row in clean.iterrows():
    team_a = row["team_a"]
    team_b = row["team_b"]

    # averages BEFORE this match, for each of the two teams
    for team, s_list, c_list in [(team_a, sc_a, co_a), (team_b, sc_b, co_b)]:
        played = games_played.get(team, 0)
        if played == 0:
            s_list.append(INITIAL)
            c_list.append(INITIAL)
        else:
            s_list.append(scored_total[team] / played)
            c_list.append(conceded_total[team] / played)

    # updating the totals with this match
    scored_total[team_a] = scored_total.get(team_a, 0) + row["goals_a"]
    conceded_total[team_a] = conceded_total.get(team_a, 0) + row["goals_b"]
    games_played[team_a] = games_played.get(team_a, 0) + 1

    scored_total[team_b] = scored_total.get(team_b, 0) + row["goals_b"]
    conceded_total[team_b] = conceded_total.get(team_b, 0) + row["goals_a"]
    games_played[team_b] = games_played.get(team_b, 0) + 1

clean["scored_avg_diff"] = pd.Series(sc_a) - pd.Series(sc_b) # 6
clean["conceded_avg_diff"] = pd.Series(co_a) - pd.Series(co_b) # 7 

# var 8

clean = clean.sort_values(["date", "match_id"]).reset_index(drop=True)

points_total = {}     
pts_a_list = []
pts_b_list = []

for i, row in clean.iterrows():
    team_a = row["team_a"]
    team_b = row["team_b"]

    #points neglecting recent match
    pts_a_list.append(points_total.get(team_a, 0))
    pts_b_list.append(points_total.get(team_b, 0))

    # adding this match's points
    if row["goals_a"] > row["goals_b"]:
        pa, pb = 3, 0
    elif row["goals_a"] < row["goals_b"]:
        pa, pb = 0, 3
    else:
        pa, pb = 1, 1
    points_total[team_a] = points_total.get(team_a, 0) + pa
    points_total[team_b] = points_total.get(team_b, 0) + pb

clean["points_before_diff"] = pd.Series(pts_a_list) - pd.Series(pts_b_list)

#saving to a file
final_df = clean_set.merge(
    clean[["match_id", "scored_avg_diff", "conceded_avg_diff",
           "points_before_diff"]],
    on="match_id",
    how="left"
)

final_df.to_csv("final_world_cup_dataset.csv", index=False)

#age data: https://www.rotowire.com/soccer/article/2026-fifa-world-cup-squad-ages-youngest-oldest-teams-ranked-116813
#rank: fifa official website
#2022 stats from fbref



