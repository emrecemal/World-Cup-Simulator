import numpy as np
import pandas as pd
import random
from collections import defaultdict

# -------------------------
# LOAD ELO RATINGS
# -------------------------
def load_elo(csv_path):
    df = pd.read_csv(csv_path, sep=",", quotechar='"')
    return dict(zip(df["Country"], df["Column_4"]))  # "Column_4" has the ELO ratings

import numpy as np
import pandas as pd
import random
from collections import defaultdict

# -------------------------
# MATCH
# -------------------------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def simulate_match(team1, team2, elos, draw_prob=0.25):
    r1 = elos[team1]
    r2 = elos[team2]

    e1 = expected_score(r1, r2)

    rand = random.random()

    if rand < draw_prob:
        return "draw"
    elif rand < draw_prob + (1 - draw_prob) * e1:
        return team1
    else:
        return team2

# -------------------------
# GROUP
# -------------------------
def simulate_group(group, elos):
    points = {team: 0 for team in group}

    for i in range(len(group)):
        for j in range(i+1, len(group)):
            t1, t2 = group[i], group[j]
            result = simulate_match(t1, t2, elos)

            if result == "draw":
                points[t1] += 1
                points[t2] += 1
            elif result == t1:
                points[t1] += 3
            else:
                points[t2] += 3

    # random tie-break (important!)
    teams = list(points.items())
    random.shuffle(teams)

    sorted_teams = sorted(teams, key=lambda x: x[1], reverse=True)

    return [team for team, _ in sorted_teams]

# -------------------------
# MONTE CARLO (GROUP LEVEL)
# -------------------------
def run_group_simulations(groups, elos, n_sim=10000):
    # structure: group -> team -> position counts
    results = {}

    for g_idx, group in enumerate(groups):
        group_name = f"Group_{g_idx+1}"
        results[group_name] = {
            team: [0, 0, 0, 0] for team in group
        }

    for _ in range(n_sim):
        for g_idx, group in enumerate(groups):
            group_name = f"Group_{g_idx+1}"
            ranking = simulate_group(group, elos)

            for pos, team in enumerate(ranking):
                results[group_name][team][pos] += 1

    # normalize
    for group in results:
        for team in results[group]:
            results[group][team] = [
                count / n_sim for count in results[group][team]
            ]

    return results

# -------------------------
# PRETTY PRINT
# -------------------------
def print_results(results):
    for group, data in results.items():
        print(f"\n{group}")
        print("Team | 1st | 2nd | 3rd | 4th")
        print("-" * 40)

        for team, probs in data.items():
            print(f"{team} | {probs[0]:.2f} | {probs[1]:.2f} | {probs[2]:.2f} | {probs[3]:.2f}")


# -------------------------
# Groups
# -------------------------
groups = [
    ["MX", "ZA", "KR", "CZ"],
    ["CA", "BA", "QA", "CH"],
    ["BR", "MA", "HT", "SQ"],
    ["US", "PY", "AU", "TR"],
    ["DE", "CW", "CI", "EC"],
    ["NL", "JP", "SE", "TN"],
    ["BE", "EG", "IR", "NZ"],
    ["ES", "CV", "SA", "UY"],
    ["FR", "SN", "IQ", "NO"],
    ["AR", "DZ", "AT", "JO"],
    ["PT", "CD", "UZ", "CO"],
    ["EN", "HR", "GH", "PA"]
]

# -------------------------
# RUN
# -------------------------
elos = load_elo("2026_World_Cup_Elo_Ratings_Full_Names.csv")
results = run_group_simulations(groups, elos, n_sim=10000)
print_results(results)
