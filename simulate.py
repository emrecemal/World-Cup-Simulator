import numpy as np
import pandas as pd
import random

# -------------------------
# 1. LOAD DATA
# -------------------------
def load_elo(csv_path):
    df = pd.read_csv(csv_path, sep=",", quotechar='"')
    return dict(zip(df["Country"], df["Column_4"]))

def load_country_names(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["Country"], df["Full_Name"]))

# -------------------------
# 2. MATCH SIMULATION
# -------------------------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def simulate_match(team1, team2, elos, draw_prob=0.25):
    r1 = elos.get(team1, 1500)
    r2 = elos.get(team2, 1500)

    e1 = expected_score(r1, r2)
    rand = random.random()

    if rand < draw_prob:
        return "draw"
    elif rand < draw_prob + (1 - draw_prob) * e1:
        return team1
    else:
        return team2

# -------------------------
# 3. GROUP STAGE SIMULATION
# -------------------------
def simulate_group_once(group, elos):
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

    teams = list(points.items())
    random.shuffle(teams)
    sorted_teams = sorted(teams, key=lambda x: x[1], reverse=True)
    return [team for team, _ in sorted_teams]

def run_group_simulations(groups, elos, n_sim=10000):
    results = {}
    for g_idx, group in enumerate(groups):
        group_letter = chr(65 + g_idx)
        group_name = f"Group_{group_letter}"
        results[group_name] = {team: [0, 0, 0, 0] for team in group}

    for _ in range(n_sim):
        for g_idx, group in enumerate(groups):
            group_letter = chr(65 + g_idx)
            group_name = f"Group_{group_letter}"
            ranking = simulate_group_once(group, elos)

            for pos, team in enumerate(ranking):
                results[group_name][team][pos] += 1

    for group in results:
        for team in results[group]:
            results[group][team] = [count / n_sim for count in results[group][team]]

    return results

# -------------------------
# 4. KNOCKOUT LOGIC (FIFA 2026 BRACKET)
# -------------------------
def get_group_standings(group_results):
    standings = {}
    for group, data in group_results.items():
        letter = group.split("_")[1]
        team_stats = []
        for team, probs in data.items():
            exp_pos = sum((i+1) * p for i, p in enumerate(probs))
            team_stats.append((team, exp_pos))
        
        team_stats.sort(key=lambda x: x[1])
        standings[letter] = [t[0] for t in team_stats]
    return standings

def assign_third_place_teams(standings, elos):
    # Extract all 3rd place teams and sort by Elo
    thirds = [(grp, standings[grp][2]) for grp in standings]
    thirds.sort(key=lambda x: elos.get(x[1], 1500), reverse=True)
    advancing_thirds = thirds[:8]

    # FIFA constraints for mapping 3rd-place teams
    allowed = {
        '1E': {'A', 'B', 'C', 'D', 'F'},
        '1I': {'C', 'D', 'F', 'G', 'H'},
        '1A': {'C', 'E', 'F', 'H', 'I'},
        '1L': {'E', 'H', 'I', 'J', 'K'},
        '1D': {'B', 'E', 'F', 'I', 'J'},
        '1G': {'A', 'E', 'H', 'I', 'J'},
        '1B': {'E', 'F', 'G', 'I', 'J'},
        '1K': {'D', 'E', 'I', 'J', 'L'}
    }
    slots = ['1E', '1I', '1A', '1L', '1D', '1G', '1B', '1K']

    # Backtracking algorithm to find valid assignment
    def solve(idx, current_assigned):
        if idx == 8: return current_assigned
        t_grp, t_name = advancing_thirds[idx]
        for s in slots:
            if s not in current_assigned and t_grp in allowed[s]:
                res = solve(idx + 1, {**current_assigned, s: t_name})
                if res: return res
        return None

    assignment = solve(0, {})
    # Fallback just in case (though math dictates it shouldn't fail)
    if not assignment:
        assignment = {}
        unassigned = [t[1] for t in advancing_thirds]
        for s in slots: assignment[s] = unassigned.pop()
        
    return assignment

def create_round_of_32(standings, thirds_assigned):
    # Maps directly to Matches 73-88
    return {
        73: (standings['A'][1], standings['B'][1]),
        74: (standings['E'][0], thirds_assigned['1E']),
        75: (standings['F'][0], standings['C'][1]),
        76: (standings['C'][0], standings['F'][1]),
        77: (standings['I'][0], thirds_assigned['1I']),
        78: (standings['E'][1], standings['I'][1]),
        79: (standings['A'][0], thirds_assigned['1A']),
        80: (standings['L'][0], thirds_assigned['1L']),
        81: (standings['D'][0], thirds_assigned['1D']),
        82: (standings['G'][0], thirds_assigned['1G']),
        83: (standings['K'][1], standings['L'][1]),
        84: (standings['H'][0], standings['J'][1]),
        85: (standings['B'][0], thirds_assigned['1B']),
        86: (standings['J'][0], standings['H'][1]),
        87: (standings['K'][0], thirds_assigned['1K']),
        88: (standings['D'][1], standings['G'][1])
    }

def simulate_knockout_round(matches_dict, elos, n_sim=10000):
    winners = {}
    losers = {}
    match_data = []

    for m_id in sorted(matches_dict.keys()):
        t1, t2 = matches_dict[m_id]
        probs = {"team1": 0, "draw": 0, "team2": 0}
        
        for _ in range(n_sim):
            r = simulate_match(t1, t2, elos)
            if r == "draw":
                probs["draw"] += 1
            elif r == t1:
                probs["team1"] += 1
            else:
                probs["team2"] += 1

        for k in probs:
            probs[k] /= n_sim

        # Advance higher overall probability (W + half of draws)
        adv1 = probs["team1"] + (probs["draw"] / 2)
        adv2 = probs["team2"] + (probs["draw"] / 2)
        
        winner = t1 if adv1 >= adv2 else t2
        loser = t2 if winner == t1 else t1
        
        winners[m_id] = winner
        losers[m_id] = loser
        match_data.append((m_id, t1, t2, probs, winner))

    return winners, losers, match_data

def simulate_full_tournament(group_results, elos, n_sim=10000):
    standings = get_group_standings(group_results)
    thirds_assigned = assign_third_place_teams(standings, elos)

    r32_matches = create_round_of_32(standings, thirds_assigned)
    w_32, _, d_32 = simulate_knockout_round(r32_matches, elos, n_sim)

    # Round of 16 (Matches 89-96)
    r16_matches = {
        89: (w_32[74], w_32[77]), 90: (w_32[73], w_32[75]),
        91: (w_32[76], w_32[78]), 92: (w_32[79], w_32[80]),
        93: (w_32[83], w_32[84]), 94: (w_32[81], w_32[82]),
        95: (w_32[86], w_32[88]), 96: (w_32[85], w_32[87])
    }
    w_16, _, d_16 = simulate_knockout_round(r16_matches, elos, n_sim)

    # Quarterfinals (Matches 97-100)
    qf_matches = {
        97: (w_16[89], w_16[90]), 98: (w_16[93], w_16[94]),
        99: (w_16[91], w_16[92]), 100: (w_16[95], w_16[96])
    }
    w_qf, _, d_qf = simulate_knockout_round(qf_matches, elos, n_sim)

    # Semifinals (Matches 101-102)
    sf_matches = {
        101: (w_qf[97], w_qf[98]), 
        102: (w_qf[99], w_qf[100])
    }
    w_sf, l_sf, d_sf = simulate_knockout_round(sf_matches, elos, n_sim)

    # Third Place (Match 103) & Final (Match 104)
    finals_matches = {
        103: (l_sf[101], l_sf[102]),
        104: (w_sf[101], w_sf[102])
    }
    w_fin, _, d_fin = simulate_knockout_round(finals_matches, elos, n_sim)

    return {
        "Round of 32": d_32,
        "Round of 16": d_16,
        "Quarter-Finals": d_qf,
        "Semi-Finals": d_sf,
        "Third Place / Final": d_fin
    }, w_fin[104]

# -------------------------
# 5. MARKDOWN GENERATION
# -------------------------
def generate_final_markdown(group_results, knockout_data, champion, country_map, filename="README.md"):
    lines = ["# 🏆 2026 FIFA World Cup - Monte Carlo Simulation\n"]
    
    # Group Stages
    lines.append("## 📊 Group Stage Probabilities\n")
    for group, data in group_results.items():
        lines.append(f"### {group}")
        lines.append("| Team | 1st | 2nd | 3rd | 4th |")
        lines.append("|------|-----|-----|-----|-----|")

        team_stats = []
        for team, probs in data.items():
            exp_pos = sum((i+1) * p for i, p in enumerate(probs))
            team_stats.append((team, probs, exp_pos))

        team_stats.sort(key=lambda x: x[2])

        for team, probs, _ in team_stats:
            name = country_map.get(team, team)
            lines.append(f"| **{name}** | {probs[0]:.2%} | {probs[1]:.2%} | {probs[2]:.2%} | {probs[3]:.2%} |")
        lines.append("\n")

    # Knockout Stages
    lines.append("## ⚔️ Knockout Stage Results\n")
    order = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Third Place / Final"]

    for stage in order:
        lines.append(f"### {stage}\n")
        lines.append("| Match | Matchup | Team 1 Win | Draw (90m) | Team 2 Win | Advanced |")
        lines.append("|---|---------|------------|------------|------------|----------|")
        
        for m_id, t1, t2, probs, winner in knockout_data[stage]:
            name1 = country_map.get(t1, t1)
            name2 = country_map.get(t2, t2)
            winner_name = country_map.get(winner, winner)
            
            lines.append(
                f"| {m_id} | **{name1}** vs **{name2}** | {probs['team1']:>7.2%} | {probs['draw']:>10.2%} | {probs['team2']:>10.2%} | 🟢 **{winner_name}** |"
            )
        lines.append("\n")

    lines.append(f"## 🏆 Tournament Champion: **{country_map.get(champion, champion)}** 🏆\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Simulation complete! Results successfully written to '{filename}'.")

# -------------------------
# 6. EXECUTION
# -------------------------
if __name__ == "__main__":
    groups = [
        ["MX", "ZA", "KR", "CZ"], # Group A
        ["CA", "BA", "QA", "CH"], # Group B
        ["BR", "MA", "HT", "SQ"], # Group C
        ["US", "PY", "AU", "TR"], # Group D
        ["DE", "CW", "CI", "EC"], # Group E
        ["NL", "JP", "SE", "TN"], # Group F
        ["BE", "EG", "IR", "NZ"], # Group G
        ["ES", "CV", "SA", "UY"], # Group H
        ["FR", "SN", "IQ", "NO"], # Group I
        ["AR", "DZ", "AT", "JO"], # Group J
        ["PT", "CD", "UZ", "CO"], # Group K
        ["EN", "HR", "GH", "PA"]  # Group L
    ]

    print("Loading data...")
    elos = load_elo("2026_World_Cup_Elo_Ratings_Full_Names.csv")
    country_map = load_country_names("2026_World_Cup_Elo_Ratings_Country_Codes.csv")

    print("Running Group Stage Monte Carlo Simulations (10,000 runs)...")
    group_results = run_group_simulations(groups, elos, n_sim=10000)

    print("Running Knockout Stage Monte Carlo Simulations (10,000 runs per match)...")
    knockout_data, champion = simulate_full_tournament(group_results, elos, n_sim=10000)

    print("Generating Markdown Report...")
    generate_final_markdown(group_results, knockout_data, champion, country_map, "README.md")