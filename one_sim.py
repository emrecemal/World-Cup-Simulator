import random

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
        
        
        
if __name__ == "__main__":
    team1 = "Bayern Munich"
    team2 = "Real Madrid"
    elos = {
        "Bayern Munich": 2387,
        "Real Madrid": 2281
    }
    matches_dict = {
        "M1": (team1, team2)
    }
    w, _, d = simulate_knockout_round(matches_dict, elos, n_sim=100000)
    
    print(w)
    print(d)