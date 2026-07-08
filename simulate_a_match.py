import math
import numpy as np
import pandas as pd
from collections import Counter

def calibrate_poisson_params(elo_home, elo_away, base_total_goals, home_advantage=50):
    """
    Calibrates expected goals (lambda for home, mu for away) based on Elo ratings.
    Includes an optional home_advantage boost added directly to the home team's Elo.
    """
    # Adjust home Elo for home-field advantage
    adjusted_elo_home = elo_home + home_advantage
    
    # Calculate xG ratio based on Elo difference
    xg_ratio = 10 ** ((adjusted_elo_home - elo_away) / 400)
    
    # Distribute the baseline goals according to the ratio
    mu = base_total_goals / (xg_ratio + 1)  # Away team xG
    lambda_ = base_total_goals - mu         # Home team xG
    
    return lambda_, mu

def dixon_coles_adjustment(x, y, lambda_, mu, rho):
    """
    Returns the Dixon-Coles multiplier tau(x, y) to correct low-scoring match dependencies.
    """
    if x == 0 and y == 0:
        return 1 - (lambda_ * mu * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_ * rho)
    elif x == 1 and y == 0:
        return 1 + (mu * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0

def generate_probability_matrix(lambda_, mu, max_goals=10, rho=-0.15):
    """
    Generates a 2D grid of exact score probabilities adjusted by Dixon-Coles.
    """
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            # Independent Poisson probability
            p_x = (math.exp(-lambda_) * (lambda_ ** x)) / math.factorial(x)
            p_y = (math.exp(-mu) * (mu ** y)) / math.factorial(y)
            p_independent = p_x * p_y
            
            # Apply Dixon-Coles adjustment
            tau = dixon_coles_adjustment(x, y, lambda_, mu, rho)
            prob_matrix[x, y] = p_independent * tau
            
    # Normalize matrix to ensure it sums perfectly to 1.0
    prob_matrix /= prob_matrix.sum()
    return prob_matrix

def run_monte_carlo(prob_matrix, num_simulations=10_000):
    """
    Simulates the match outcomes N times based on the probability matrix.
    """
    max_goals = prob_matrix.shape[0] - 1
    
    # Flatten the matrix and create an array of matching score labels
    flattened_probs = prob_matrix.flatten()
    score_labels = []
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            score_labels.append(f"{x}-{y}")
            
    # Monte Carlo sampling: Pick 10,000 scores based on our calculated probabilities
    simulated_scores = np.random.choice(score_labels, size=num_simulations, p=flattened_probs)
    
    return simulated_scores

def calculate_expected_points(pred_x, pred_y, prob_matrix):
    """
    Calculates the Expected Points (EV) of a specific score prediction,
    given the actual mathematical probability distribution.
    """
    expected_points = 0.0
    max_goals = prob_matrix.shape[0] - 1
    
    for actual_x in range(max_goals + 1):
        for actual_y in range(max_goals + 1):
            prob = prob_matrix[actual_x, actual_y]
            if prob <= 0:
                continue
                
            pts = 0
            # 1. Winner Selection (3 pts)
            if np.sign(pred_x - pred_y) == np.sign(actual_x - actual_y):
                pts += 3
            
            # 2. Exact Score (2 pts)
            if pred_x == actual_x and pred_y == actual_y:
                pts += 2
                
            # 3. Goals of one team (1 pt per correct team)
            if pred_x == actual_x:
                pts += 1
            if pred_y == actual_y:
                pts += 1
                
            # 4. Goal Difference (1 pt)
            if (pred_x - pred_y) == (actual_x - actual_y):
                pts += 1
                
            expected_points += prob * pts
            
    return expected_points

def optimize_predictions_for_tournament(prob_matrix):
    """
    Evaluates all reasonable scoreline predictions to find the ones that yield
    the highest Expected Points based on the tournament's specific scoring rules.
    """
    max_goals = 8
    ev_results = []
    
    for pred_x in range(max_goals + 1):
        for pred_y in range(max_goals + 1):
            ev = calculate_expected_points(pred_x, pred_y, prob_matrix)
            ev_results.append((f"{pred_x}-{pred_y}", ev))
            
    # Sort by highest Expected Value (EV)
    ev_results.sort(key=lambda x: x[1], reverse=True)
    return ev_results[:5]

def predict_match(csv_path, home_team, away_team):
    # Load Elo ratings
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df['Team'] = df['Team'].str.strip()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Extract Elo values
    try:
        elo_home = df.loc[df['Team'] == home_team, 'Elo'].values[0]
        elo_away = df.loc[df['Team'] == away_team, 'Elo'].values[0]
    except IndexError:
        print(f"Error: Could not find '{home_team}' or '{away_team}' in the CSV file.")
        return

    # 1. Host Nation Advantage (USA, Mexico, Canada)
    home_advantage = 0
    # FIX: Corrected the logical evaluation so it only applies to actual hosts
    if home_team in ["USA", "Mexico", "Canada"] or away_team in ["USA", "Mexico", "Canada"]:
        home_advantage = 50

    # ==========================================
    # ROUND OF 16 PARAMETERS APPLIED HERE
    # ==========================================
    # 1. Dynamic Base Goals: Adds extra goals to the baseline if there is a heavy mismatch 
    # (Accounts for underdogs opening up late to chase the game)
    elo_diff = abs((elo_home + home_advantage) - elo_away)
    mismatch_boost = (elo_diff / 2000) 
    
    R16_BASE_GOALS = 2.30 + mismatch_boost
    R8_BASE_GOALS = 2.20 + mismatch_boost  # Quarterfinals
    BASE_GOALS = R8_BASE_GOALS  # Default to Quarterfinals unless specified otherwise
    
    # 2. Relaxed Rho: Dropping from -0.16 to -0.10. 
    # This stops the model from hiding behind 1-1 draws and encourages 2-1 predictions.
    R16_RHO = -0.10
    R8_RHO = -0.15  # Quarterfinals
    RHO = R8_RHO  # Default to Quarterfinals unless specified otherwise 
    
    lambda_, mu = calibrate_poisson_params(elo_home, elo_away, base_total_goals=BASE_GOALS, home_advantage=home_advantage)
    
    # 2. Build the adjusted probability distribution grid using knockout Rho
    prob_matrix = generate_probability_matrix(lambda_, mu, max_goals=8, rho=RHO)
    
    # 3. Execute 10,000 Monte Carlo Simulation runs
    simulations = run_monte_carlo(prob_matrix, num_simulations=10000)
    
    # 4. Calculate Expected Value (EV) for Tournament Points
    top_ev_predictions = optimize_predictions_for_tournament(prob_matrix)
    
    # Count frequencies
    score_counts = Counter(simulations)
    top_5 = score_counts.most_common(5)
    
    # Output results
    print(f"\n=== ROUND OF 16 MATCH SIMULATION REPORT ===")
    print(f"Fixture: {home_team} vs. {away_team}")
    print(f"Elo Ratings: {home_team} ({elo_home}) | {away_team} ({elo_away})")
    print(f"Calibrated Expected Goals (xG): {home_team}: {lambda_:.2f} | {away_team}: {mu:.2f}")
    print(f"Model Parameters: Base Goals = {BASE_GOALS:.2f} (Dynamic), Dixon-Coles Rho = {RHO}")
    print("-" * 45)
    print("TOP 5 MOST PROBABLE EXACT SCORES (Regular Time - 90 Mins):")
    
    for rank, (score, count) in enumerate(top_5, 1):
        percentage = (count / 10000) * 100
        print(f"{rank}. Score: {score} ({percentage:.2f}% chance)")

    print("-" * 45)
    print("🏆 TOURNAMENT OPTIMIZED PICKS (Highest Expected Points) 🏆")
    print("Strategy: These picks act as an umbrella, maximizing partial points across all likely outcomes.")
    
    for rank, (score, ev) in enumerate(top_ev_predictions, 1):
        print(f"{rank}. Predict {score} (Expected Yield: {ev:.3f} points)")

if __name__ == "__main__":
    # Ensure you have 'elo.csv' in your working directory formatted as "Team,Elo"
    predict_match("elo.csv", "Argentina", "Switzerland")  # Example match for testing