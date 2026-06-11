import math
import numpy as np
import pandas as pd
from collections import Counter

def calibrate_poisson_params(elo_home, elo_away, base_total_goals=2.7, home_advantage=50):
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

def dixon_coles_adjustment(x, y, lambda_, mu, rho=-0.12):
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

def generate_probability_matrix(lambda_, mu, max_goals=10, rho=-0.12):
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

def run_monte_carlo(prob_matrix, num_simulations=10000):
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

    # 1. Calibrate Goal Parameters (Using a 50 Elo point home advantage)
    if ("USA" or "Mexico" or "Canada" in home_team) or ("USA" or "Mexico" or "Canada" in away_team):
        home_advantage = 50
    else:
        home_advantage = 0
    lambda_, mu = calibrate_poisson_params(elo_home, elo_away, base_total_goals=2.7, home_advantage=home_advantage)
    
    # 2. Build the adjusted probability distribution grid
    prob_matrix = generate_probability_matrix(lambda_, mu, max_goals=8, rho=-0.12)
    
    # 3. Execute 10,000 Monte Carlo Simulation runs
    simulations = run_monte_carlo(prob_matrix, num_simulations=10000)
    
    # Count frequencies
    score_counts = Counter(simulations)
    top_5 = score_counts.most_common(5)
    
    # Output results
    print(f"\n=== MATCH SIMULATION REPORT ===")
    print(f"Fixture: {home_team} (Home) vs. {away_team} (Away)")
    print(f"Elo Ratings: {home_team} ({elo_home}) | {away_team} ({elo_away})")
    print(f"Calibrated Expected Goals (xG): {home_team}: {lambda_:.2f} | {away_team}: {mu:.2f}")
    print("-" * 45)
    print("TOP 5 MOST PROBABLE EXACT SCORES (Monte Carlo 10k runs):")
    
    for rank, (score, count) in enumerate(top_5, 1):
        percentage = (count / 10000) * 100
        print(f"{rank}. Score: {score} ({percentage:.2f}% chance)")



if __name__ == "__main__":
    # Use the CSV file path directly
    predict_match("elo.csv", "Ghana", "Panama")