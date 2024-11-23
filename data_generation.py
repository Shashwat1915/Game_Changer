import random
import pandas as pd
def generate_team_stats():
    num_teams = int(input("Enter the number of teams: "))
    teams = []

    for _ in range(num_teams):
        team_name = input("Enter team name: ")
        teams.append(team_name)
    data = []

    for team in teams:
        wins = random.randint(0, 30)
        losses = random.randint(0, 30)
        draws = random.randint(0, 30)
        total_points = (wins * 3) + (draws * 1)
        
        data.append([team, wins, losses, draws, total_points])
    
    df = pd.DataFrame(data, columns=["Team Name", "Matches Won", "Matches Lost", "Matches Draw", "Total Points"])
    
    df.to_csv('wc.csv', index=False)
    
    print("\nGenerated Team Statistics:")
    print(df)

generate_team_stats()


