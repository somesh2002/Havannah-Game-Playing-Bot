import os

runs = 12
players  = ['ai12','ai4','ai11','ai10','ai9']
for player1 in players:
    for player2 in players:
        if(player1!=player2):
            for i in range(runs):
                os.system(f"python3 game.py {player1} {player2} --dim 6 --time 400 --mode server")


