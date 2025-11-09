"""
Test trained RL agents against each other using the official game.
"""
import numpy as np
from stable_baselines3 import PPO
from case_closed_game import Game, Direction, GameResult

# Try to import MaskablePPO if available
try:
    from sb3_contrib import MaskablePPO
    USE_MASKING = True
except ImportError:
    USE_MASKING = False


def observation_from_game(game, agent_id, include_boost_features=False):
    """
    Create observation vector matching the training environment format.
    Old models: 2520 features (no boost info)
    New models: 2522 features (with boost info)
    """
    height, width = 18, 20
    obs = np.zeros((height, width, 7), dtype=np.float32)
    
    # Channel 0: Empty cells
    for y in range(height):
        for x in range(width):
            if game.board.grid[y][x] == 0:
                obs[y, x, 0] = 1.0
    
    # Channel 1: My trail
    # Channel 2: My head
    # Channel 3: Opponent trail
    # Channel 4: Opponent head
    if agent_id == 1:
        my_agent = game.agent1
        opp_agent = game.agent2
    else:
        my_agent = game.agent2
        opp_agent = game.agent1
    
    # My trail and head
    if my_agent.alive:
        for pos in my_agent.trail:
            x, y = pos
            obs[y, x, 1] = 1.0
        if len(my_agent.trail) > 0:
            head_x, head_y = my_agent.trail[-1]
            obs[head_y, head_x, 2] = 1.0
    
    # Opponent trail and head
    if opp_agent.alive:
        for pos in opp_agent.trail:
            x, y = pos
            obs[y, x, 3] = 1.0
        if len(opp_agent.trail) > 0:
            head_x, head_y = opp_agent.trail[-1]
            obs[head_y, head_x, 4] = 1.0
    
    # Channels 5-6: Territory maps (simplified - just set to 0)
    # In training these are computed via flood fill, but for testing we can skip
    
    # Flatten spatial features (2520 features)
    obs_flat = obs.flatten()
    
    # Optionally add boost counts for newer models
    if include_boost_features:
        my_boosts = my_agent.boosts_remaining / 3.0
        opp_boosts = opp_agent.boosts_remaining / 3.0
        obs_with_boosts = np.concatenate([obs_flat, [my_boosts, opp_boosts]])
        return obs_with_boosts
    
    return obs_flat


def action_to_direction(action, current_direction):
    """
    Convert action index to Direction and boost flag.
    Handles 180° reversal auto-correction like the training environment.
    """
    # Actions 0-3: UP, DOWN, LEFT, RIGHT (no boost)
    # Actions 4-7: UP, DOWN, LEFT, RIGHT (with boost)
    use_boost = action >= 4
    base_action = action % 4
    
    direction_map = {
        0: Direction.UP,
        1: Direction.DOWN,
        2: Direction.LEFT,
        3: Direction.RIGHT
    }
    
    requested_dir = direction_map[base_action]
    
    # Check for 180-degree reversal
    is_reverse = (
        requested_dir.value[0] == -current_direction.value[0]
        and requested_dir.value[1] == -current_direction.value[1]
    )
    
    # If reversing, use current direction instead (continue straight)
    used_dir = current_direction if is_reverse else requested_dir
    
    # If trying to boost but reversing, don't boost
    if is_reverse:
        use_boost = False
    
    return used_dir, use_boost


def play_game(agent1_path, agent2_path, render=True):
    """Play a game between two trained agents."""
    
    # Load agents
    print(f"Loading Agent 1 from: {agent1_path}")
    if USE_MASKING:
        try:
            agent1_model = MaskablePPO.load(agent1_path, device="cpu")
            use_boost_features1 = True  # Newer models with masking
        except:
            agent1_model = PPO.load(agent1_path, device="cpu")
            use_boost_features1 = False  # Older models
    else:
        agent1_model = PPO.load(agent1_path, device="cpu")
        use_boost_features1 = False  # Older models
    
    print(f"Loading Agent 2 from: {agent2_path}")
    if USE_MASKING:
        try:
            agent2_model = MaskablePPO.load(agent2_path, device="cpu")
            use_boost_features2 = True  # Newer models with masking
        except:
            agent2_model = PPO.load(agent2_path, device="cpu")
            use_boost_features2 = False  # Older models
    else:
        agent2_model = PPO.load(agent2_path, device="cpu")
        use_boost_features2 = False  # Older models
    
    # Create game
    game = Game()
    
    print("\n" + "="*60)
    print("Starting Game!")
    print("="*60)
    
    if render:
        print(f"\nInitial Board:")
        print(game.board)
    
    # Game loop
    turn = 0
    result = None
    
    while result is None and turn < 200:
        turn += 1
        
        # Get observations for both agents (with correct feature count)
        obs1 = observation_from_game(game, agent_id=1, include_boost_features=use_boost_features1)
        obs2 = observation_from_game(game, agent_id=2, include_boost_features=use_boost_features2)
        
        # Get actions from models
        action1, _ = agent1_model.predict(obs1, deterministic=True)
        action2, _ = agent2_model.predict(obs2, deterministic=True)
        
        # Convert actions to directions and boost flags (with 180° auto-correction)
        dir1, boost1 = action_to_direction(int(action1), game.agent1.direction)
        dir2, boost2 = action_to_direction(int(action2), game.agent2.direction)
        
        # Step the game
        result = game.step(dir1, dir2, boost1, boost2)
        
        if render and turn % 10 == 0:
            print(f"\n--- Turn {turn} ---")
            print(f"Agent 1: {dir1.name} (boost: {boost1}), Boosts left: {game.agent1.boosts_remaining}")
            print(f"Agent 2: {dir2.name} (boost: {boost2}), Boosts left: {game.agent2.boosts_remaining}")
            print(game.board)
    
    # Print result
    print("\n" + "="*60)
    print("Game Over!")
    print("="*60)
    print(f"Total turns: {turn}")
    print(f"Agent 1 trail length: {game.agent1.length}")
    print(f"Agent 2 trail length: {game.agent2.length}")
    
    if result == GameResult.AGENT1_WIN:
        print("🏆 Agent 1 WINS!")
        return 1
    elif result == GameResult.AGENT2_WIN:
        print("🏆 Agent 2 WINS!")
        return 2
    else:
        print("🤝 DRAW!")
        return 0


def run_tournament(agent1_path, agent2_path, num_games=10):
    """Run multiple games and report statistics."""
    print("\n" + "="*60)
    print(f"TOURNAMENT: {num_games} games")
    print("="*60)
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for i in range(num_games):
        print(f"\n### Game {i+1}/{num_games} ###")
        result = play_game(agent1_path, agent2_path, render=False)
        
        if result == 1:
            agent1_wins += 1
        elif result == 2:
            agent2_wins += 2
        else:
            draws += 1
    
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    print(f"Agent 1 wins: {agent1_wins} ({agent1_wins/num_games*100:.1f}%)")
    print(f"Agent 2 wins: {agent2_wins} ({agent2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Default paths
    agent1_path = "rl_training/model/independent_pop/agent_0_final.zip"
    agent2_path = "rl_training/model/independent_pop/agent_1_final.zip"
    
    # Check if paths provided as arguments
    if len(sys.argv) >= 3:
        agent1_path = sys.argv[1]
        agent2_path = sys.argv[2]
    
    print("="*60)
    print("RL AGENT TESTING")
    print("="*60)
    print(f"Agent 1: {agent1_path}")
    print(f"Agent 2: {agent2_path}")
    
    # Play single game with rendering
    play_game(agent1_path, agent2_path, render=True)
    
    # Optional: Run tournament
    response = input("\nRun tournament (10 games)? (y/n): ")
    if response.lower() == 'y':
        run_tournament(agent1_path, agent2_path, num_games=10)
