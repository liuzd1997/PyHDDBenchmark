"""
HDD Notch Filter Auto Design Tool - Simple Test
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from simple_rl_notch_designer import SimpleHDDNotchDesignEnv, NotchFilterParams, PPOAgent
    import config
    print("✓ Successfully imported RL modules")
except ImportError as e:
    print(f"✗ Failed to import RL modules: {e}")
    sys.exit(1)

def quick_test():
    """Quick test of system functionality"""
    print("=== HDD Notch Filter Designer Test ===")
    
    try:
        # Create environment
        env_config = config.get_simple_config()
        env = SimpleHDDNotchDesignEnv(env_config)
        
        print(f"✓ Environment created successfully")
        print(f"  State dimension: {env.observation_space['shape'][0]}")
        print(f"  Action dimension: {env.action_space['shape'][0]}")
        
        # Test Reset
        state = env.reset()
        print(f"✓ Reset successful. State shape: {state.shape}")
        
        # Test Step (Iterative)
        print("\n=== Iterative Step Test ===")
        # Action: Increase VCM Freq, Decrease PZT Depth
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, -0.1]) 
        
        next_state, reward, done, info = env.step(action)
        
        print(f"✓ Step successful")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Performance: PM={info['performance']['phase_margin']:.1f}°, GM={info['performance']['gain_margin']:.1f}dB")
        print(f"  Notch Params: {info['notch_params']}")
        
        # Test Randomization
        print("\n=== Randomization Test ===")
        env.reset()
        f1 = env.plant_features[0] # First peak freq
        env.reset()
        f2 = env.plant_features[0]
        print(f"  Peak 1 Freq (Ep 1): {f1:.4f}")
        print(f"  Peak 1 Freq (Ep 2): {f2:.4f}")
        if f1 != f2:
            print("✓ Randomization active (features changed)")
        else:
            print("? Randomization might be small or unlucky (features identical)")
            
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """Test training functionality"""
    print("\n=== Training Functionality Test ===")
    
    try:
        # Create environment
        env_config = config.get_simple_config()
        env = SimpleHDDNotchDesignEnv(env_config)
        
        # Create agent
        train_config = config.TrainingConfig()
        train_config.max_episodes = 5  # Only train 5 episodes for testing
        train_config.max_steps_per_episode = 10
        train_config.log_interval = 1
        
        state_dim = env.observation_space['shape'][0]
        action_dim = env.action_space['shape'][0]
        agent = PPOAgent(state_dim, action_dim, train_config)
        # Set action bounds for proper clipping
        agent.action_low = env.action_space['low']
        agent.action_high = env.action_space['high']
        
        # Training loop
        episode_rewards = []
        models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for episode in range(train_config.max_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(train_config.max_steps_per_episode):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                agent.store_transition(state, action, reward, value, log_prob, done)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            agent.update()
            episode_rewards.append(episode_reward)
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
        
        avg_reward = np.mean(episode_rewards)
        print(f"✓ Training test completed")
        print(f"  Average reward: {avg_reward:.2f}")
        
        # Save reward history for inspection
        rewards_array = np.array(episode_rewards, dtype=np.float32)
        rewards_npy = os.path.join(models_dir, f"test_simple_notch_rewards_{timestamp}.npy")
        rewards_csv = os.path.join(models_dir, f"test_simple_notch_rewards_{timestamp}.csv")
        np.save(rewards_npy, rewards_array)
        np.savetxt(rewards_csv, rewards_array, fmt='%.6f', delimiter=',', header='episode_reward', comments='')
        
        plt.figure(figsize=(8, 3))
        plt.plot(episode_rewards, marker='o')
        plt.title('Test Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        reward_plot = os.path.join(models_dir, f"test_simple_notch_rewards_{timestamp}.png")
        plt.savefig(reward_plot)
        plt.close()
        print(f"  Reward history saved to: {rewards_npy}, {rewards_csv}, {reward_plot}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting tests...")
    
    # Basic functionality test
    success1 = quick_test()
    
    # Training functionality test
    success2 = test_training()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed, please check system configuration")