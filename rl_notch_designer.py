"""
HDD Notch Filter Auto Design Tool - Full Version
Automated notch filter design for HDD control systems based on reinforcement learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import control.matlab as matlab
from control import freqresp
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
import plant
import utils
from config import get_default_config


class NotchFilterParams:
    """Notch filter parameters"""
    def __init__(self, center_freq, bandwidth, depth):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.depth = depth
        # Q factor is calculated from center_freq and bandwidth
        self.q_factor = center_freq / bandwidth


class HDDNotchDesignEnv:
    """HDD Notch Filter Design Environment"""
    
    def __init__(self, config):
        self.config = config
        
        # System parameters
        self.Ts = config.system_config.Ts
        self.Mr_f = config.system_config.Mr_f
        
        # Plant cases
        self.plant_cases = [
            ('RT', plant.Sys_Pc_vcm_c2, plant.Sys_Pc_pzt_c2),
            ('LT', plant.Sys_Pc_vcm_c3, plant.Sys_Pc_pzt_c3),
            ('HT', plant.Sys_Pc_vcm_c4, plant.Sys_Pc_pzt_c4),
            ('RL', plant.Sys_Pc_vcm_c5, plant.Sys_Pc_pzt_c5),
            ('LL', plant.Sys_Pc_vcm_c6, plant.Sys_Pc_pzt_c6),
            ('HL', plant.Sys_Pc_vcm_c7, plant.Sys_Pc_pzt_c7),
            ('RH', plant.Sys_Pc_vcm_c8, plant.Sys_Pc_pzt_c8),
            ('LH', plant.Sys_Pc_vcm_c9, plant.Sys_Pc_pzt_c9),
            ('HH', plant.Sys_Pc_vcm_c1, plant.Sys_Pc_pzt_c1)
        ]
        
        # Controllers
        self.Sys_Cd_vcm = utils.Sys_Cd_vcm
        self.Sys_Cd_pzt = utils.Sys_Cd_pzt
        
        # Multi-rate filters
        self.Sys_Fm_vcm = utils.Sys_Fm_vcm
        self.Sys_Fm_pzt = utils.Sys_Fm_pzt
        
        # Reward weights and targets
        self.weights = config.env_config['weights']
        self.targets = config.env_config['targets']
        
        # Frequency range
        self.freq_range = np.logspace(1, 5, 1000)  # 10Hz to 100kHz
        
        # State and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # Current state
        self.current_case = None
        self.vcm_plant = None
        self.pzt_plant = None
        self.vcm_features = None
        self.pzt_features = None
        
    def _create_observation_space(self):
        """Create observation space"""
        # 30D state: VCM(15D) + PZT(15D)
        # Each system 5 peaks × 3 features (frequency, magnitude, phase)
        return {
            'shape': (30,),
            'low': np.array([0] * 30),
            'high': np.array([1] * 30)
        }
    
    def _create_action_space(self):
        """Create action space"""
        # 6D action: VCM(3D) + PZT(3D)
        # Each system: center frequency, bandwidth, depth
        # Q factor is calculated as center_freq / bandwidth
        return {
            'shape': (6,),
            'low': np.array([1000, 100, -60, 1000, 100, -60]),  # VCM + PZT
            'high': np.array([50000, 5000, 0, 50000, 5000, 0])
        }
    
    def reset(self):
        """Reset environment"""
        # Randomly select a plant case
        case_idx = np.random.randint(0, len(self.plant_cases))
        self.current_case = self.plant_cases[case_idx]
        
        case_name, self.vcm_plant, self.pzt_plant = self.current_case
        
        # Extract plant features as state
        state = self._extract_plant_features()
        
        return state
    
    def step(self, action):
        """Execute action"""
        # Parse 6D action
        vcm_params = NotchFilterParams(action[0], action[1], action[2])
        pzt_params = NotchFilterParams(action[3], action[4], action[5])
        
        # Create notch filter
        vcm_notch = self._create_notch_filter(vcm_params)
        pzt_notch = self._create_notch_filter(pzt_params)
        
        # Evaluate control system performance
        performance = self._evaluate_control_system(vcm_notch, pzt_notch)
        
        # Calculate reward
        reward = self._calculate_reward(performance)
        
        # Check if done
        done = self._is_done(performance)
        
        # Info
        info = {
            'performance': performance,
            'case': self.current_case[0],
            'vcm_params': vcm_params,
            'pzt_params': pzt_params
        }
        
        return self._extract_plant_features(), reward, done, info
    
    def _extract_plant_features(self):
        """Extract plant features as state"""
        # Calculate frequency response
        vcm_freq_resp = freqresp(self.vcm_plant, self.freq_range)
        pzt_freq_resp = freqresp(self.pzt_plant, self.freq_range)
        
        # Extract peak features
        vcm_features = self._extract_peak_features(vcm_freq_resp)
        pzt_features = self._extract_peak_features(pzt_freq_resp)
        
        # Normalize features
        features = []
        
        # VCM features (15D)
        for peak in vcm_features:
            features.extend([
                peak['freq'] / 50000,  # Normalize frequency
                peak['mag'] / 60,      # Normalize magnitude
                peak['phase'] / 180    # Normalize phase
            ])
        
        # PZT features (15D)
        for peak in pzt_features:
            features.extend([
                peak['freq'] / 50000,  # Normalize frequency
                peak['mag'] / 60,      # Normalize magnitude
                peak['phase'] / 180    # Normalize phase
            ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_peak_features(self, freq_resp):
        """Extract peak features"""
        magnitude = np.abs(freq_resp)
        phase = np.angle(freq_resp) * 180 / np.pi
        
        # Find peaks
        peaks = []
        for i in range(1, len(magnitude) - 1):
            if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                if magnitude[i] > 0.1:  # Only consider significant peaks
                    peaks.append({
                        'freq': self.freq_range[i],
                        'mag': 20 * np.log10(magnitude[i]),
                        'phase': phase[i]
                    })
        
        # Sort by magnitude, take top 5
        peaks.sort(key=lambda x: x['mag'], reverse=True)
        peaks = peaks[:5]
        
        # If less than 5 peaks, pad with zeros
        while len(peaks) < 5:
            peaks.append({'freq': 0, 'mag': 0, 'phase': 0})
        
        return peaks
    
    def _create_notch_filter(self, params):
        """Create notch filter"""
        # Second-order notch filter: H(s) = (s^2 + 2*zeta*wn*s + wn^2) / (s^2 + 2*zeta*wn*s + wn^2)
        wn = 2 * np.pi * params.center_freq
        
        # Calculate damping ratio from bandwidth
        # zeta = bandwidth / (2 * center_freq)
        zeta = params.bandwidth / (2 * params.center_freq)
        
        # Calculate notch depth
        depth_linear = 10**(params.depth / 20)
        
        # Create notch filter
        num = [1, 2*zeta*wn, wn**2]
        den = [1, 2*zeta*wn*depth_linear, wn**2]
        
        notch_filter = matlab.tf(num, den)
        
        # Convert to discrete time
        notch_filter_d = matlab.c2d(notch_filter, self.Ts/self.Mr_f, 'zoh')
        
        return notch_filter_d
    
    def _evaluate_control_system(self, vcm_notch, pzt_notch):
        """Evaluate control system performance"""
        # Create controlled objects with original multirate filters + RL notch filters
        Sys_Pdm0_vcm = matlab.c2d(self.vcm_plant, self.Ts/self.Mr_f, 'zoh')
        Sys_Pdm_vcm = Sys_Pdm0_vcm * self.Sys_Fm_vcm * vcm_notch  # plant × multirate × RL notch
        Sys_Pd_vcm = utils.dts_resampling(Sys_Pdm_vcm, self.Mr_f)
        
        Sys_Pdm0_pzt = matlab.c2d(self.pzt_plant, self.Ts/self.Mr_f, 'zoh')
        Sys_Pdm_pzt = Sys_Pdm0_pzt * self.Sys_Fm_pzt * pzt_notch  # plant × multirate × RL notch
        Sys_Pd_pzt = utils.dts_resampling(Sys_Pdm_pzt, self.Mr_f)
        
        # Calculate frequency response
        Fr_Pd_vcm = utils.freqresp(Sys_Pd_vcm, self.freq_range)
        Fr_Pd_pzt = utils.freqresp(Sys_Pd_pzt, self.freq_range)
        Fr_Cd_vcm = utils.freqresp(self.Sys_Cd_vcm, self.freq_range)
        Fr_Cd_pzt = utils.freqresp(self.Sys_Cd_pzt, self.freq_range)
        
        # Calculate open-loop transfer function
        Fr_L_vcm = Fr_Pd_vcm * Fr_Cd_vcm
        Fr_L_pzt = Fr_Pd_pzt * Fr_Cd_pzt
        Fr_L = Fr_L_vcm + Fr_L_pzt
        
        # Calculate sensitivity function
        Fr_S = 1.0 / (1.0 + Fr_L)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(Fr_L, Fr_S)
        
        return performance
    
    def _calculate_performance_metrics(self, Fr_L, Fr_S):
        """Calculate performance metrics"""
        # Phase margin
        phase_margin = self._calculate_phase_margin(Fr_L)
        
        # Gain margin
        gain_margin = self._calculate_gain_margin(Fr_L)
        
        # Sensitivity peak
        sensitivity_peak = np.max(20 * np.log10(np.abs(Fr_S)))
        
        # Stability
        stability = 1.0 if phase_margin > 0 and gain_margin > 0 else 0.0
        
        # Tracking error (simplified calculation)
        tracking_error = np.mean(20 * np.log10(np.abs(Fr_S)))
        
        return {
            'phase_margin': phase_margin,
            'gain_margin': gain_margin,
            'sensitivity_peak': sensitivity_peak,
            'stability': stability,
            'tracking_error': tracking_error
        }
    
    def _calculate_phase_margin(self, Fr_L):
        """Calculate phase margin"""
        magnitude = np.abs(Fr_L)
        phase = np.angle(Fr_L) * 180 / np.pi
        
        # Find gain crossover frequency
        for i in range(len(magnitude)):
            if magnitude[i] <= 1.0:
                return 180 + phase[i]
        
        return 0
    
    def _calculate_gain_margin(self, Fr_L):
        """Calculate gain margin"""
        magnitude = np.abs(Fr_L)
        phase = np.angle(Fr_L) * 180 / np.pi
        
        # Find phase crossover frequency
        for i in range(len(phase)):
            if phase[i] <= -180:
                return -20 * np.log10(magnitude[i])
        
        return 0
    
    def _calculate_reward(self, performance):
        """Calculate reward"""
        reward = 0
        
        # Phase margin reward
        pm_diff = abs(performance['phase_margin'] - self.targets['phase_margin'])
        reward += self.weights['phase_margin'] * np.exp(-pm_diff / 10)
        
        # Gain margin reward
        gm_diff = abs(performance['gain_margin'] - self.targets['gain_margin'])
        reward += self.weights['gain_margin'] * np.exp(-gm_diff / 2)
        
        # Sensitivity peak reward
        sp_diff = abs(performance['sensitivity_peak'] - self.targets['sensitivity_peak'])
        reward += self.weights['sensitivity_peak'] * np.exp(-sp_diff / 3)
        
        # Stability reward
        reward += self.weights['stability'] * performance['stability']
        
        # Tracking error reward
        te_diff = abs(performance['tracking_error'] - self.targets['tracking_error'])
        reward += self.weights['tracking_error'] * np.exp(-te_diff / 0.1)
        
        # Penalize unstable systems
        if performance['stability'] < 0.5:
            reward -= 10
        
        return reward
    
    def _is_done(self, performance):
        """Check if done"""
        # If all metrics are close to target values, consider done
        pm_ok = abs(performance['phase_margin'] - self.targets['phase_margin']) < 5
        gm_ok = abs(performance['gain_margin'] - self.targets['gain_margin']) < 1
        sp_ok = abs(performance['sensitivity_peak'] - self.targets['sensitivity_peak']) < 2
        stable = performance['stability'] > 0.8
        
        return pm_ok and gm_ok and sp_ok and stable
    
    def get_plant_info(self):
        """Get plant information"""
        return {
            'current_case': self.current_case[0] if self.current_case else None,
            'vcm_features': self.vcm_features,
            'pzt_features': self.pzt_features,
            'freq_range': self.freq_range
        }


class PPOPolicy(nn.Module):
    """PPO Policy Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy network (mean)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        
        # Policy network (std)
        self.policy_std = nn.Linear(hidden_dim, action_dim)
        
        # Value network
        self.value_net = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass"""
        shared_features = self.shared_net(state)
        
        # Policy output
        mean = self.policy_mean(shared_features)
        std = torch.exp(self.policy_std(shared_features))  # Ensure std is positive
        
        # Value output
        value = self.value_net(shared_features)
        
        return mean, std, value
    
    def get_action(self, state, deterministic=False):
        """Get action"""
        mean, std, value = self.forward(state)
        
        if deterministic:
            return mean, value
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPOAgent:
    """PPO Agent"""
    
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy = PPOPolicy(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # PPO parameters
        self.clip_ratio = config.clip_ratio
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def get_action(self, state, deterministic=False):
        """Get action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action, value = self.policy.get_action(state_tensor, deterministic=True)
                return action.cpu().numpy()[0], value.cpu().numpy()[0]
            else:
                action, log_prob, value = self.policy.get_action(state_tensor)
                return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def update(self):
        """Update policy"""
        if len(self.states) < self.config.batch_size:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Calculate discounted rewards
        rewards = self._compute_discounted_rewards()
        
        # Calculate advantages
        advantages = rewards - old_values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Forward pass
            mean, std, values = self.policy(states)
            
            # Calculate new log probabilities
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            # Calculate entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear buffer
        self.clear_buffer()
    
    def _compute_discounted_rewards(self):
        """Compute discounted rewards"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        discounted_rewards = np.zeros_like(rewards)
        running_reward = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_reward = 0
            running_reward = rewards[t] + self.config.gamma * running_reward
            discounted_rewards[t] = running_reward
        
        return torch.FloatTensor(discounted_rewards).to(self.device)
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save_model(self, filepath):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_notch_designer():
    """Train notch filter designer"""
    print("=== Training HDD Notch Filter Designer ===")
    
    # Get configuration
    config = get_default_config()
    
    # Create environment
    env = HDDNotchDesignEnv(config)
    
    # Create agent
    state_dim = env.observation_space['shape'][0]
    action_dim = env.action_space['shape'][0]
    agent = PPOAgent(state_dim, action_dim, config.training_config)
    
    print(f"Environment info:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Plant cases: {len(env.plant_cases)}")
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(config.training_config.max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.training_config.max_steps_per_episode):
            # Get action
            action, log_prob, value = agent.get_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update policy
        agent.update()
        
        episode_rewards.append(episode_reward)
        
        # Record best performance
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model('best_notch_designer.pth')
        
        # Print progress
        if episode % config.training_config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.training_config.log_interval:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}")
    
    # Save final model
    agent.save_model('final_notch_designer.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()
    
    print("Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    print("Models saved: best_notch_designer.pth, final_notch_designer.pth")


if __name__ == "__main__":
    train_notch_designer()