import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import gym

scalar_writer = SummaryWriter('TD3')
class Critic(nn.Module):
    def __init__(self, obs_dim ,action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=1)
        q = torch.relu(self.l1(sa))
        q = torch.relu(self.l2(q))
        q = self.l3(q)
        return q

class Actor(nn.Module):
    '''一般环境的a的上下界对称，所以直接forward输出对应环境的动作'''
    def __init__(self, obs_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, obs):
        '''返回值直接乘上action的范围高度，得到可以用到环境的动作'''
        a = torch.relu(self.l1(obs))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        
        return self.max_action * a

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

def update_net(model, target_model, tau=1.):
    '''更新目标网络'''
    for tar_param, param in zip(target_model.parameters(), model.parameters()):
        tar_param.data.copy_(param.data * tau + tar_param.data * (1.0 - tau))

class TD3:
    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step, noise_std, noise_bound, critic_lr, actor_lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.update_step = 0
        self.delay_step = delay_step
        
        # 初始化6个网络
        self.actor = Actor(self.obs_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim, self.max_action).to(self.device)
        
        self.critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic1_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        #  初始化目标网络的权重
        update_net(self.actor, self.actor_target, tau=1.)
        update_net(self.critic1, self.critic1_target, tau=1.)
        update_net(self.critic2, self.critic2_target, tau=1.)
        
        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # 设置一个mse函数
        self.loss_fn = torch.nn.MSELoss()
        
        # 初始化经验池
        self.replay_buffer = BasicBuffer(buffer_maxlen)
        
        # 初始化记录scalar的字典
        self.summaries = {}
        
    def get_action(self, obs):
        '''因为网络输出的直接是满足动作区间的动作，所以不需要rescale'''
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.squeeze(0).cpu().detach().numpy()
        return action
    
    def update(self, batch_size):
        '''更新网络'''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        done_batch = done_batch.view(-1, 1)  # 转换成 （batchsize, 1） 的形状，为了下面的相乘
        
        action_noise = self.generate_noise(action_batch) # 产生一批和一批动作一样形状的高斯噪声
        # actions加过噪音要裁剪到目标范围，这里为何是next_state??论文和论文的代码不符合啊，可能就是个形式？还是要根据Q里的s或者s'
        actions_hat = (self.actor_target(next_state_batch) + action_noise).clamp(-self.max_action, self.max_action)
        next_q1 = self.critic1_target(next_state_batch, actions_hat)
        next_q2 = self.critic2_target(next_state_batch, actions_hat)
        min_next_q = torch.min(next_q1, next_q2)
        y = (reward_batch + (1.-done_batch) * self.gamma * min_next_q).detach()   # 这一项就是为了让done之后的数据没有意义！不需要每done一次就训练，而是以多少步进行一次训练，这样就变成离线学习了啊！
        curr_q1 = self.critic1(state_batch, action_batch)
        curr_q2 = self.critic2(state_batch, action_batch)

        loss_critic1 = self.loss_fn(curr_q1, y)
        loss_critic2 = self.loss_fn(curr_q2, y)
        self.summaries['critic_loss'] = loss_critic1.detach().item()
        
        # 更新两个critic网络
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        
        loss_critic1.backward()
        loss_critic2.backward()
        
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # 延迟更新策略网络和目标网络
        if self.update_step % self.delay_step == 0:
            actor_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()
            self.summaries['actor_loss'] = actor_loss.detach().item()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新目标网络
            update_net(self.actor, self.actor_target, tau=self.tau)
            update_net(self.critic1, self.critic1_target, tau=self.tau)
            update_net(self.critic2, self.critic2_target, tau=self.tau)
        
        self.update_step += 1
        
    def generate_noise(self, action_batch):
        '''对一批动作产生同样维度的噪声，用于探索'''
        # torch.normal(mean, std) 他们只要有一个有形状就行
        noise = torch.normal(mean=torch.zeros(action_batch.size()), std=self.noise_std)
        noise = torch.clamp(noise, -self.noise_bound, self.noise_bound).to(self.device)
        return noise

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))



def train(env, agent, max_episodes, max_steps, batch_size, std_train=0.1, render=False):
    '''按照episode训练, 可能会浪费一些步数，刚开始每回合训练不满，
    优点：可以自定义每回合最高步数限制，对一些非回合制游戏有效
    缺点：如果设置了每回合最大目标步数，智能体就会被限制住，到了这个目标就不会再继续增长了
    '''
    total_steps = 0
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
                
            if total_steps < 25e3:
                action = env.action_space.sample()
            else:
                action = (agent.get_action(state) + np.random.normal(0, agent.max_action * std_train, size=agent.action_dim)).astype(np.float32)
                action = np.clip(action, -agent.max_action, agent.max_action)
            
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            total_steps += 1
            
            if total_steps > 25e3:
                agent.update(batch_size)     
            
            # 为什么奖励很少刚开始，因为，我直接在done的时候就终止训练了，done之后给任何action都没有意义了。
            # if step == max_steps -1 :
            #     print('totle_step {}, episode_reward {}, step {}'.format(total_steps, episode_reward, step))
            #     break
            if done or step == max_steps -1:
                print('totle_step {}, episode_reward {}, step {}'.format(total_steps, episode_reward, step))
                break
            state = next_state

def train_step(env, agent, max_steps, begin_steps, batch_size, std_train=0.1, render=False):
    '''按照所有步数进行训练：
    优点：每回合步数没有上限，一直会达到环境默认上限。
    缺点：如果环境本身没有最高步数限制返回done，会陷入死循环，对一些非回合制游戏无效
    这么写其实有点像在线训练了，离线训练是先收集数据，再进行训练'''
    state, done = env.reset(), False
    episode_steps = 0
    episode_reward = 0
    episode_num = 0
    
    for t in range(max_steps):
        episode_steps += 1
        if render:
                env.render()
        if t < begin_steps:
            action = env.action_space.sample()
        else:
            action = (agent.get_action(state) + np.random.normal(0, agent.max_action * std_train, size=agent.action_dim)).astype(np.float32)
            action = np.clip(action, -agent.max_action, agent.max_action)
            
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        if t >= begin_steps:
            agent.update(batch_size)     
            scalar_writer.add_scalar('loss/actor_loss',agent.summaries['actor_loss'], t)
            scalar_writer.add_scalar('loss/critic_loss',agent.summaries['critic_loss'], t)

        if t % 20e3 == 0:
            agent.save_model('TD3/TD3_hopper.pth')

        if done:
            print('totle_step {},episode_num {}, episode_steps {}, episode_reward {}'.format(t+1, episode_num, episode_steps, episode_reward))
            scalar_writer.add_scalar('main/episode_reward', episode_reward, episode_num)
            scalar_writer.add_scalar('main/episode_steps', episode_steps, episode_num)
            #  重置各种东西
            state, done = env.reset(), False
            episode_reward = 0
            episode_steps = 0
            episode_num += 1


if __name__ == '__main__':
    env = gym.make("Hopper-v2")   # __init__默认是1000步一回合！
    gamma = 0.99
    tau = 0.005
    noise_std = 0.2
    bound = 0.5
    delay_step = 2
    buffer_maxlen = int(1e6)
    critic_lr = 3e-4
    actor_lr = 3e-4

    agent = TD3(env, gamma, tau, buffer_maxlen, delay_step, noise_std, bound, critic_lr, actor_lr)
    # # 按照回合训练
    # max_episodes = 10000
    # max_steps = 1000
    # batch_size = 64
    # train(env, agent, max_episodes, max_steps, batch_size, render=False)

    # 按照总步数训练
    max_steps = int(1e6)
    begin_steps = 25e3
    batch_size = 64
    train_step(env, agent, max_steps, begin_steps, batch_size, std_train=0.1, render=False)