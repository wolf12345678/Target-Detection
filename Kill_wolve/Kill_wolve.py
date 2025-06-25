import numpy as np
import pandas as pd
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces


class BTCTEnvironment(gym.Env):
    """血染钟楼游戏环境，用于强化学习训练"""

    def __init__(self, num_players=12):
        super(BTCTEnvironment, self).__init__()
        self.num_players = num_players

        # 定义动作空间：选择三名玩家作为狼坑
        self.action_space = spaces.MultiDiscrete([num_players, num_players, num_players])

        # 定义观察空间：每个玩家的特征向量
        # 特征包括：存活状态、声称角色、投票历史、被投票情况、发言特征等
        player_features = 15  # 每个玩家的特征数量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_players, player_features),
            dtype=np.float32
        )

        # 游戏状态
        self.players = []
        self.day = 0
        self.round = 0  # 每天内的轮次
        self.game_state = {}
        self.true_evil_team = []  # 真实的邪恶阵营成员

        # 历史信息
        self.speech_history = []
        self.vote_history = []
        self.death_history = []
        self.action_history = []

        # 上一轮的预测
        self.previous_prediction = None

    def reset(self):
        """重置环境"""
        self.day = 0
        self.round = 0

        # 初始化玩家
        self.players = []
        for i in range(self.num_players):
            self.players.append({
                'id': i,
                'alive': True,
                'claimed_role': None,
                'votes_given': [],
                'votes_received': [],
                'speech_features': np.zeros(5),  # 5维发言特征向量
                'suspicious_score': 0.5,  # 初始可疑度为中等
                'info_shared': 0,  # 分享的信息量
                'consistency_score': 0.5,  # 行为一致性
                'relationships': np.zeros(self.num_players)  # 与其他玩家的关系
            })

        # 随机设置真实的邪恶阵营(1个恶魔+2个爪牙)
        evil_indices = random.sample(range(self.num_players), 3)
        self.true_evil_team = evil_indices

        # 重置历史记录
        self.speech_history = []
        self.vote_history = []
        self.death_history = []
        self.action_history = []
        self.previous_prediction = None

        # 返回初始观察
        return self._get_observation()

    def step(self, action):
        """执行一步动作并返回新状态"""
        # 动作是预测的三狼坑: [demon_id, minion1_id, minion2_id]
        prediction = [int(action[0]), int(action[1]), int(action[2])]
        self.action_history.append(prediction)

        # 计算奖励
        reward = self._calculate_reward(prediction)

        # 更新轮次
        self.round += 1
        if self.round >= 3:  # 假设每天有3轮
            self.round = 0
            self.day += 1

        # 检查游戏是否结束
        done = self.day >= 5  # 假设5天后游戏结束

        # 存储当前预测以便下一次比较
        self.previous_prediction = prediction

        # 更新游戏状态(这里应该集成真实游戏中的新信息)
        self._update_game_state()

        # 返回新的观察、奖励、是否结束和额外信息
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """生成当前游戏状态的观察向量"""
        observations = np.zeros((self.num_players, 15), dtype=np.float32)

        for i, player in enumerate(self.players):
            # 基本特征
            observations[i, 0] = 1.0 if player['alive'] else 0.0
            observations[i, 1] = len(player['votes_given']) / max(1, self.day)
            observations[i, 2] = len(player['votes_received']) / max(1, self.day)

            # 角色类型(编码)
            role_code = 0.0
            if player['claimed_role']:
                if player['claimed_role'] in ['fortune_teller', 'empath', 'undertaker']:
                    role_code = 1.0  # 信息型镇民
                elif player['claimed_role'] in ['monk', 'slayer', 'soldier']:
                    role_code = 2.0  # 保护型镇民
                elif player['claimed_role'] in ['butler', 'drunk', 'recluse', 'saint']:
                    role_code = 3.0  # 外来者
                elif player['claimed_role'] in ['poisoner', 'spy', 'scarlet_woman', 'baron']:
                    role_code = 4.0  # 爪牙
                elif player['claimed_role'] == 'imp':
                    role_code = 5.0  # 恶魔
            observations[i, 3] = role_code

            # 可疑度分数
            observations[i, 4] = player['suspicious_score']

            # 发言特征
            observations[i, 5:10] = player['speech_features']

            # 信息分享
            observations[i, 10] = player['info_shared']

            # 行为一致性
            observations[i, 11] = player['consistency_score']

            # 平均关系得分
            observations[i, 12] = np.mean(player['relationships'])

            # 投票一致性(与大多数人投票相同的比例)
            vote_consistency = 0.0
            if player['votes_given']:
                matching_votes = 0
                for day, target in player['votes_given']:
                    # 统计当天投给相同目标的玩家数
                    day_votes = [v for p in self.players for d, v in p['votes_given'] if d == day]
                    if day_votes:
                        most_common_vote = max(set(day_votes), key=day_votes.count)
                        if target == most_common_vote:
                            matching_votes += 1
                vote_consistency = matching_votes / len(player['votes_given'])
            observations[i, 13] = vote_consistency

            # 被目标次数(成为技能或投票目标的频率)
            observations[i, 14] = self._get_targeting_frequency(i)

        return observations

    def _calculate_reward(self, prediction):
        """计算预测的奖励"""
        # 1. 与真实邪恶团队的匹配程度
        match_count = len(set(prediction) & set(self.true_evil_team))
        match_reward = match_count / 3  # 完全匹配得1分

        # 2. 预测稳定性奖励(避免频繁无意义的改变)
        stability_reward = 0
        if self.previous_prediction:
            stability = len(set(prediction) & set(self.previous_prediction)) / 3
            # 只有当匹配度提高时才给予稳定性奖励
            stability_reward = 0.2 * stability if stability > 0.5 else -0.1

        # 3. 基于新证据的调整奖励
        evidence_reward = 0
        if self.previous_prediction and self.game_state.get('new_evidence'):
            # 如果有新证据指向某个玩家，且预测调整为包含该玩家，给予奖励
            suspected_player = self.game_state['new_evidence'].get('suspected_player')
            if suspected_player is not None and suspected_player in prediction and suspected_player not in self.previous_prediction:
                evidence_reward = 0.3

        # 总奖励
        total_reward = match_reward + stability_reward + evidence_reward
        return total_reward

    def _update_game_state(self):
        """更新游戏状态，集成新信息"""
        # 这个函数应该集成真实游戏中的新信息
        # 在实际应用中，这里会接收玩家的发言、投票和死亡信息
        pass

    def _get_targeting_frequency(self, player_id):
        """计算玩家被针对的频率"""
        # 简化实现
        return random.uniform(0, 1)  # 在实际应用中应该基于游戏记录计算


class ActorCritic(nn.Module):
    """Actor-Critic网络，用于强化学习训练"""

    def __init__(self, num_players, num_features):
        super(ActorCritic, self).__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_players * num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor网络 - 输出动作概率
        self.actor_demon = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_players),
            nn.Softmax(dim=-1)
        )

        self.actor_minion1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_players),
            nn.Softmax(dim=-1)
        )

        self.actor_minion2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_players),
            nn.Softmax(dim=-1)
        )

        # Critic网络 - 评估状态价值
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平输入

        features = self.feature_extractor(x)

        # 生成三个角色的动作概率
        demon_probs = self.actor_demon(features)
        minion1_probs = self.actor_minion1(features)
        minion2_probs = self.actor_minion2(features)

        # 状态价值
        value = self.critic(features)

        return demon_probs, minion1_probs, minion2_probs, value

    def act(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        demon_probs, minion1_probs, minion2_probs, _ = self(state)

        # 采样动作
        demon_dist = Categorical(demon_probs)
        minion1_dist = Categorical(minion1_probs)
        minion2_dist = Categorical(minion2_probs)

        demon = demon_dist.sample().item()

        # 确保不重复选择同一玩家
        minion1_adjusted = minion1_probs.clone()
        minion1_adjusted[0, demon] = 0
        minion1_adjusted = minion1_adjusted / minion1_adjusted.sum()
        minion1_dist = Categorical(minion1_adjusted)
        minion1 = minion1_dist.sample().item()

        minion2_adjusted = minion2_probs.clone()
        minion2_adjusted[0, demon] = 0
        minion2_adjusted[0, minion1] = 0
        minion2_adjusted = minion2_adjusted / minion2_adjusted.sum()
        minion2_dist = Categorical(minion2_adjusted)
        minion2 = minion2_dist.sample().item()

        return [demon, minion1, minion2]


class BTCTAnalyzer:
    """血染钟楼实时分析器"""

    def __init__(self, num_players=12):
        self.num_players = num_players
        self.env = BTCTEnvironment(num_players)

        # 创建强化学习模型
        self.model = ActorCritic(num_players, 15)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 游戏状态
        self.players = []
        self.day = 0

        # 历史预测
        self.predictions_history = []

        # 预训练模型路径
        self.model_path = "btct_analyzer_model.pt"
        self.try_load_model()

    def try_load_model(self):
        """尝试加载预训练模型"""
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            print("成功加载预训练模型")
        except:
            print("无法加载预训练模型，使用新模型")

    def save_model(self):
        """保存模型"""
        torch.save(self.model.state_dict(), self.model_path)
        print("模型已保存")

    def initialize_game(self, player_names):
        """初始化新游戏"""
        self.players = []
        for i, name in enumerate(player_names):
            self.players.append({
                'id': i,
                'name': name,
                'alive': True,
                'claimed_role': None,
                'votes_given': [],
                'votes_received': [],
                'speech_records': [],
                'suspicious_features': {
                    'contradictions': 0,
                    'information_sharing': 0,
                    'voting_pattern': 0.5,
                    'role_conflicts': 0,
                    'behavioral_changes': 0
                },
                'suspicious_score': 0.5,
                'feature_vector': np.zeros(15)
            })

        self.day = 0
        self.predictions_history = []
        print(f"游戏初始化完成，{len(self.players)}名玩家已准备就绪")

    def update_player_info(self, player_updates):
        """更新玩家信息"""
        for update in player_updates:
            player_id = update['id']
            player = next((p for p in self.players if p['id'] == player_id), None)
            if not player:
                continue

            # 更新基本信息
            if 'alive' in update:
                player['alive'] = update['alive']
            if 'claimed_role' in update:
                player['claimed_role'] = update['claimed_role']

            # 更新发言记录
            if 'speech' in update:
                player['speech_records'].append({
                    'day': self.day,
                    'content': update['speech']
                })

                # 分析发言，更新可疑特征
                self._analyze_speech(player, update['speech'])

            # 更新投票信息
            if 'vote_target' in update:
                player['votes_given'].append((self.day, update['vote_target']))
                target = next((p for p in self.players if p['id'] == update['vote_target']), None)
                if target:
                    target['votes_received'].append((self.day, player_id))

                # 分析投票，更新可疑特征
                self._analyze_vote(player, update['vote_target'])

            # 更新特征向量
            self._update_feature_vector(player)

    def record_execution(self, player_id):
        """记录处决结果"""
        player = next((p for p in self.players if p['id'] == player_id), None)
        if player:
            player['alive'] = False
            print(f"记录: 玩家{player['name']}被处决")

    def record_night_death(self, player_id):
        """记录夜间死亡"""
        player = next((p for p in self.players if p['id'] == player_id), None)
        if player:
            player['alive'] = False
            print(f"记录: 玩家{player['name']}夜间死亡")

    def next_day(self):
        """进入下一天"""
        self.day += 1
        print(f"\n=== 第{self.day}天 ===")

    def analyze_wolf_team(self):
        """实时分析狼队阵容"""
        print("\n=== 狼人阵营实时分析 ===")
        print(f"当前游戏天数: {self.day}")

        # 获取当前游戏状态观察
        observation = self._get_current_observation()

        # 使用模型预测最可能的狼队阵容
        with torch.no_grad():
            prediction = self.model.act(observation)

        # 保存预测历史
        self.predictions_history.append({
            'day': self.day,
            'prediction': prediction
        })

        # 获取预测的玩家名称
        demon_id, minion1_id, minion2_id = prediction
        demon = next((p for p in self.players if p['id'] == demon_id), None)
        minion1 = next((p for p in self.players if p['id'] == minion1_id), None)
        minion2 = next((p for p in self.players if p['id'] == minion2_id), None)

        # 输出预测结果
        print("\n当前预测的狼坑组合:")
        if demon:
            print(f"恶魔: 玩家{demon['name']} (可疑度: {demon['suspicious_score']:.2f})")
        if minion1:
            print(f"爪牙1: 玩家{minion1['name']} (可疑度: {minion1['suspicious_score']:.2f})")
        if minion2:
            print(f"爪牙2: 玩家{minion2['name']} (可疑度: {minion2['suspicious_score']:.2f})")

        # 解释预测理由
        print("\n分析理由:")
        if demon:
            self._explain_prediction(demon)
        if minion1:
            self._explain_prediction(minion1)
        if minion2:
            self._explain_prediction(minion2)

        # 提供备选组合
        self._suggest_alternative_combinations()

        return {
            'demon': demon_id,
            'minions': [minion1_id, minion2_id]
        }

    def train_model(self, num_episodes=100):
        """训练强化学习模型"""
        print(f"开始训练模型，共{num_episodes}个回合...")

        # 存储训练数据
        all_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = []

            while not done:
                # 选择动作
                action = self.model.act(state)

                # 执行动作
                next_state, reward, done, _ = self.env.step(action)

                # 存储奖励
                episode_rewards.append(reward)

                # 更新状态
                state = next_state

            # 计算总奖励
            total_reward = sum(episode_rewards)
            all_rewards.append(total_reward)

            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = sum(all_rewards[-10:]) / 10
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

            # 更新模型
            self._update_model(episode_rewards)

        # 保存训练好的模型
        self.save_model()
        print("模型训练完成！")

        return all_rewards

    def _get_current_observation(self):
        """获取当前游戏状态的观察向量"""
        observation = np.zeros((self.num_players, 15), dtype=np.float32)

        for i, player in enumerate(self.players):
            observation[i] = player['feature_vector']

        return observation

    def _update_feature_vector(self, player):
        """更新玩家的特征向量"""
        feature_vector = np.zeros(15)

        # 基本特征
        feature_vector[0] = 1.0 if player['alive'] else 0.0
        feature_vector[1] = len(player['votes_given']) / max(1, self.day)
        feature_vector[2] = len(player['votes_received']) / max(1, self.day)

        # 角色类型(编码)
        role_code = 0.0
        if player['claimed_role']:
            if player['claimed_role'] in ['fortune_teller', 'empath', 'undertaker']:
                role_code = 1.0
            elif player['claimed_role'] in ['monk', 'slayer', 'soldier']:
                role_code = 2.0
            elif player['claimed_role'] in ['butler', 'drunk', 'recluse', 'saint']:
                role_code = 3.0
            elif player['claimed_role'] in ['poisoner', 'spy', 'scarlet_woman', 'baron']:
                role_code = 4.0
            elif player['claimed_role'] == 'imp':
                role_code = 5.0
        feature_vector[3] = role_code

        # 可疑度分数
        suspicious_score = self._calculate_suspicious_score(player)
        player['suspicious_score'] = suspicious_score
        feature_vector[4] = suspicious_score

        # 发言特征 - 简化为全0
        feature_vector[5:10] = 0

        # 信息分享 - 简化实现
        feature_vector[10] = self._calculate_info_sharing(player)

        # 行为一致性 - 简化实现
        feature_vector[11] = 1.0 - player['suspicious_features']['contradictions'] / max(1,
                                                                                         len(player['speech_records']))

        # 关系得分 - 简化为0
        feature_vector[12] = 0

        # 投票一致性
        feature_vector[13] = player['suspicious_features']['voting_pattern']

        # 被目标频率
        feature_vector[14] = len(player['votes_received']) / max(1, self.day * self.num_players)

        # 更新特征向量
        player['feature_vector'] = feature_vector

    def _calculate_suspicious_score(self, player):
        """计算玩家的可疑度分数"""
        if not player['alive']:
            return 0

        # 加权计算可疑度得分
        features = player['suspicious_features']
        weights = {
            'contradictions': 0.25,
            'information_sharing': 0.2,
            'voting_pattern': 0.3,
            'role_conflicts': 0.15,
            'behavioral_changes': 0.1
        }

        score = 0
        for feature, weight in weights.items():
            if feature == 'information_sharing':
                # 信息分享越多越不可疑
                score += weight * (1.0 - features[feature])
            else:
                # 其他特征越高越可疑
                score += weight * features[feature]

        return min(1.0, max(0.0, score))

    def _calculate_info_sharing(self, player):
        """计算玩家分享信息的程度"""
        if not player['speech_records']:
            return 0

        info_count = 0
        for speech in player['speech_records']:
            content = speech['content'].lower()
            info_keywords = ['查验', '预言', '灵媒', '邻座', '送葬', '死亡']
            if any(keyword in content for keyword in info_keywords):
                info_count += 1

        return info_count / len(player['speech_records'])

    def _analyze_speech(self, player, speech):
        """分析发言，更新可疑特征"""
        # 检查矛盾
        if player['speech_records']:
            for previous_speech in player['speech_records'][:-1]:  # 排除当前发言
                if self._check_contradiction(previous_speech['content'], speech):
                    player['suspicious_features']['contradictions'] += 1

        # 检查信息分享
        if player['claimed_role'] in ['fortune_teller', 'empath', 'undertaker']:
            if not self._contains_role_specific_info(speech, player['claimed_role']):
                player['suspicious_features']['information_sharing'] -= 0.1
            else:
                player['suspicious_features']['information_sharing'] += 0.2

        # 检查角色冲突
        if '我是' in speech or '我的角色' in speech:
            role_mentioned = None
            role_keywords = {
                'fortune_teller': ['预言家'],
                'empath': ['灵媒', '共情者'],
                'undertaker': ['送葬者'],
                'monk': ['僧侣'],
                'slayer': ['杀手', '猎人'],
                'mayor': ['市长'],
                'soldier': ['士兵']
            }

            for role, keywords in role_keywords.items():
                if any(keyword in speech for keyword in keywords):
                    role_mentioned = role
                    break

            if role_mentioned and player['claimed_role'] and role_mentioned != player['claimed_role']:
                player['suspicious_features']['role_conflicts'] += 1

    def _analyze_vote(self, player, target_id):
        """分析投票，更新可疑特征"""
        # 检查是否投票给被确认为好人的玩家
        target = next((p for p in self.players if p['id'] == target_id), None)
        if target and self._is_likely_good(target):
            player['suspicious_features']['voting_pattern'] += 0.2

        # 检查是否与大多数人投票一致
        if self.day > 0:
            # 获取当天所有投票
            day_votes = [vote_target for p in self.players
                         for day, vote_target in p['votes_given'] if day == self.day]

            if day_votes:
                most_common_vote = max(set(day_votes), key=day_votes.count)
                if target_id != most_common_vote:
                    player['suspicious_features']['voting_pattern'] += 0.1
                else:
                    player['suspicious_features']['voting_pattern'] -= 0.1

                    # 确保得分在0-1范围内
                    player['suspicious_features']['voting_pattern'] = max(0, min(1, player['suspicious_features'][
                        'voting_pattern']))

    def _check_contradiction(self, speech1, speech2):
        """检查两次发言是否矛盾"""
        # 简化实现，检查角色声明冲突
        role_keywords = {
            'fortune_teller': ['预言家'],
            'empath': ['灵媒', '共情者'],
            'undertaker': ['送葬者'],
            'monk': ['僧侣'],
            'slayer': ['杀手', '猎人'],
            'mayor': ['市长'],
            'soldier': ['士兵']
        }

        role1 = None
        role2 = None

        for role, keywords in role_keywords.items():
            for keyword in keywords:
                if keyword in speech1 and ('我是' in speech1 or '我的角色' in speech1):
                    role1 = role
                if keyword in speech2 and ('我是' in speech2 or '我的角色' in speech2):
                    role2 = role

        return role1 and role2 and role1 != role2

    def _contains_role_specific_info(self, speech, role):
        """检查发言是否包含角色特定信息"""
        speech = speech.lower()

        if role == 'fortune_teller':
            return '查验' in speech or '是好人' in speech or '是狼人' in speech or '是邪恶' in speech
        elif role == 'empath':
            return '邻居' in speech or '相邻' in speech or '邪恶数量' in speech
        elif role == 'undertaker':
            return '死亡' in speech or '处决' in speech or '角色是' in speech

        return False

    def _is_likely_good(self, player):
        """判断玩家是否可能是好人"""
        # 简化实现
        if player['claimed_role'] in ['fortune_teller', 'empath', 'undertaker']:
            if player['suspicious_score'] < 0.4:
                return True
        return False

    def _explain_prediction(self, player):
        """解释为什么预测该玩家是狼队"""
        print(f"\n玩家{player['name']}可疑点:")

        # 主要可疑点
        suspicious_features = player['suspicious_features']

        if suspicious_features['contradictions'] > 0:
            print(f"- 发言自相矛盾: {suspicious_features['contradictions']}次")

        if player['claimed_role'] in ['fortune_teller', 'empath', 'undertaker'] and suspicious_features[
            'information_sharing'] < 0.5:
            print(f"- 声称为信息型角色但分享信息不足")

        if suspicious_features['voting_pattern'] > 0.6:
            print(f"- 投票模式可疑: 可能投票给好人或与大多数人投票不一致")

        if suspicious_features['role_conflicts'] > 0:
            print(f"- 角色声明冲突: {suspicious_features['role_conflicts']}次")

        if suspicious_features['behavioral_changes'] > 0:
            print(f"- 行为突然改变: {suspicious_features['behavioral_changes']}次")

        # 如果没有明显可疑点但综合分数高
        if all(v <= 0.3 for v in suspicious_features.values()) and player['suspicious_score'] > 0.5:
            print(f"- 综合行为表现可疑，但没有明显单一可疑点")

    def _suggest_alternative_combinations(self):
        """提供备选的狼坑组合"""
        # 按可疑度排序玩家
        sorted_players = sorted(self.players, key=lambda p: p['suspicious_score'], reverse=True)
        alive_suspicious = [p for p in sorted_players if p['alive'] and p['suspicious_score'] > 0.4]

        if len(alive_suspicious) >= 5:
            print("\n备选狼坑组合:")

            # 备选组合1: 2,3,4号可疑玩家
            alt1 = alive_suspicious[1:4]
            print(f"备选组合1: 恶魔-{alt1[0]['name']}, 爪牙-{alt1[1]['name']}和{alt1[2]['name']}")

            # 备选组合2: 0,2,4号可疑玩家
            alt2 = [alive_suspicious[0], alive_suspicious[2], alive_suspicious[4]]
            print(f"备选组合2: 恶魔-{alt2[0]['name']}, 爪牙-{alt2[1]['name']}和{alt2[2]['name']}")

    def _update_model(self, rewards):
        """使用策略梯度更新模型"""
        # 这里是一个简化的策略梯度更新
        # 实际应用中可能需要更复杂的算法如PPO或A2C

        # 计算折扣奖励
        discounted_rewards = []
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + 0.99 * cumulative
            discounted_rewards.insert(0, cumulative)

        # 标准化奖励
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # 计算损失
        loss = -torch.sum(discounted_rewards)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 使用示例
def main():
    """主函数示例"""
    # 创建分析器
    analyzer = BTCTAnalyzer(num_players=12)

    # 可选：训练模型
    # analyzer.train_model(num_episodes=100)

    # 初始化游戏
    player_names = ["玩家1", "玩家2", "玩家3", "玩家4", "玩家5", "玩家6",
                    "玩家7", "玩家8", "玩家9", "玩家10", "玩家11", "玩家12"]
    analyzer.initialize_game(player_names)

    # 游戏进行中...
    # 第一天
    analyzer.next_day()

    # 更新玩家信息 - 发言阶段
    player_updates = [
        {'id': 0, 'speech': "我是预言家，昨晚查验了玩家3是好人"},
        {'id': 1, 'speech': "我是村民，没有特殊能力"},
        {'id': 2, 'speech': "我是村民，支持1号预言家"},
        # ... 更多发言
    ]
    analyzer.update_player_info(player_updates)

    # 投票阶段
    vote_updates = [
        {'id': 0, 'vote_target': 3},
        {'id': 1, 'vote_target': 3},
        # ... 更多投票
    ]
    analyzer.update_player_info(vote_updates)

    # 分析当前狼坑
    analyzer.analyze_wolf_team()

    # 处决结果
    analyzer.record_execution(3)

    # 夜间死亡
    analyzer.record_night_death(8)

    # 后续天数...
    # 第二天
    analyzer.next_day()

    # 更新玩家信息
    player_updates = [
        {'id': 0, 'speech': "我昨晚查验了玩家10，是邪恶阵营"},
        # ... 更多发言
    ]
    analyzer.update_player_info(player_updates)

    # 再次分析狼坑
    analyzer.analyze_wolf_team()

    # 游戏继



    """主函数示例(续)"""
    # 继续第三天...

    # 更新玩家信息
    player_updates = [
        {'id': 0, 'speech': "我昨晚查验了玩家12，是邪恶阵营，我们已经找到两个狼人了"},
        {'id': 1, 'speech': "我支持1号预言家的判断，应该处决12号"},
        {'id': 2, 'speech': "我同意1号的看法，10号和12号都是狼人"},
        {'id': 5, 'speech': "我的邻座有0个邪恶玩家，说明我旁边都是好人"},
        {'id': 6, 'speech': "我认为12号很可疑，建议今天处决他"},
        {'id': 7, 'speech': "我怀疑10号和12号是狼队，但还有一个没找到"},
        {'id': 11, 'speech': "1号预言家说的对，应该处决12号"},
        {'id': 12, 'speech': "我不是狼人，1号才是恶魔，他在骗大家"}
    ]
    analyzer.update_player_info(player_updates)

    # 投票
    vote_updates = [
        {'id': 0, 'vote_target': 12},
        {'id': 1, 'vote_target': 12},
        {'id': 2, 'vote_target': 12},
        {'id': 5, 'vote_target': 12},
        {'id': 6, 'vote_target': 12},
        {'id': 7, 'vote_target': 12},
        {'id': 11, 'vote_target': 12},
        {'id': 12, 'vote_target': 0}
    ]
    analyzer.update_player_info(vote_updates)

    # 处决
    analyzer.record_execution(12)

    # 分析狼坑
    analyzer.analyze_wolf_team()

    # 游戏结束时保存模型
    analyzer.save_model()


class DynamicEvilTeamPredictor:
    """多模型集成的动态恶魔阵营预测器"""

    def __init__(self, num_players=12):
        self.num_players = num_players

        # 初始化多个模型
        self.models = {
            'base_rl': ActorCritic(num_players, 15),  # 基础强化学习模型
            'behavior': None,  # 行为分析模型(将在后续初始化)
            'social_network': None,  # 社交网络分析模型
            'vote_pattern': None  # 投票模式分析模型
        }

        # 模型权重(初始均等权重)
        self.model_weights = {
            'base_rl': 0.25,
            'behavior': 0.25,
            'social_network': 0.25,
            'vote_pattern': 0.25
        }

        # 初始化额外模型
        self._initialize_models()

        # 历史预测
        self.history = []

        # 性能统计
        self.performance = {model: [] for model in self.models}

    def _initialize_models(self):
        """初始化额外的预测模型"""
        # 行为分析模型
        self.models['behavior'] = self._create_behavior_model()

        # 社交网络分析模型
        self.models['social_network'] = self._create_social_network_model()

        # 投票模式分析模型
        self.models['vote_pattern'] = self._create_vote_pattern_model()

    def _create_behavior_model(self):
        """创建基于行为特征的分析模型"""
        # 这里使用简化的神经网络模型
        model = nn.Sequential(
            nn.Linear(self.num_players * 15, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_players * 3)  # 输出每个玩家作为恶魔/爪牙的概率
        )
        return model

    def _create_social_network_model(self):
        """创建社交网络分析模型"""

        # 这里使用简单的图卷积网络
        class SimpleGCN(nn.Module):
            def __init__(self, num_nodes):
                super(SimpleGCN, self).__init__()
                self.conv1 = nn.Linear(num_nodes, 32)
                self.conv2 = nn.Linear(32, 16)
                self.fc = nn.Linear(16 * num_nodes, num_nodes * 3)

            def forward(self, adj_matrix, features):
                # 简化的图卷积操作
                x = torch.matmul(adj_matrix, features)
                x = torch.relu(self.conv1(x))
                x = torch.matmul(adj_matrix, x)
                x = torch.relu(self.conv2(x))
                x = x.view(1, -1)
                x = self.fc(x)
                return x

        return SimpleGCN(self.num_players)

    def _create_vote_pattern_model(self):
        """创建投票模式分析模型"""

        # 使用LSTM模型分析投票序列
        class VotePatternLSTM(nn.Module):
            def __init__(self, num_players):
                super(VotePatternLSTM, self).__init__()
                self.num_players = num_players
                self.lstm = nn.LSTM(
                    input_size=num_players * 2,  # 投票矩阵
                    hidden_size=32,
                    num_layers=2,
                    batch_first=True
                )
                self.fc = nn.Linear(32, num_players * 3)

            def forward(self, vote_sequences):
                # vote_sequences: [batch, seq_len, players*2]
                output, (hidden, _) = self.lstm(vote_sequences)
                pred = self.fc(hidden[-1])
                return pred

        return VotePatternLSTM(self.num_players)

    def predict(self, game_state):
        """融合多个模型进行预测"""
        # 提取当前游戏状态特征
        observation = self._extract_features(game_state)

        # 获取各模型的预测
        predictions = {}
        confidence_scores = {}

        # 基础强化学习模型预测
        with torch.no_grad():
            predictions['base_rl'] = self.models['base_rl'].act(observation)
            confidence_scores['base_rl'] = 1.0  # 默认置信度

        # 行为分析模型预测
        if self.models['behavior']:
            behavior_pred = self._predict_with_behavior_model(observation)
            predictions['behavior'] = behavior_pred
            confidence_scores['behavior'] = self._calculate_confidence('behavior', game_state)

        # 社交网络模型预测
        if self.models['social_network'] and 'social_graph' in game_state:
            social_pred = self._predict_with_social_network(game_state['social_graph'])
            predictions['social_network'] = social_pred
            confidence_scores['social_network'] = self._calculate_confidence('social_network', game_state)

        # 投票模式模型预测
        if self.models['vote_pattern'] and 'vote_history' in game_state and len(game_state['vote_history']) > 0:
            vote_pred = self._predict_with_vote_pattern(game_state['vote_history'])
            predictions['vote_pattern'] = vote_pred
            confidence_scores['vote_pattern'] = self._calculate_confidence('vote_pattern', game_state)

        # 动态调整模型权重
        self._update_model_weights(confidence_scores)

        # 加权融合预测结果
        final_prediction = self._ensemble_predictions(predictions)

        # 记录本次预测
        self._record_prediction(final_prediction, game_state)

        return final_prediction

    def _extract_features(self, game_state):
        """从游戏状态中提取特征"""
        # 简化实现，假设game_state包含已经格式化好的观察向量
        if 'observation' in game_state:
            return game_state['observation']

        # 返回默认观察空间
        return np.zeros((self.num_players, 15), dtype=np.float32)

    def _predict_with_behavior_model(self, observation):
        """使用行为模型预测"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).view(1, -1)
            output = self.models['behavior'](obs_tensor)
            probs = output.view(3, self.num_players)

            # 获取每个角色的最可能玩家
            demon_idx = torch.argmax(probs[0]).item()
            minion1_idx = torch.argmax(probs[1]).item()
            minion2_idx = torch.argmax(probs[2]).item()

            # 确保不重复
            if minion1_idx == demon_idx:
                probs[1, demon_idx] = -float('inf')
                minion1_idx = torch.argmax(probs[1]).item()

            if minion2_idx in [demon_idx, minion1_idx]:
                probs[2, demon_idx] = -float('inf')
                probs[2, minion1_idx] = -float('inf')
                minion2_idx = torch.argmax(probs[2]).item()

            return [demon_idx, minion1_idx, minion2_idx]

    def _predict_with_social_network(self, social_graph):
        """使用社交网络模型预测"""
        with torch.no_grad():
            # 创建邻接矩阵
            adj_matrix = torch.FloatTensor(social_graph['adjacency_matrix'])
            features = torch.FloatTensor(social_graph['node_features'])

            # 预测
            output = self.models['social_network'](adj_matrix, features)
            probs = output.view(3, self.num_players)

            # 获取每个角色的最可能玩家
            demon_idx = torch.argmax(probs[0]).item()
            minion1_idx = torch.argmax(probs[1]).item()
            minion2_idx = torch.argmax(probs[2]).item()

            # 确保不重复
            if minion1_idx == demon_idx:
                probs[1, demon_idx] = -float('inf')
                minion1_idx = torch.argmax(probs[1]).item()

            if minion2_idx in [demon_idx, minion1_idx]:
                probs[2, demon_idx] = -float('inf')
                probs[2, minion1_idx] = -float('inf')
                minion2_idx = torch.argmax(probs[2]).item()

            return [demon_idx, minion1_idx, minion2_idx]

    def _predict_with_vote_pattern(self, vote_history):
        """使用投票模式模型预测"""
        with torch.no_grad():
            # 格式化投票历史
            vote_seq = self._format_vote_history(vote_history)
            vote_tensor = torch.FloatTensor(vote_seq).unsqueeze(0)  # 添加batch维度

            # 预测
            output = self.models['vote_pattern'](vote_tensor)
            probs = output.view(3, self.num_players)

            # 获取每个角色的最可能玩家
            demon_idx = torch.argmax(probs[0]).item()
            minion1_idx = torch.argmax(probs[1]).item()
            minion2_idx = torch.argmax(probs[2]).item()

            # 确保不重复
            if minion1_idx == demon_idx:
                probs[1, demon_idx] = -float('inf')
                minion1_idx = torch.argmax(probs[1]).item()

            if minion2_idx in [demon_idx, minion1_idx]:
                probs[2, demon_idx] = -float('inf')
                probs[2, minion1_idx] = -float('inf')
                minion2_idx = torch.argmax(probs[2]).item()

            return [demon_idx, minion1_idx, minion2_idx]

    def _format_vote_history(self, vote_history):
        """格式化投票历史为模型输入"""
        # 简化实现，将投票历史转换为序列
        seq_len = len(vote_history)
        vote_seq = np.zeros((seq_len, self.num_players * 2))

        for i, votes in enumerate(vote_history):
            # 第一部分：谁投票
            for voter, target in votes:
                vote_seq[i, voter] = 1

            # 第二部分：投给谁
            for voter, target in votes:
                vote_seq[i, self.num_players + target] = 1

        return vote_seq

    def _calculate_confidence(self, model_name, game_state):
        """计算模型预测的置信度"""
        # 根据游戏阶段和模型特性动态调整置信度

        if model_name == 'behavior':
            # 行为模型在游戏后期更可靠
            day = game_state.get('day', 0)
            return min(1.0, 0.5 + day * 0.1)

        elif model_name == 'social_network':
            # 社交网络模型需要足够的互动数据
            interactions = len(game_state.get('speech_history', []))
            return min(1.0, interactions / 20)

        elif model_name == 'vote_pattern':
            # 投票模式模型需要足够的投票数据
            votes = len(game_state.get('vote_history', []))
            return min(1.0, votes / 3)

        return 1.0  # 默认置信度

    def _update_model_weights(self, confidence_scores):
        """根据置信度更新模型权重"""
        # 计算新权重
        total_confidence = sum(confidence_scores.values())
        if total_confidence > 0:
            for model in self.model_weights:
                if model in confidence_scores:
                    self.model_weights[model] = confidence_scores[model] / total_confidence

    def _ensemble_predictions(self, predictions):
        """集成多个模型的预测结果"""
        # 每个玩家的得分
        player_scores = {i: {'demon': 0, 'minion': 0} for i in range(self.num_players)}

        # 加权累计得分
        for model, pred in predictions.items():
            weight = self.model_weights.get(model, 0)

            # 恶魔得分
            player_scores[pred[0]]['demon'] += weight

            # 爪牙得分
            for minion_idx in pred[1:3]:
                player_scores[minion_idx]['minion'] += weight / 2  # 两个爪牙平分权重

        # 选择得分最高的玩家作为恶魔
        demon = max(range(self.num_players), key=lambda i: player_scores[i]['demon'])

        # 选择除恶魔外得分最高的两个玩家作为爪牙
        minion_candidates = {i: score['minion'] for i, score in player_scores.items() if i != demon}
        minions = sorted(minion_candidates.keys(), key=lambda i: minion_candidates[i], reverse=True)[:2]

        return [demon] + minions

    def _record_prediction(self, prediction, game_state):
        """记录预测结果"""
        self.history.append({
            'day': game_state.get('day', 0),
            'prediction': prediction,
            'weights': self.model_weights.copy()
        })

    def update_model(self, game_state, feedback=None):
        """使用新数据更新模型"""
        # 这里可以实现在线学习或微调逻辑
        # 简化实现，仅作为示例
        if feedback and 'true_evil_team' in feedback:
            # 如果有真实标签，可以计算性能并更新模型
            true_team = feedback['true_evil_team']

            # 更新基础RL模型
            # 简化实现，实际应用中应使用正确的RL更新方法
            pass

            # 更新其他模型
            # 这里应该使用适当的方法更新每个模型
            pass

    def export_analysis(self):
        """导出分析结果"""
        # 返回最新预测及其理由
        if not self.history:
            return {"error": "没有预测历史"}

        latest = self.history[-1]

        # 计算预测变化
        changes = []
        if len(self.history) > 1:
            prev = self.history[-2]
            for i, player in enumerate(latest['prediction']):
                if player not in prev['prediction']:
                    role = "恶魔" if i == 0 else "爪牙"
                    changes.append(f"新增{role}: 玩家{player}")

        # 返回分析结果
        return {
            "day": latest['day'],
            "prediction": {
                "demon": latest['prediction'][0],
                "minions": latest['prediction'][1:3]
            },
            "model_weights": latest['weights'],
            "changes": changes
        }


class RealTimeAnalysisSystem:
    """实时狼坑分析系统"""

    def __init__(self, num_players=12):
        self.num_players = num_players

        # 初始化预测器
        self.predictor = DynamicEvilTeamPredictor(num_players)

        # 游戏状态
        self.game_state = {
            'day': 0,
            'players': [],
            'alive_count': num_players,
            'observation': np.zeros((num_players, 15), dtype=np.float32),
            'speech_history': [],
            'vote_history': [],
            'death_history': [],
            'social_graph': {
                'adjacency_matrix': np.zeros((num_players, num_players)),
                'node_features': np.zeros((num_players, 10))
            }
        }

        # 分析结果历史
        self.analysis_history = []

        # 事件处理器
        self.event_handlers = {
            'speech': self._handle_speech_event,
            'vote': self._handle_vote_event,
            'execution': self._handle_execution_event,
            'night_death': self._handle_night_death_event,
            'role_claim': self._handle_role_claim_event,
            'skill_info': self._handle_skill_info_event
        }

    def initialize_game(self, player_names):
        """初始化新游戏"""
        # 重置游戏状态
        self.game_state = {
            'day': 0,
            'players': [],
            'alive_count': self.num_players,
            'observation': np.zeros((self.num_players, 15), dtype=np.float32),
            'speech_history': [],
            'vote_history': [],
            'death_history': [],
            'social_graph': {
                'adjacency_matrix': np.zeros((self.num_players, self.num_players)),
                'node_features': np.zeros((self.num_players, 10))
            }
        }

        # 初始化玩家
        for i, name in enumerate(player_names):
            self.game_state['players'].append({
                'id': i,
                'name': name,
                'alive': True,
                'claimed_role': None,
                'suspicious_score': 0.5,
                'feature_vector': np.zeros(15)
            })

        # 重置分析历史
        self.analysis_history = []

        print(f"游戏初始化完成，{len(player_names)}名玩家已准备就绪")

    def process_event(self, event_type, event_data):
        """处理游戏事件"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type](event_data)

            # 更新特征向量
            self._update_feature_vectors()

            # 如果是重要事件，执行分析
            if event_type in ['vote', 'execution', 'night_death', 'role_claim']:
                self.analyze_wolf_team()

    def next_day(self):
        """进入下一天"""
        self.game_state['day'] += 1
        print(f"\n=== 第{self.game_state['day']}天 ===")

        # 执行分析
        self.analyze_wolf_team()

    def analyze_wolf_team(self):
        """分析狼队阵容"""
        day = self.game_state['day']
        print(f"\n=== 第{day}天狼人阵营分析 ===")

        # 使用预测器进行预测
        prediction = self.predictor.predict(self.game_state)

        # 获取预测的玩家名称
        demon_id = prediction[0]
        minion1_id, minion2_id = prediction[1:3]

        demon = self._get_player_by_id(demon_id)
        minion1 = self._get_player_by_id(minion1_id)
        minion2 = self._get_player_by_id(minion2_id)

        # 输出预测结果
        print("\n当前预测的狼坑组合:")
        print(f"恶魔: 玩家{demon['name']} (ID: {demon_id}, 可疑度: {demon['suspicious_score']:.2f})")
        print(f"爪牙1: 玩家{minion1['name']} (ID: {minion1_id}, 可疑度: {minion1['suspicious_score']:.2f})")
        print(f"爪牙2: 玩家{minion2['name']} (ID: {minion2_id}, 可疑度: {minion2['suspicious_score']:.2f})")

        # 分析模型贡献
        analysis = self.predictor.export_analysis()
        print("\n模型贡献:")
        for model, weight in analysis['model_weights'].items():
            print(f"{model}: {weight:.2f}")

        # 分析变化
        if analysis['changes']:
            print("\n预测变化:")
            for change in analysis['changes']:
                print(f"- {change}")

        # 保存分析结果
        self.analysis_history.append({
            'day': day,
            'prediction': prediction,
            'analysis': analysis
        })

        return prediction

    def export_report(self, day=None):
        """导出特定天数的分析报告"""
        if not self.analysis_history:
            return {"error": "没有分析历史"}

        # 获取指定天数的分析结果，默认为最新
        if day is None:
            analysis = self.analysis_history[-1]
        else:
            matches = [a for a in self.analysis_history if a['day'] == day]
            if not matches:
                return {"error": f"没有第{day}天的分析结果"}
            analysis = matches[-1]  # 取最新的分析

        return {
            "day": analysis['day'],
            "prediction": {
                "demon": {
                    "id": analysis['prediction'][0],
                    "name": self._get_player_by_id(analysis['prediction'][0])['name'],
                    "suspicious_score": self._get_player_by_id(analysis['prediction'][0])['suspicious_score']
                },
                "minions": [
                    {
                        "id": analysis['prediction'][1],
                        "name": self._get_player_by_id(analysis['prediction'][1])['name'],
                        "suspicious_score": self._get_player_by_id(analysis['prediction'][1])['suspicious_score']
                    },
                    {
                        "id": analysis['prediction'][2],
                        "name": self._get_player_by_id(analysis['prediction'][2])['name'],
                        "suspicious_score": self._get_player_by_id(analysis['prediction'][2])['suspicious_score']
                    }
                ]
            },
            "model_weights": analysis['analysis']['model_weights'],
            "changes": analysis['analysis'].get('changes', []),
            "game_state": {
                "alive_count": self._count_alive_players(),
                "vote_summary": self._get_vote_summary()
            }
        }

    def _handle_speech_event(self, data):
        """处理发言事件"""
        player_id = data['player_id']
        content = data['content']

        # 记录发言
        self.game_state['speech_history'].append({
            'day': self.game_state['day'],
            'player_id': player_id,
            'content': content
        })

        # 更新社交图谱
        self._update_social_graph_from_speech(player_id, content)

        # 分析发言内容
        self._analyze_speech_content(player_id, content)

    def _handle_vote_event(self, data):
        """处理投票事件"""
        voter_id = data['voter_id']
        target_id = data['target_id']

        # 记录投票
        day = self.game_state['day']

        # 如果这是该天的第一次投票，创建新列表
        if len(self.game_state['vote_history']) <= day:
            self.game_state['vote_history'].append([])

        self.game_state['vote_history'][day].append((voter_id, target_id))

        # 更新社交图谱
        self._update_social_graph_from_vote(voter_id, target_id)

    def _handle_execution_event(self, data):
        """处理处决事件"""
        player_id = data['player_id']

        # 更新玩家状态
        player = self._get_player_by_id(player_id)
        if player:
            player['alive'] = False
            self.game_state['alive_count'] -= 1

        # 记录死亡
        self.game_state['death_history'].append({
            'day': self.game_state['day'],
            'player_id': player_id,
            'type': 'execution'
        })

    def _handle_night_death_event(self, data):
        """处理夜间死亡事件"""
        player_id = data['player_id']

        # 更新玩家状态
        player = self._get_player_by_id(player_id)
        if player:
            player['alive'] = False
            self.game_state['alive_count'] -= 1

        # 记录死亡
        self.game_state['death_history'].append({
            'day': self.game_state['day'],
            'player_id': player_id,
            'type': 'night'
        })

    def _handle_role_claim_event(self, data):
        """处理角色声明事件"""
        player_id = data['player_id']
        role = data['role']

        # 更新玩家角色声明
        player = self._get_player_by_id(player_id)
        if player:
            player['claimed_role'] = role

    def _handle_skill_info_event(self, data):
        """处理技能信息事件"""
        # 这个事件处理器用于接收玩家使用特殊技能获得的信息
        # 例如预言家的查验结果、灵媒的邻座信息等
        pass

    def _get_player_by_id(self, player_id):
        """根据ID获取玩家信息"""
        for player in self.game_state['players']:
            if player['id'] == player_id:
                return player
        return None

    def _count_alive_players(self):
        """统计存活玩家数量"""
        return sum(1 for player in self.game_state['players'] if player['alive'])

    def _get_vote_summary(self):
        """获取投票摘要"""
        if not self.game_state['vote_history']:
            return {}

        latest_votes = self.game_state['vote_history'][-1]
        summary = {}

        for voter, target in latest_votes:
            if target not in summary:
                summary[target] = []
            summary[target].append(voter)

        return summary

    def _update_feature_vectors(self):
        """更新所有玩家的特征向量"""
        for player in self.game_state['players']:
            self._update_player_feature_vector(player)

    def _update_player_feature_vector(self, player):
        """更新单个玩家的特征向量"""
        # 简化实现，实际应用中需要更复杂的特征工程
        feature_vector = np.zeros(15)

        # 存活状态
        feature_vector[0] = 1.0 if player['alive'] else 0.0

        # 角色声明（编码）
        if player['claimed_role']:
            if player['claimed_role'] in ['fortune_teller', 'empath', 'undertaker']:
                feature_vector[3] = 1.0
            elif player['claimed_role'] in ['monk', 'slayer', 'soldier']:
                feature_vector[3] = 2.0

        # 可疑度
        feature_vector[4] = player['suspicious_score']

        # 更新玩家特征向量
        player['feature_vector'] = feature_vector

        # 更新观察矩阵中的对应行
        self.game_state['observation'][player['id']] = feature_vector

    def _update_social_graph_from_speech(self, speaker_id, content):
        """基于发言更新社交图谱"""
        # 根据发言内容分析玩家之间的关系
        for player in self.game_state['players']:
            if player['id'] == speaker_id:
                continue

            # 检查发言中是否提到该玩家（简化实现）
            if player['name'] in content:
                # 检查是正面还是负面提及
                positive_words = ['支持', '赞同', '好人', '信任']
                negative_words = ['怀疑', '不信', '狼人', '邪恶', '恶魔']

                sentiment = 0
                for word in positive_words:
                    if word in content and player['name'] in content.split(word)[1]:
                        sentiment += 1
                        break

                for word in negative_words:
                    if word in content and player['name'] in content.split(word)[1]:
                        sentiment -= 1
                        break

                # 更新邻接矩阵
                self.game_state['social_graph']['adjacency_matrix'][speaker_id][player['id']] += sentiment

    def _update_social_graph_from_vote(self, voter_id, target_id):
        """基于投票更新社交图谱"""
        # 投票给某人通常表示怀疑/负面关系
        self.game_state['social_graph']['adjacency_matrix'][voter_id][target_id] -= 1

    def _analyze_speech_content(self, player_id, content):
        """分析发言内容，更新玩家可疑度"""
        player = self._get_player_by_id(player_id)
        if not player:
            return

        # 检查可疑模式
        suspicious_patterns = [
            '我不确定', '可能是', '也许', '不记得了',  # 回避性表达
            '我绝对不是', '我肯定是好人', '我一定不是狼人',  # 过度否认
            '相信我', '不要怀疑我'  # 过度寻求信任
        ]

        # 计算可疑度增量
        suspicion_delta = 0
        for pattern in suspicious_patterns:
            if pattern in content:
                suspicion_delta += 0.05

        # 信息分享减少可疑度
        info_keywords = ['查验', '预言', '灵媒', '邻座', '送葬', '死亡']
        if any(keyword in content for keyword in info_keywords) and player['claimed_role']:
            suspicion_delta -= 0.1

        # 更新可疑度分数
        player['suspicious_score'] = max(0, min(1, player['suspicious_score'] + suspicion_delta))


# 实时可视化和分析界面（这部分在实际应用中可以使用前端框架实现）
def create_visualization_interface(analyzer):
    """创建可视化分析界面"""
    print("=== 血染钟楼实时狼坑分析系统 ===")
    print("- 可视化界面模拟 -")

    # 获取最新分析结果
    latest_analysis = analyzer.export_report()

    # 打印预测结果
    print("\n当前预测恶魔阵营:")
    print(
        f"恶魔: {latest_analysis['prediction']['demon']['name']} (可疑度: {latest_analysis['prediction']['demon']['suspicious_score']:.2f})")
    print(
        f"爪牙1: {latest_analysis['prediction']['minions'][0]['name']} (可疑度: {latest_analysis['prediction']['minions'][0]['suspicious_score']:.2f})")
    print(
        f"爪牙2: {latest_analysis['prediction']['minions'][1]['name']} (可疑度: {latest_analysis['prediction']['minions'][1]['suspicious_score']:.2f})")

    # 打印模型贡献度
    print("\n模型贡献度:")
    for model, weight in latest_analysis['model_weights'].items():
        print(f"{model}: {'=' * int(weight * 20)} {weight:.2f}")

    # 打印游戏状态
    print(f"\n游戏状态: 第{latest_analysis['day']}天, {latest_analysis['game_state']['alive_count']}名玩家存活")

    # 打印投票摘要
    print("\n最近投票情况:")
    for target, voters in latest_analysis['game_state']['vote_summary'].items():
        target_name = analyzer._get_player_by_id(target)['name']
        voter_names = [analyzer._get_player_by_id(v)['name'] for v in voters]
        print(f"{target_name} 被 {', '.join(voter_names)} 投票")

