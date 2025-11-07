from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Plotting additional values in tensorboard.
    Values
        attack_on_tree
    """
    def __init__(self, env, verbose: int = 0):
        super().__init__(verbose)
        self.env = env

    def _on_rollout_end(self) -> None:
        count_total_reward_ego = self.env.base_env.env.env.env.env.env.count_total_reward_ego
        count_nut_reward_ego = self.env.base_env.env.env.env.env.env.count_nut_reward_ego
        num_attack_on_tree_ego = self.env.base_env.env.env.env.env.env.num_attack_on_tree_ego
        count_apple_reward_ego = self.env.base_env.env.env.env.env.env.count_apple_reward_ego
        count_single_attack_reward_ego = self.env.base_env.env.env.env.env.env.count_single_attack_reward_ego
        count_double_attack_reward_ego = self.env.base_env.env.env.env.env.env.count_double_attack_reward_ego
        num_get_nut_ego = self.env.base_env.env.env.env.env.env.num_get_nut_ego
        num_single_attack_hit_ego = self.env.base_env.env.env.env.env.env.num_single_attack_hit_ego
        num_double_attack_hit_ego = self.env.base_env.env.env.env.env.env.num_double_attack_hit_ego
        num_apple_tree_now_ego = self.env.base_env.env.env.env.env.env.num_apple_tree_now_ego

        count_total_reward_partner = self.env.base_env.env.env.env.env.env.count_total_reward_partner
        count_nut_reward_partner = self.env.base_env.env.env.env.env.env.count_nut_reward_partner
        num_attack_on_tree_partner = self.env.base_env.env.env.env.env.env.num_attack_on_tree_partner
        count_apple_reward_partner = self.env.base_env.env.env.env.env.env.count_apple_reward_partner
        count_single_attack_reward_partner = self.env.base_env.env.env.env.env.env.count_single_attack_reward_partner
        count_double_attack_reward_partner = self.env.base_env.env.env.env.env.env.count_double_attack_reward_partner
        num_get_nut_partner = self.env.base_env.env.env.env.env.env.num_get_nut_partner
        num_single_attack_hit_partner = self.env.base_env.env.env.env.env.env.num_single_attack_hit_partner
        num_double_attack_hit_partner = self.env.base_env.env.env.env.env.env.num_double_attack_hit_partner
        num_apple_tree_now_partner = self.env.base_env.env.env.env.env.env.num_apple_tree_now_partner

        total_reward = self.env.base_env.env.env.env.env.env.total_reward
        total_nut_reward = self.env.base_env.env.env.env.env.env.total_nut_reward
        total_num_attack_on_tree = self.env.base_env.env.env.env.env.env.total_num_attack_on_tree
        total_apple_reward = self.env.base_env.env.env.env.env.env.total_apple_reward
        total_single_attack_reward = self.env.base_env.env.env.env.env.env.total_single_attack_reward
        total_double_attack_reward = self.env.base_env.env.env.env.env.env.total_double_attack_reward
        total_num_get_nut = self.env.base_env.env.env.env.env.env.total_num_get_nut
        total_num_single_attack_hit = self.env.base_env.env.env.env.env.env.total_num_single_attack_hit
        total_num_double_attack_hit = self.env.base_env.env.env.env.env.env.total_num_double_attack_hit
        total_num_apple_tree_together = self.env.base_env.env.env.env.env.env.total_num_apple_tree_together

        self.logger.record("ego/total_reward", count_total_reward_ego)
        self.logger.record("ego/nut_reward", count_nut_reward_ego)
        self.logger.record("ego/num_attack_on_tree", num_attack_on_tree_ego)
        self.logger.record("ego/apple_reward", count_apple_reward_ego)
        self.logger.record("ego/single_attack_reward", count_single_attack_reward_ego)
        self.logger.record("ego/double_attack_reward", count_double_attack_reward_ego)
        self.logger.record("ego/num_get_nut", num_get_nut_ego)
        self.logger.record("ego/num_single_attack_hit", num_single_attack_hit_ego)
        self.logger.record("ego/num_double_attack_hit", num_double_attack_hit_ego)
        self.logger.record("ego/num_apple_tree_now", num_apple_tree_now_ego)

        self.logger.record("partner/total_reward", count_total_reward_partner)
        self.logger.record("partner/nut_reward", count_nut_reward_partner)
        self.logger.record("partner/num_attack_on_tree", num_attack_on_tree_partner)
        self.logger.record("partner/apple_reward", count_apple_reward_partner)
        self.logger.record("partner/single_attack_reward", count_single_attack_reward_partner)
        self.logger.record("partner/double_attack_reward", count_double_attack_reward_partner)
        self.logger.record("partner/num_get_nut", num_get_nut_partner)
        self.logger.record("partner/num_single_attack_hit", num_single_attack_hit_partner)
        self.logger.record("partner/num_double_attack_hit", num_double_attack_hit_partner)
        self.logger.record("partner/num_apple_tree_now", num_apple_tree_now_partner)

        self.logger.record("total/total_reward", total_reward)
        self.logger.record("total/nut_reward", total_nut_reward)
        self.logger.record("total/num_attack_on_tree", total_num_attack_on_tree)
        self.logger.record("total/apple_reward", total_apple_reward)
        self.logger.record("total/single_attack_reward", total_single_attack_reward)
        self.logger.record("total/double_attack_reward", total_double_attack_reward)
        self.logger.record("total/num_get_nut", total_num_get_nut)
        self.logger.record("total/num_single_attack_hit", total_num_single_attack_hit)
        self.logger.record("total/num_double_attack_hit", total_num_double_attack_hit)
        self.logger.record("total/num_apple_tree_together", total_num_apple_tree_together)
        
        return super()._on_rollout_end()  # almost same as return None
    
    def _on_step(self):
        return super()._on_step()
