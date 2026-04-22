import toml

cfg = toml.load(r"E:\kaiwu2025\hok_prelim\code\agent_ppo\conf\train_env_conf.toml")
print(cfg)
cfg["env_conf"]["map_fish"]["treasure_count"] = 0
print(cfg)