from configs.config import shared_configs

pretraining_config = shared_configs.get_pretraining_args()

print(pretraining_config)
print(pretraining_config.learning_rate)