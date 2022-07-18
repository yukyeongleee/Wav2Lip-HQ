# Changelog

## 1.0 @ Feb 28, 2022

- Open CodeTemplate repository 🔥

## 1.1 @ Mar 01, 2022

- Add ./lib 
- Add a method CreateModel

## 1.2 @ Mar 12, 2022

- Add ./scripts/download_ptnns.py
- Add .gitkeep files and update .gitignore

## 1.3 @ Mar 14, 2022

- Add a function, wandb alert, in ./scripts/train.py: 
- Add sys.path-related lines in ./train/scripts.py

# 1.4 @ Mar 18, 2022

- Add valid images of source and target
- Add ./lib/config.py and your_model/configs.yaml
- Add SingleFace Dataset
- Add a method, requires_grad(), in lib/utils.py
- Fix a method, load_checkpoints, to take "ckpt_path" as a argument
- Correct an error in L_reg
- Remove ./scripts/download_ptnns.py
- Remove option-related python files

# 1.5 @ Apr 25, 2022

- Move ./your_model/config.yaml to ./configs/train_configs.yaml
- Fix Configs.yaml to take ckpt_id and ckpt_step separately.
- Fix a method, load_checkpoint, to do not use "try and except".
- Fix a method, save_yaml, to save self.__dict__ instead of copying the original file. 

# 1.6 @ May 11, 2022

- Add README and Issues templates

# 1.7 @ May 21, 2022

- Add a method, setup_model, in the model_interface class
- Fix ./README.md, put ./README_BLACK.md to ./.gitgub 
- Fix ./YourModel to ./MyModel
- Fix a method, set_attribute, in the Config class
- Change name ptnn to ckpt
- Remove a function, wandb alert
- Remove ./ptnns, add ./packages
- Delete ./lib/Model_loader