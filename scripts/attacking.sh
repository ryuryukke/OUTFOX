cd ../src/attacking
# OUTFOX attack
python outfox_attacking.py

# DIPPER attack on various models
python dipper_attacking.py --model_name chatgpt
python dipper_attacking.py --model_name flan_t5_xxl
python dipper_attacking.py --model_name text_davinci_003
