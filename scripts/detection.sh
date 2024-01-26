# OUTFOX detection without considering attacks on test set by various lms.
cd ../src/detection/
python outfox_detection_without_considering_attack.py --model_name chatgpt
python outfox_detection_without_considering_attack.py --model_name flan_t5_xxl
python outfox_detection_without_considering_attack.py --model_name text_davinci_003

# OUTFOX detection with considering various attacks on test set by various lms.
python outfox_detection_with_considering_attack.py --attacking_method outfox --model_name chatgpt
python outfox_detection_with_considering_attack.py --attacking_method outfox --model_name flan_t5_xxl
python outfox_detection_with_considering_attack.py --attacking_method outfox --model_name text_davinci_003

python outfox_detection_with_considering_attack.py --attacking_method paraphrase --model_name chatgpt
python outfox_detection_with_considering_attack.py --attacking_method paraphrase --model_name flan_t5_xxl
python outfox_detection_with_considering_attack.py --attacking_method paraphrase --model_name text_davinci_003

# Baseline detections on a dev set by various lms (and get thresholds).
python baseline_detection_dev.py --base_model_name chatgpt
python baseline_detection_dev.py --base_model_name flan_t5_xxl
python baseline_detection_dev.py --base_model_name text_davinci_003

# Baseline detections on a test set
python baseline_detection_test.py --base_model_name chatgpt
python baseline_detection_test.py --base_model_name flan_t5_xxl
python baseline_detection_test.py --base_model_name text_davinci_003
