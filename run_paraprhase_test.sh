#!/bin/bash
conda activate cs224n_dfp
# Define colors
YELLOW="\e[33m"
GREEN="\e[32m"
RESET="\e[0m"

epochs=7
model="gpt2"

# echo "Running with epochs=$epochs and model_size=$model"

echo -e "${YELLOW}Running experiment 1: QLORA${RESET}"
python paraphrase_detection_ext.py -e $epochs --model_size $model --qlora --lr "1e-4"

echo -e "${YELLOW}Running experiment 2: QLORA + SMART${RESET}"

python paraphrase_detection_ext.py -e $epochs --model_size $model --qlora --lr "1e-4" --smart --smart_lambda "1e-6"

echo -e "${YELLOW}Running experiment 3: QLORA + JACOBIAN${RESET}"
python paraphrase_detection_ext.py -e $epochs --model_size $model --qlora --lr "1e-4" --jacobian --jreg_lambda "1e-6"

epochs=5
echo -e "${YELLOW}Running experiment 4: Standard Experiment${RESET}"
python paraphrase_detection_ext.py -e $epochs --model_size $model --lr "2e-5"

echo -e "${YELLOW}Running experiment 5: Standard Experiment with Dropout .15${RESET}"
python paraphrase_detection_ext.py -e $epochs --model_size $model --lr "2e-5" --change_dropout --dropout 0.15

echo -e "${YELLOW}Running experiment 6: Standard Experiment with Dropout .2${RESET}"
python paraphrase_detection_ext.py -e $epochs --model_size $model --lr "2e-5" --change_dropout --dropout 0.15 --attn_dropout 0.15

# echo -e "${YELLOW}Running experiment 2: Standard Model${RESET} with Temperature 0.8"
# python sonnet_generation_ext.py -e $epochs --temperature 0.8 --lr "1e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 3: Standard Model with dropout 0.05${RESET} "
# python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.05 --lr "1e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 4: Standard Model with dropout 0.15${RESET}"
# python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.15 --lr "1e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 5: Standard Model${RESET}"
# python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.2 --lr "1e-4" --model_size $model

# ### Extension with Regularization techniques
# echo -e "${YELLOW}Running experiment 6: LoRA Fine-Tuning${RESET}"
# python sonnet_generation_ext.py -e $epochs --lora --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 7: QLoRA Fine-Tuning${RESET}"
# python sonnet_generation_ext.py -e $epochs --qlora --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 8: Smart Lambda 1e-6${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-6" --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 9: Smart Lambda 1e-5${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-5" --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 10: Smart Lambda 1e-4${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-4" --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 11: Jacobian Regularization Lambda 1e-6${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 12: Jacobian Regularization Lambda 1e-5${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-5" --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 13: Spectrum Analysis - Top 25%${RESET}"
# python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 25 --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 14: Spectrum Analysis - Top 50%${RESET}"
# python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 50 --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 15: Spectrum Analysis - Top 75%${RESET}"
# python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 75 --lr "2e-4" --model_size $model

# ### Extension of regularization with Jacobian and Smart
# echo -e "${YELLOW}Running experiment 16: Jacobian + Spectrum - Top 25%${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --spectrum --top_percent 25  --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 17: Jacobian + Spectrum - Top 50%${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --spectrum --top_percent 50 --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 18: Jacobian + Spectrum - Top 75%${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --spectrum --top_percent 75 --lr "2e-4" --model_size $model

# #### Extension of regularization with Smart and Spectrum
# echo -e "${YELLOW}Running experiment 19: Smart + Spectrum - Top 25%${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --spectrum --top_percent 25  --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 20: Smart + Spectrum - Top 50%${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --spectrum --top_percent 50 --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 21: Smart + Spectrum - Top 75%${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --spectrum --top_percent 75 --lr "2e-4" --model_size $model

# ##### Experiments LORA and regularizers
# echo -e "${YELLOW}Running experiment 22: Jacobian + LORA${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --lora --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 23: Smart + LORA${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --lora --lr "2e-4" --model_size $model

# ##### Experiments QLoRA and regularizers
# echo -e "${YELLOW}Running experiment 24: Jacobian + QLORA${RESET}"
# python sonnet_generation_ext.py -e $epochs --jacobian --qlora --lr "2e-4" --model_size $model

# echo -e "${YELLOW}Running experiment 25: Smart + QLORA${RESET}"
# python sonnet_generation_ext.py -e $epochs --smart --qlora --lr "2e-4" --model_size $model

# # Completion message
# echo -e "${GREEN}All experiments completed successfully! 🚀${RESET}"