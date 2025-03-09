#!/bin/bash

# Define colors
YELLOW="\e[33m"
GREEN="\e[32m"
RESET="\e[0m"

epochs=20
echo "Debug: Running with epochs=$epochs"

echo -e "${YELLOW}Running experiment 1: Standard Model${RESET}"
python sonnet_generation_ext.py -e $epochs

echo -e "${YELLOW}Running experiment 2: Standard Model${RESET} with Temperature 0.3"
python sonnet_generation_ext.py -e $epochs --temperature 0.3

echo -e "${YELLOW}Running experiment 3: Standard Model${RESET} with Temperature 0.5"
python sonnet_generation_ext.py -e $epochs --temperature 0.5

echo -e "${YELLOW}Running experiment 4: Standard Model${RESET} with Temperature 0.8"
python sonnet_generation_ext.py -e $epochs --temperature 0.8

echo -e "${YELLOW}Running experiment 5: Standard Model${RESET} with Temperature 1.0"
python sonnet_generation_ext.py -e $epochs --temperature 1.0

echo -e "${YELLOW}Running experiment 6: Standard Model${RESET} with Learning Rate 1e-3"
python sonnet_generation_ext.py -e $epochs --lr "1e-3"

echo -e "${YELLOW}Running experiment 7: Standard Model${RESET} with Learning Rate 1e-5"
python sonnet_generation_ext.py -e $epochs --lr "1e-5"

echo -e "${YELLOW}Running experiment 8: Standard Model${RESET} with Weight Decay 1e-3"
python sonnet_generation_ext.py -e $epochs --weight_decay "1e-3"

echo -e "${YELLOW}Running experiment 9: Standard Model${RESET} with Weight Decay 1e-2"
python sonnet_generation_ext.py -e $epochs --weight_decay "1e-2"

echo -e "${YELLOW}Running experiment 10: Standard Model${RESET} with Weight Decay 1e-4"
python sonnet_generation_ext.py -e $epochs --weight_decay "1e-4"


### Extension with Regularization techniques
epochs=35
echo -e "${YELLOW}Running experiment 1: LoRA Fine-Tuning${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --lr "2e-4"

echo -e "${YELLOW}Running experiment 2: QLoRA Fine-Tuning${RESET}"
python sonnet_generation_ext.py -e $epochs --qlora --lr "2e-4"

echo -e "${YELLOW}Running experiment 3: Dropout 0.2${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.05 --lr "2e-4"

echo -e "${YELLOW}Running experiment 4: Dropout 0.1${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.1 --lr "2e-4"

echo -e "${YELLOW}Running experiment 5: Dropout 0.3${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.15 --lr "2e-4"

echo -e "${YELLOW}Running experiment 6: Smart Lambda 1e-4${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-6" --lr "2e-4"

echo -e "${YELLOW}Running experiment 7: Smart Lambda 1e-3${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-5" --lr "2e-4"

echo -e "${YELLOW}Running experiment 8: Smart Lambda 1e-2${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-4" --lr "2e-4"

echo -e "${YELLOW}Running experiment 9: Jacobian Regularization Lambda 1e-4$ {RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --lr "2e-4"

echo -e "${YELLOW}Running experiment 10: Jacobian Regularization Lambda 1e-5${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-4" --lr "2e-4"

echo -e "${YELLOW}Running experiment 11: Jacobian Regularization Lambda 1e-2${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-2" --lr "2e-4"

echo -e "${YELLOW}Running experiment 12: Spectrum Analysis - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 25 --lr "2e-4"

echo -e "${YELLOW}Running experiment 13: Spectrum Analysis - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 50 --lr "2e-4"

echo -e "${YELLOW}Running experiment 14: Spectrum Analysis - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 75 --lr "2e-4"

echo -e "${YELLOW}Running experiment 15: LoRA + Spectrum - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --spectrum --top_percent 25  --lr "2e-4"

echo -e "${YELLOW}Running experiment 16: LoRA + Spectrum - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --spectrum --top_percent 50 --lr "2e-4"

echo -e "${YELLOW}Running experiment 17: LoRA + Spectrum - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --spectrum --top_percent 75 --lr "2e-4"

# Completion message
echo -e "${GREEN}All experiments completed successfully! 🚀${RESET}"