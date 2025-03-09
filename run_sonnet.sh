#!/bin/bash

# Define colors
YELLOW="\e[33m"
GREEN="\e[32m"
RESET="\e[0m"

epochs=35
echo "Debug: Running with epochs=$epochs"

echo -e "${YELLOW}Running experiment 1: Standard Model${RESET}"
python sonnet_generation_ext.py -e $epochs

echo -e "${YELLOW}Running experiment 2: LoRA Fine-Tuning${RESET}"
python sonnet_generation_ext.py -e $epochs --lora

echo -e "${YELLOW}Running experiment 3: QLoRA Fine-Tuning${RESET}"
python sonnet_generation_ext.py -e $epochs --qlora

echo -e "${YELLOW}Running experiment 4: Dropout 0.2${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.2

echo -e "${YELLOW}Running experiment 5: Dropout 0.1${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.1

echo -e "${YELLOW}Running experiment 6: Dropout 0.3${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.3

echo -e "${YELLOW}Running experiment 7: Smart Lambda 1e-4${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-4"

echo -e "${YELLOW}Running experiment 8: Smart Lambda 1e-3${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-3"

echo -e "${YELLOW}Running experiment 9: Smart Lambda 1e-2${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-2"

echo -e "${YELLOW}Running experiment 10: Jacobian Regularization Lambda 1e-4${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-4"

echo -e "${YELLOW}Running experiment 11: Jacobian Regularization Lambda 1e-3${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-3"

echo -e "${YELLOW}Running experiment 12: Jacobian Regularization Lambda 1e-2${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-2"

echo -e "${YELLOW}Running experiment 13: Spectrum Analysis - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 25

echo -e "${YELLOW}Running experiment 14: Spectrum Analysis - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 50

echo -e "${YELLOW}Running experiment 15: Spectrum Analysis - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 75

echo -e "${YELLOW}Running experiment 16: LoRA + Spectrum - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --spectrum --top_percent 25

echo -e "${YELLOW}Running experiment 17: LoRA + Spectrum - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --spectrum --top_percent 50

echo -e "${YELLOW}Running experiment 18: LoRA + Spectrum - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --spectrum --top_percent 75

echo -e "${YELLOW}Running experiment 19: QLoRA + Spectrum - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --qlora --spectrum --top_percent 25

echo -e "${YELLOW}Running experiment 20: QLoRA + Spectrum - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --qlora --spectrum --top_percent 50

echo -e "${YELLOW}Running experiment 21: QLoRA + Spectrum - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --qlora --spectrum --top_percent 75

# Completion message
echo -e "${GREEN}All experiments completed successfully! 🚀${RESET}"