#!/bin/bash

# Define colors
YELLOW="\e[33m"
GREEN="\e[32m"
RESET="\e[0m"

epochs=20
echo "Debug: Running with epochs=$epochs"

echo -e "${YELLOW}Running experiment 1: Standard Model${RESET}"
python sonnet_generation_ext.py -e $epochs


echo -e "${YELLOW}Running experiment 2: Standard Model${RESET} with Temperature 0.5"
python sonnet_generation_ext.py -e $epochs --temperature 0.5 --lr "1e-3"

echo -e "${YELLOW}Running experiment 3: Standard Model${RESET} with Temperature 0.8"
python sonnet_generation_ext.py -e $epochs --temperature 0.8 --lr "1e-3"

echo -e "${YELLOW}Running experiment 4: Standard Model${RESET} with Temperature 1.0"
python sonnet_generation_ext.py -e $epochs --temperature 1.0 --lr "1e-3"

echo -e "${YELLOW}Running experiment 5: Standard Model${RESET} with Learning Rate 1e-3"
python sonnet_generation_ext.py -e $epochs --lr "1e-3"

echo -e "${YELLOW}Running experiment 6: Standard Model${RESET} with Weight Decay 1e-3"
python sonnet_generation_ext.py -e $epochs --weight_decay "1e-3" --lr "1e-3"

echo -e "${YELLOW}Running experiment 7: Standard Model${RESET} with Weight Decay 1e-2"
python sonnet_generation_ext.py -e $epochs --weight_decay "1e-2" --lr "1e-3"

echo -e "${YELLOW}Running experiment 8: Standard Model${RESET} with Weight Decay 1e-4"
python sonnet_generation_ext.py -e $epochs --weight_decay "1e-4" --lr "1e-3"


### Extension with Regularization techniques
epochs=35
echo -e "${YELLOW}Running experiment 1: LoRA Fine-Tuning${RESET}"
python sonnet_generation_ext.py -e $epochs --lora --lr "1e-3"

echo -e "${YELLOW}Running experiment 2: QLoRA Fine-Tuning${RESET}"
python sonnet_generation_ext.py -e $epochs --qlora --lr "1e-3"

echo -e "${YELLOW}Running experiment 3: Dropout 0.2${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.05 --lr "1e-3"

echo -e "${YELLOW}Running experiment 4: Dropout 0.1${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.1 --lr "1e-3"

echo -e "${YELLOW}Running experiment 5: Dropout 0.3${RESET}"
python sonnet_generation_ext.py -e $epochs --change_dropout --dropout 0.15 --lr "1e-3"

echo -e "${YELLOW}Running experiment 6: Smart Lambda 1e-4${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-6" --lr "1e-3"

echo -e "${YELLOW}Running experiment 7: Smart Lambda 1e-3${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-5" --lr "1e-3"

echo -e "${YELLOW}Running experiment 8: Smart Lambda 1e-2${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --smart_lambda "1e-4" --lr "1e-3"

echo -e "${YELLOW}Running experiment 9: Jacobian Regularization Lambda 1e-4$ {RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-6" --lr "1e-3"

echo -e "${YELLOW}Running experiment 10: Jacobian Regularization Lambda 1e-5${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-4" --lr "1e-3"

echo -e "${YELLOW}Running experiment 11: Jacobian Regularization Lambda 1e-2${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --jreg_lambda "1e-2" --lr "1e-3"

echo -e "${YELLOW}Running experiment 12: Spectrum Analysis - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 25 --lr "1e-3"

echo -e "${YELLOW}Running experiment 13: Spectrum Analysis - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 50 --lr "1e-3"

echo -e "${YELLOW}Running experiment 14: Spectrum Analysis - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --spectrum --top_percent 75 --lr "1e-3"

### Extemsion of regularization with Jacobian and Smart
echo -e "${YELLOW}Running experiment 15: Jacobian + Spectrum - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --spectrum --top_percent 25  --lr "1e-3"

echo -e "${YELLOW}Running experiment 16: Jacobian + Spectrum - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --spectrum --top_percent 50 --lr "1e-3"

echo -e "${YELLOW}Running experiment 17: Jacobian + Spectrum - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --spectrum --top_percent 75 --lr "1e-3"

#### Extemsion of regularization with Smart and Spectrum
echo -e "${YELLOW}Running experiment 18: Smart + Spectrum - Top 25%${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --spectrum --top_percent 25  --lr "1e-3"

echo -e "${YELLOW}Running experiment 19: smart+ Spectrum - Top 50%${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --spectrum --top_percent 50 --lr "1e-3"

echo -e "${YELLOW}Running experiment 20: smart + Spectrum - Top 75%${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --spectrum --top_percent 75 --lr "1e-3"


##### Experiments lora and regularizers

echo -e "${YELLOW}Running experiment 21: Jacobian + LORA${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --lora --lr "1e-3"

echo -e "${YELLOW}Running experiment 22: Smart + LORA${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --lora --lr "1e-3"


##### Experiments qlora and regularizers

echo -e "${YELLOW}Running experiment 23: Jacobian + QLORA${RESET}"
python sonnet_generation_ext.py -e $epochs --jacobian --qlora --lr "1e-3"

echo -e "${YELLOW}Running experiment 24: Smart + QLORA ${RESET}"
python sonnet_generation_ext.py -e $epochs --smart --qlora --lr "1e-3"



# Completion message
echo -e "${GREEN}All experiments completed successfully! 🚀${RESET}"