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

echo -e "${GREEN}All experiments completed successfully! 🚀${RESET}"