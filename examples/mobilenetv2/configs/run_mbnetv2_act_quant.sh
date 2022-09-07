
for BITS in 8 7 6 5 4 3 2
do
  python3 distillation_finetune.py --workdir=../../mbnet2_act_${BITS} --config=mobilenetv2/configs/mobilenetv2_a_int${BITS}_distillation.py
done