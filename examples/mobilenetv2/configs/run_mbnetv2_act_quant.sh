
for BITS in 5 4
do
  python3 distillation_finetune.py --workdir=../../mbnet2_act_${BITS} --config=mobilenetv2/configs/mobilenetv2_a_int${BITS}_distillation.py
done