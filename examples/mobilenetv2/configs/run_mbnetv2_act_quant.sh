
for BITS in 8 5 4
do
  python3 distillation_finetune.py --workdir=../../mbnet2_act_${BITS}_t2 --config=mobilenetv2/configs/mobilenetv2_a_int${BITS}_distillation.py
done