
# weight_act
# NAME="ste_psgd" 
# NAME="ste_ewgs"
# NAME="ste_invtanh"
# NAME="psgd_ste" 
# NAME="ewgs_ste"
# NAME="invtanh_ste"
# NAME="acos_ste"
# NAME="invtanh_ewgs"
# NAME="psgd_invtanh"
# NAME="ewgs_invtanh"

counter=0
for SEED in 193012823 235899598 8627169 103372330 14339038 7066636907367338630 6341454152401905754 1959330209 4275582192035986655 5329177101860618450 649323830 341384131 980310836 749032139 251745660 891479909 536363141 362693951 8488585 29492181
do
  python3 train.py --workdir=../../efficientnet-lite0_w3a3_${NAME}_${counter} --config=efficientnet/configs/efficientnet-lite0_${NAME}.py --config.seed=$SEED
  gsutil cp -r ./../efficientnet-lite0_mixed_${NAME}_${counter} gs://imagenet_clemens/surrogate_sweeps
  counter=$((counter+1))
done