
for RHO in 5. 10.
do
  python3 train.py --workdir=../../admm_${RHO} --config=efficientnet/configs/efficientnet-lite0_admm_mixed_gran_sur.py --config.rho=${RHO}
done