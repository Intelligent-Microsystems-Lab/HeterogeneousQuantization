for RHO in 1 2 5 10 15 20 30 50
do
  rm -rf ../../mnist_admm_${RHO}
  python3 train.py --workdir=../../mnist_admm_t1_${RHO} --config=mnist/configs/mnist_admm.py --config.num_steps=${RHO}
done
