for RHO in 0. 0.01 0.1 0.5 1.0 1.5 2.0 5.0 10.
do
  rm -rf ../../mnist_admm_${RHO}
  python3 train.py --workdir=../../mnist_admm_${RHO} --config=mnist/configs/mnist_admm.py --config.rho=${RHO}
done
