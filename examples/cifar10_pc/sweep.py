import argparse
import subprocess
import numpy as np

# sweep.py --parameter noise_weight --from .0 --to .25 --steps 25 --network_type pc --result_dir cifar10_pc_weight_noise

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--parameter", type=str, default=None, help="Random Seed"
)
parser.add_argument(
    "--start", type=float, default=None, help="Random Seed"
)
parser.add_argument(
    "--stop", type=float, default=None, help="Random Seed"
)
parser.add_argument(
    "--step", type=float, default=None, help="Random Seed"
)
parser.add_argument(
    "--network_type", type=str, default=None, help="Random Seed"
)
parser.add_argument(
    "--result_dir", type=str, default=None, help="Random Seed"
)
parser.add_argument(
    "--user", type=str, default=None, help="Random Seed"
)
parser.add_argument(
    "--queue", type=str, default=None, help="Random Seed"
)
args = parser.parse_args()


if __name__ == "__main__":
  # creat result dir
  subprocess.call(["mkdir", args.result_dir])

  # create sweep values
  for val in np.arange(start=args.start, stop=args.stop, step=args.step):
    subprocess.call(["mkdir", args.result_dir+'/{:.6f}'.format(val)])

    name = args.network_type + '_' + args.parameter + '_' + str(val)

    job_script = "#!/bin/csh \n#$ -M "
    job_script += args.user
    job_script += " \n#$ -m abe\n#$ -q "
    job_script += args.queue
    job_script += "\n#$ -l gpu_card=1\n#$ -N "
    job_script += name
    job_script += "\n#$ -o "
    job_script += args.result_dir+"/{:.6f}".format(val) + "/out_" + name
    job_script += ".log \n#$ -e "
    job_script += args.result_dir+"/{:.6f}".format(val) + "/error_" + name
    job_script += ".log\n\nmodule load python cuda/11.2 tensorflow/2.6\n"
    job_script += "setenv XLA_FLAGS \"--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/11.2\"\n"
    job_script += "setenv OMP_NUM_THREADS $NSLOTS\n"
    job_script += "pip3 install --user clu\n"
    job_script += "python3 train"
    if args.network_type == 'pc':
      pass
    elif args.network_type == 'bp':
      job_script += "_bp"
    else:
      raise Exception('Unknown network type:' + args.network_type)
    job_script += ".py --workdir="
    job_script += args.result_dir+'/{:.6f}'.format(val)
    job_script += " --config=configs/default.py"
    job_script += " --config."+args.parameter + '=' + str(val)
    job_script += "\n"

    with open(args.result_dir+'/{:.6f}'.format(val)+'/job.script', "w") as text_file:
      text_file.write(job_script)

    subprocess.call(
        ["qsub", args.result_dir+'/{:.6f}'.format(val)+'/job.script'])
