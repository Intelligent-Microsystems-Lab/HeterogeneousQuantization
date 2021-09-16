import argparse
import subprocess
import numpy as np

# Script to setup noise sweeps on Notre Dame CRC cluster.

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--parameter", type=str, default=None, help="Random Seed")
parser.add_argument("--start", type=float, default=None, help="Random Seed")
parser.add_argument("--stop", type=float, default=None, help="Random Seed")
parser.add_argument("--step", type=float, default=None, help="Random Seed")
parser.add_argument(
    "--network_type", type=str, default=None, help="Random Seed"
)
parser.add_argument("--result_dir", type=str, default=None, help="Random Seed")
parser.add_argument("--user", type=str, default=None, help="Random Seed")
parser.add_argument("--queue", type=str, default=None, help="Random Seed")
parser.add_argument("--trials", type=int, default=1, help="Random Seed")
args = parser.parse_args()


seed_list = [2029492581, 2223210, 1594305760, 87953651674304230, 2467475923055248755, 203853699, 2151901553968352745]

if __name__ == "__main__":
  # creat result dir
  subprocess.call(["mkdir", args.result_dir])

  # create sweep values
  for i in range(args.trials):
    for val in np.arange(start=args.start, stop=args.stop, step=args.step):
      work_dir =  args.result_dir + "/{:.6f}_t".format(val, i)
      subprocess.call(["mkdir", work_dir])

      name = (
          args.parameter
          + "_"
          + "{:.6f}".format(val)
          + "_"
          + args.network_type
          + "_t"
          + str(i)
      )

      job_script = "#!/bin/csh \n#$ -M "
      job_script += args.user
      job_script += " \n#$ -m abe\n#$ -q "
      job_script += args.queue
      job_script += "\n#$ -l gpu_card=1\n#$ -N "
      job_script += name
      job_script += "\n#$ -o "
      job_script += work_dir + "/out_" + name
      job_script += ".log \n#$ -e "
      job_script += (
          work_dir + "/error_" + name
      )
      job_script += ".log\n\nmodule load python cuda/11.2 tensorflow/2.6\n"
      job_script += 'setenv XLA_FLAGS "--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/11.2"\n'
      job_script += "setenv OMP_NUM_THREADS $NSLOTS\n"
      job_script += "python3 train"
      if args.network_type == "pc":
        pass
      elif args.network_type == "bp":
        job_script += "_bp"
      else:
        raise Exception("Unknown network type:" + args.network_type)
      job_script += ".py --workdir="
      job_script += work_dir
      job_script += " --config=configs/default.py"
      job_script += " --config." + args.parameter + "=" + str(val)
      job_script += " --config.seed=" + str(seed_list[i])
      job_script += "\n"

      with open(
          work_dir + "/job.script", "w"
      ) as text_file:
        text_file.write(job_script)

      subprocess.call(
          ["qsub", work_dir + "/job.script"]
      )
