import os
import numpy as np

ident_word = "snn_pattern"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q gpu@@joshi" 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/runs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/runs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python cuda/10.2\nsetenv XLA_FLAGS --xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2\nsetenv OMP_NUM_THREADS $NSLOTS\npython bptt_snn.py"


random_seeds = [193012823 ,235899598, 8627169, 103372330, 14339038, 221706254, 46192121, 188833202, 37306063, 171928928]

trials = 3

architecture = ["700-400-250", "700-300-250", "700-500-250", "700-600-400-250"] 
l_rate       = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .5, 1]
alpha        = np.arange(.1,1.01,.05)        
alpha_vr     = np.arange(.1,1.01,.05)           
thr          = np.arange(.01,1.04,.05)

for i in range(trials):
    for a1 in architecture:
        for lr in l_rate:
            for a2 in alpha:
                for a3 in alpha_vr:
                    for ths in thr:

                        name = ident_word + "_a1_"+a1+ "_lr_"+str(lr) + "_a2_"+str(a2)+ "_a3_"+str(a3)+ "_thr_"+str(ths) +"_" + str(i)
                        with open('jobscripts/'+name+'.script', 'w') as f:
                            f.write(part1 + part11  + name + part2 + name + part3 + name + part4 + " --architecture" + " \"" + a1+ "\" --l_rate " + str(lr)+ " --alpha "+ str(a2) + " --alpha_vr " + str(a3) + " --thr " + str(ths) +" --seed " + str(random_seeds[i]) + " --log-file " + ident_word + ".csv") 
                        import pdb; pdb.set_trace()
                        os.system("qsub "+ 'jobscripts/'+name+'.script')