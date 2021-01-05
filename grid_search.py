import os
import numpy as np

ident_word = "nmnist"

part1 = "#!/bin/csh \n#$ -M cschaef6@nd.edu \n#$ -m abe\n#$ -q gpu@@joshi" 
part11 = "\n#$ -l gpu_card=1\n#$ -N "

part2 = "\n#$ -o ./logs/runs/output_"+ident_word+"_"

part3 = ".txt\n#$ -e ./logs/runs/error_"+ident_word+"_"

part4 = ".txt\nmodule load python cuda/10.2\nsetenv XLA_FLAGS --xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2\nsetenv OMP_NUM_THREADS $NSLOTS\npython bptt_snn.py"


random_seeds = [193012823 ,235899598, 8627169, 103372330, 14339038, 221706254, 46192121, 188833202, 37306063, 171928928]

trials = 1

# parameters = {
# 'architecture' : ["700-400-250", "700-300-250", "700-500-250", "700-600-400-250"] ,
# 'l_rate'       : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .5, 1],
# 'alpha'        : np.arange(.1,1.01,.05)  , 
# 'gamma'        : np.arange(.1,1.01,.05)  ,      
# 'alpha_vr'     : np.arange(.1,1.01,.05) ,         
# 'thr'          : np.arange(.01,1.04,.05) 
# }

parameters = {
'l_rate'       : [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
'alpha'        : [.5, .55, .6, .65, .7, .75, .8, .85, .9], 
'gamma'        : [.1, .15, .2, .25, .3],
}


for i in range(trials):
    for variable in parameters:
        for value in parameters[variable]:
            name = ident_word + "_" +variable + "_" + str(value).replace(",","") + "_" + str(i)
            with open('jobscripts/'+name+'.script', 'w') as f:
                if isinstance(value, str):
                    f.write(part1 + part11  + name + part2 + name + part3 + name + part4 + " --" + variable + " \"" + value+ "\" --data-set \"NMNIST\" --architecture \"2048-500-100-10\" --epochs 20 --seed " + str(random_seeds[i]) + " --log-file logs/" + ident_word + "_" + variable + ".csv") 
                else:
                    f.write(part1 + part11  + name + part2 + name + part3 + name + part4 + " --" + variable + " " + str(value)+ " --data-set \"NMNIST\" --architecture \"2048-500-100-10\" --epochs 20 --seed " + str(random_seeds[i]) + " --log-file logs/" + ident_word + "_" + variable + ".csv") 
            os.system("qsub "+ 'jobscripts/'+name+'.script')