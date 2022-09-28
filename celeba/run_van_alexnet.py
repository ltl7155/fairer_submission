import os

name = "celeba"
fair_method = "van"

small = False

if small:
    for exp in range(0, 5):
        for lam2 in lams2:
            sh = f"CUDA_VISIBLE_DEVICES=3 python main_prune_small.py --fair_method {fair_method} --mode {mode} --lam {lam} --lam2 {lam2} --exp {exp} --epochs 20 "
            fname = f"results/Small/{fair_method}/Exp{exp}_{mode}_{name}_{fair_method}-lam-{lam}-lam2-{lam2}.txt"
            if not os.path.exists(fname):
                os.system(sh)

else:
    for exp in range(2, 3):
        for mode in ["eo", "dp"]:
            sh = f"CUDA_VISIBLE_DEVICES=6 python main_van.py --fair_method van --mode {mode} --exp {exp} --epochs 20"
            
#             sh = f"CUDA_VISIBLE_DEVICES=4 python main_prune_alexnet.py --fair_method CAIGA --mode {mode} --lam 0.0 --lam2 0.0 --exp {exp} --epochs 20"
            os.system(sh)
