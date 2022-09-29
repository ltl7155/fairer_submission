import os

name = "celeba"
mode = "dp"
fair_method = "CAIGA"
if mode == "dp":
#     lams2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    lams2 = [0.6]
elif mode == "eo":
#     lams2 = [0.0, 0.1, 0.4, 0.7, 1.0, 1.3, 1.6]
    lams2 = [0.0] 

small = False
sl = 0

if small:
    for exp in range(0, 5):
        for lam2 in lams2:
            sh = f"CUDA_VISIBLE_DEVICES=3 python main_prune_small.py --fair_method {fair_method} --mode {mode} --lam {lam} --lam2 {lam2} --exp {exp} --epochs 20  "
            fname = f"results/Small/{fair_method}/Exp{exp}_{mode}_{name}_{fair_method}-lam-{lam}-lam2-{lam2}.txt"
            if not os.path.exists(fname):
                os.system(sh)

else:
    for exp in range(0, 5):
        for lam2 in lams2:
            lam = 0.0
            sh = f"CUDA_VISIBLE_DEVICES=5 python main_prune_alexnet.py --fair_method {fair_method} --mode {mode} --lam {lam} --lam2 {lam2} --exp {exp} --epochs 20 --sl {sl} --target_id 2"
            fname = f"results_alexnet/Normal/{fair_method}/Exp{exp}_{mode}_{name}_{fair_method}-lam-{lam}-lam2-{lam2}_sl{sl}.txt"
            if not os.path.exists(fname):
                os.system(sh)

