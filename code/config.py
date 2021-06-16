from itertools import product


class Config():
    def __init__(self):
        project_dir = "/cluster/work/grlab/home/ajoudaki/seqCNN"
        self.dic = dict(
            project_dir = project_dir,
            tmp_dir = "/tmp/seqCNN",
            seqs_dir = f"{project_dir}/seqs",
            networks_dir = f"{project_dir}/networks",
            runs_dir = f"{project_dir}/runs",
            seqgen_dir = f"{project_dir}/data/generated_seqs",
            viral_dir = f"{project_dir}/data/viral",
            viral_ed = f"{project_dir}/data/viral_ed.csv",
            viral_benchmark = f"{project_dir}/data/viral_benchmark"
        )
    
    def __getitem__(self, instance):
        return self.dic[instance]
    

def grid_params(d):
    for k,v in d.items():
        if not isinstance(v,list):
            d[k] = [v]
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))
        

config = Config()
