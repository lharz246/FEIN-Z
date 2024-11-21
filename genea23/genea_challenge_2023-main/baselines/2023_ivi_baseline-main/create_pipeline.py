import os
import numpy as np
from pymo.parsers import *
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from sklearn.pipeline import Pipeline
import joblib as jl


if __name__ == "__main__":
    parser = BVHParser()
    data_dir = "/media/compute/homes/lharz/genea23/genea_challenge_2023-main/dataset/genea2023_dataset"
    parsed_example = parser.parse(os.path.join(data_dir, "trn/main-agent/bvh/trn_2023_v0_000_main-agent.bvh"))

    joint = ['b_root', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_neck0', 'b_head',
             'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg',
             'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1',
             'b_l_pinky2', 'b_l_pinky3',
             'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1',
             'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1',
             'b_l_thumb2', 'b_l_thumb3',
             'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0',
             'b_r_thumb1', 'b_r_thumb2',
             'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3',
             'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1',
             'b_r_index2', 'b_r_index3']

    mexp_full = Pipeline([
        ('jtsel', JointSelector(joint, include_root=True)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_withroot()),
        ('np', Numpyfier()),
    ])
    
    fullexpdata = mexp_full.fit_transform([parsed_example])[0]

    mexp_upperbody = Pipeline([
        #('jtsel', JointSelector(joint, include_root=False)),
        ('param', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover_()),
        ('np', Numpyfier()),
    ])
    upperexpdata = mexp_upperbody.fit_transform([parsed_example])[0]
    
    jl.dump(mexp_full, "pipeline_expmap_full.sav")
    jl.dump(mexp_upperbody, "pipeline_expmap_upper.sav")
    