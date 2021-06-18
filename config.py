"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = ''
LSP_ROOT = ''
LSP_ORIGINAL_ROOT = ''
LSPET_ROOT = '/home/ubuntu/data/hr-lspet'
MPII_ROOT = '/home/ubuntu/data/mpii/images'
COCO_ROOT = '/home/ubuntu/data/coco/train2014'
COCO_SAMPLE_ROOT = '/home/ubuntu/data/coco/train2014'
COCO_TEST_ROOT = '/home/ubuntu/data/coco/train2014'
MPI_INF_3DHP_ROOT = '/home/ubuntu/data/mpi_inf_3dhp'
PW3D_ROOT = ''
UPI_S1H_ROOT = ''

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                  #  'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_train.npz'),

                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'coco-test': join(DATASET_NPZ_PATH, 'coco_with_3d-test.npz')
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii3D.npz'),
                   'coco-sample': join(DATASET_NPZ_PATH, 'coco_2014_train_sample.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'lsp3D.npz'),
                   'coco3D': join(DATASET_NPZ_PATH, 'coco3D.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi3D.npz')
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   'coco3D': COCO_ROOT,
                   'coco-sample': COCO_SAMPLE_ROOT,
                   'coco-test': COCO_TEST_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
