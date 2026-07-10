from .causal_cnn import CausalCNNEncoder
from .causal_cnn_old import CausalCNNEncoder as CausalCNNEncoderOld
from .causal_cnn_cinc import CausalCNNClassifier as CausalCNNCINC


NETWORKS = {
  'gtcn': CausalCNNEncoderOld,
  'causalcnn': CausalCNNEncoder,
  'causalcnn_cinc': CausalCNNCINC
}


def get_network(name):
  return NETWORKS[name]
