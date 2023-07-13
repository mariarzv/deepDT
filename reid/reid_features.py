import torchreid
from .utils.feature_extractor import FeatureExtractor

class REIDFeatures:
    """
    This class generates features from bbox image patches.
    """

    def __init__(self, modelID):
        self.modelID = modelID

    def gen_features(self, patches):

        # default model (modelID == 5)
        te = FeatureExtractor('osnet_x1_0', 'cpu')

        if self.modelID == 0:  # mudeep
            te = FeatureExtractor('MuDeep', 'cpu')
        if self.modelID == 1:  # resnet
            te = FeatureExtractor('resnet50', 'cpu')
        if self.modelID == 2:  # hacnn
            te = FeatureExtractor('HACNN', 'cpu')
        if self.modelID == 3:  # pcb
            te = FeatureExtractor('pcb_p6', 'cpu')
        if self.modelID == 4:  # mlfn
            te = FeatureExtractor('mlfn', 'cpu')

        features_torch = te(patches)
