from provnet_utils import *
from config import *

from labelling import get_ground_truth
from detection.evaluation_methods.evaluation_utils import *

def main():
    TP = 115
    FN = 3
    TN = 304271
    FP = 394906

    MCC = (TP * TN - FP * FN)/ (((TP + FP) * (TN + FN) * (TP + FN) * (TN + FP))**0.5 )
    print(MCC)

if __name__ == "__main__":
    main()