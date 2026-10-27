from .drugdataset import DrugOODDataset
from .good_dataset import LOCAL_GOOD_DATASETS, LocalGOODDataset
from .molnet_dataset import MoleculeScaffoldDataset


MOLECULENET_DATASETS = {'BBBP', 'BACE'}


def build_non_good_dataset(args):
    if args.dataset in LOCAL_GOOD_DATASETS:
        return LocalGOODDataset(
            name=args.dataset,
            root=args.data_root,
            domain=args.domain,
            shift=args.shift,
            fp_type=args.fp_type,
        )
    if args.dataset in MOLECULENET_DATASETS:
        return MoleculeScaffoldDataset(
            name=args.dataset,
            root=args.data_root,
            fp_type=args.fp_type,
            split_seed=args.split_seed,
        )
    return DrugOODDataset(name=args.dataset, root=args.data_root, fp_type=args.fp_type)
