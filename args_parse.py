import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    fp_choices = ['MorganFP', 'RdkitFP', 'EstateFP', 'MACCSFP', 'PubChemFP', 'RGroupFP', 'FragmentFP']
    # exp
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--dump_path", default="dump/", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--random_seed", default=[0, 1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument("--load_path", default=None, type=str)

    # dataset
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--config_path", default='configs', type=str)
    parser.add_argument("--dataset", default='GOODHIV', type=str)
    parser.add_argument("--domain", default='scaffold', type=str)
    parser.add_argument("--shift", default='covariate', type=str)
    parser.add_argument("--split_seed", default=0, type=int)

    # VQ
    parser.add_argument("--num_e", default=512, type=int)
    parser.add_argument("--commitment_weight", default=0.1, type=float)

    # Encoder
    parser.add_argument("--emb_dim", default=128, type=int)
    parser.add_argument("--layer", default=4, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--gnn_type", default='gin', type=str, choices=['gcn', 'gin'])
    parser.add_argument("--fp_type", default='MorganFP', type=str, choices=fp_choices)
    parser.add_argument("--fp_pool_type", default='attn', type=str, choices=['sum', 'attn'])
    parser.add_argument("--fp_attn_heads", default=4, type=int)
    parser.add_argument("--use_vq_residual", default=1, type=int, choices=[0, 1])
    parser.add_argument("--concrete_temp", default=1.0, type=float)
    parser.add_argument("--fp_score_mode", default='bit_repr', type=str,
                        choices=['bit_repr', 'graph_fp_pretrain'])
    parser.add_argument("--logit_fusion_mode", default='avg', type=str,
                        choices=['avg', 'rescale'])
    parser.add_argument("--prediction_mode", default='fp_only', type=str,
                        choices=['fp_only', 'fusion'])

    # Training
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument("--best_score_warmup", default=1, type=int)

    args = parser.parse_args()

    return args
