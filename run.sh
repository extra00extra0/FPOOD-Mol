#!/bin/bash



# python run.py --dataset ec50_assay --gnn_type gcn --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 40 --exp_name ec50_assay_gcn --exp_id seed0-4

# python run.py --dataset ec50_scaffold --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 40 --exp_name ec50_scaffold_gcn --exp_id seed0-4

# python run.py --dataset ec50_size --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 40 --exp_name ec50_size_gcn --exp_id seed0-4

# python run.py --dataset ic50_assay --gnn_type gcn --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 40 --exp_name ic50_assay_gcn --exp_id seed0-4

# python run.py --dataset ic50_scaffold --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 40 --exp_name ic50_scaffold_gcn --exp_id seed0-4

# python run.py --dataset ic50_size --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 40 --exp_name ic50_size_gcn --exp_id seed0-4


# python run.py --dataset BBBP --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name BBBP_sum/bit/no --exp_id seed0-4
# python run.py --dataset BBBP --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name BBBP_sum/fp/no --exp_id seed0-4
# python run.py --dataset BBBP --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name BBBP_attn/bit/no --exp_id seed0-4
# python run.py --dataset BBBP --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name BBBP_attn/fp/no --exp_id seed0-4

# python run.py --dataset BACE --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name BACE_sum/bit/no --exp_id seed0-4
# python run.py --dataset BACE --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name BACE_sum/fp/no --exp_id seed0-4
# python run.py --dataset BACE --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name BACE_attn/bit/no --exp_id seed0-4
# python run.py --dataset BACE --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name BACE_attn/fp/no --exp_id seed0-4

# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-concept_sum/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-concept_sum/fp/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-concept_attn/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-concept_attn/fp/no --exp_id seed0-4

# python run.py --dataset GOODZINC --domain scaffold --shift covariate --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-covariate_sum/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift covariate --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-covariate_sum/fp/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift covariate --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC_-scaf-covariate-attn/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift covariate --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-scaf-covariate_attn/fp/no --exp_id seed0-4

python run.py --dataset GOODZINC --domain size --shift concept --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 40 --exp_name GOODZINC-size-concept_sum/bit/no40 --exp_id seed0-4
# python run.py --dataset GOODZINC --domain size --shift concept --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-concept_sum/fp/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain size --shift concept --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-concept_attn/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain size --shift concept --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-concept_attn/fp/no --exp_id seed0-4

python run.py --dataset GOODZINC --domain size --shift covariate --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-covariate_sum/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain size --shift covariate --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-covariate_sum/fp/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain size --shift covariate --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-covariate_attn/bit/no --exp_id seed0-4
# python run.py --dataset GOODZINC --domain size --shift covariate --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 18 --exp_name GOODZINC-size-covariates_attn/fp/no --exp_id seed0-4

# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type sum --fp_score_mode bit_repr --prediction_mode fp_only --epoch 40 --exp_name GOODZINC-scaf-concept_sum/bit/no40 --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type sum --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 40 --exp_name GOODZINC-scaf-concept_sum/fp/no40 --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type attn --fp_score_mode bit_repr --prediction_mode fp_only --epoch 40 --exp_name GOODZINC-scaf-concept_attn/bit/no40 --exp_id seed0-4
# python run.py --dataset GOODZINC --domain scaffold --shift concept --fp_pool_type attn --fp_score_mode graph_fp_pretrain --prediction_mode fp_only --epoch 40 --exp_name GOODZINC-scaf-concept_attn/fp/no40 --exp_id seed0-4
