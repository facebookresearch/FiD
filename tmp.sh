# n_context=40 batch_size=1 model_size=large extra_arguments="--use_checkpointing" num_nodes=8 data_prefix=webq ../FiD/scripts/train_webq_freebase.sh webquestions_freebase_webqsp_graftnet_nxhd_rest_joint_stagg_nxhd_100tokens /private/home/xilun/hybridqa/webquestions/fid_inputs/dpr_webq_split/freebase/webq_freebase_webqsp_graftnet_nxhd_plus_joint_stagg_nxhd_100tokens/
# 
# sleep 64000
# 
# n_context=40 batch_size=1 model_size=large extra_arguments="--use_checkpointing" num_nodes=8 data_prefix=webquestions ../FiD/scripts/train_webq_freebase.sh webquestions_wikipedia_textl_tables_plus_freebase_chimera_nxhd_100tokens /private/home/xilun/hybridqa/webquestions/fid_inputs/dpr_webq_split/wikipedia_textl_tables_plus_freebase_100tokens/

n_context=100 batch_size=1 model_size=base extra_arguments="--use_checkpointing" num_nodes=8 data_prefix=webq ../FiD/scripts/train_webq_freebase.sh webquestions_freebase_webqsp_graftnet_nxhd_rest_joint_stagg_nxhd_100tokens /private/home/xilun/hybridqa/webquestions/fid_inputs/dpr_webq_split/freebase/webq_freebase_webqsp_graftnet_nxhd_plus_joint_stagg_nxhd_100tokens/

sleep 64000

n_context=100 batch_size=1 model_size=base extra_arguments="--use_checkpointing" num_nodes=8 data_prefix=webquestions ../FiD/scripts/train_webq_freebase.sh webquestions_wikipedia_textl_tables_plus_freebase_chimera_nxhd_100tokens /private/home/xilun/hybridqa/webquestions/fid_inputs/dpr_webq_split/wikipedia_textl_tables_plus_freebase_100tokens/
