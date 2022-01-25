proj_home=/li_zhengdao/github/GenerativeGNN


nohup python -u $proj_home/main.py \
--mypredictor \
--server_tag=2_clone \
--num_gumbels=1 \
--device=0 \
--num_layers=2 \
--num_samples=5 \
--node_emb=256 \
--hidden_channels=256 \
--dropout=0.3 \
--lr=0.003 \
--epochs=200 \
--log_steps=10 \
--eval_steps=1 \
--runs=10 \
> $proj_home/ppi_20220124_2.log 2>&1 &
