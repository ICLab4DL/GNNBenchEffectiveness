proj_home=/li_zhengdao/github/GenerativeGNN


nohup python -u $proj_home/main.py \
--server_tag=torch17 \
--device=0 \
--num_layers=2 \
--num_samples=5 \
--node_emb=256 \
--hidden_channels=256 \
--dropout=0.3 \
--lr=0.003 \
--epochs=50 \
--log_steps=10 \
--eval_steps=1 \
--runs=10 \
> $proj_home/ppi_20220131_test.log 2>&1 &
