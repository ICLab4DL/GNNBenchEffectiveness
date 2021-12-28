proj_home=/li_zhengdao/github/GenerativeGNN


nohup python -u $proj_home/main.py \
--device=0 \
--num_layers=2 \
--num_samples=5 \
--node_emb=256 \
--hidden_channels=256 \
--dropout=0.3 \
--lr=0.003 \
--epochs=400 \
--log_steps=1 \
--eval_steps=1 \
> $proj_home/ppi_20211228.log 2>&1 &

