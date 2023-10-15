# Reproducing results

Run the following commands in the root directory. Note that many of the hyperparameters are not extensively tuned and are adapted from [GOAT repo](https://github.com/devnkong/GOAT).

## 1. `ogbn-products`
```
python main.py \
    --dataset ogbn-products \
    --sample_node_len 100 \
    --lr 1e-3 \
    --batch_size 1024 \
    --test_batch_size 256 \
    --hidden_dim 256 \
    --global_dim 64 \
    --num_workers 4 \
    --conv_type local \
    --num_heads 2 \
    --num_centroids 4096 

python main.py \
    --dataset ogbn-products \
    --sample_node_len 100 \
    --lr 1e-3 \
    --batch_size 1024 \
    --test_batch_size 256 \
    --hidden_dim 256 \
    --global_dim 64 \
    --num_workers 4 \
    --conv_type full \
    --num_heads 2 \
    --num_centroids 4096 
```

## 2. `snap-patents`
```
python main.py \
    --dataset snap-patents \
    --sample_node_len 50 \
    --lr 1e-3 \
    --batch_size 2048 \
    --test_batch_size 1024 \
    --hidden_dim 128 \
    --global_dim 64 \
    --num_workers 4 \
    --conv_type local \
    --num_heads 2 \
    --num_centroids 4096 
    
python main.py \
    --dataset snap-patents \
    --sample_node_len 50 \
    --lr 1e-3 \
    --batch_size 2048 \
    --test_batch_size 1024 \
    --hidden_dim 128 \
    --global_dim 64 \
    --num_workers 4 \
    --conv_type full \
    --num_heads 2 \
    --num_centroids 4096 
```

## 3. `ogbn-papers100M`
```
python main.py \
    --dataset ogbn-papers100M \
    --sample_node_len 100 \
    --lr 1e-3 \
    --batch_size 1024 \
    --test_batch_size 1024 \
    --hidden_dim 512 \
    --global_dim 128 \
    --num_workers 4 \
    --conv_type full \
    --num_heads 2 \
    --num_centroids 4096 
```

## 4. Additional experiments

```
sample_node_len_values=(20 40 50 60 80 100 150 200)

for sample_node_len in "${sample_node_len_values[@]}"; do
  echo "Running with sample_node_len = $sample_node_len"

  python main.py \
      --dataset ogbn-products \
      --sample_node_len $sample_node_len \
      --lr 1e-3 \
      --batch_size 1024 \
      --test_batch_size 256 \
      --hidden_dim 256 \
      --global_dim 64 \
      --num_workers 4 \
      --conv_type full \
      --num_heads 2 \
      --num_centroids 4096

  # sleep 2s
done
```

```
sample_node_len_values=(20 40 50 60 80 100 150 200)

for sample_node_len in "${sample_node_len_values[@]}"; do
  echo "Running with sample_node_len = $sample_node_len"

    python main.py \
        --dataset snap-patents \
        --sample_node_len $sample_node_len \
        --lr 1e-3 \
        --batch_size 2048 \
        --test_batch_size 1024 \
        --hidden_dim 128 \
        --global_dim 64 \
        --num_workers 4 \
        --conv_type full \
        --num_heads 2 \
        --num_centroids 4096 

  # sleep 2s
done
```