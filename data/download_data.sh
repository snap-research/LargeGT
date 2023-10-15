# ogbn-products and ogbn-papers100M dataset are downloaded by ogb package

## 1. Mandatory download files (not automatically prepared in code repo, otherwise)

# snap-patents dataset 
curl "https://www.dropbox.com/scl/fi/upsn08zx20nsxcyhfwc8d/snap_patents.mat?rlkey=4efpj1sg9s2fe5gjf7755tg81&dl=1" -o snap_patents.mat -J -L -k

# pos_enc pre-prcessed files
curl "https://www.dropbox.com/scl/fi/k3pabro7veyufburmmqkn/ogbn-products_embedding_64.pt?rlkey=ac84eb75nj741ueic7l69pbab&dl=1" -o ogbn-products_embedding_64.pt -J -L -k
curl "https://www.dropbox.com/scl/fi/ig4t4kbk2454gteuel1xg/snap-patents_embedding_64.pt?rlkey=gduuevqbd1qsnrlat07mx7zka&dl=1" -o snap-patents_embedding_64.pt -J -L -k
curl "https://www.dropbox.com/scl/fi/361vcdkk459kpj19tsqz2/ogbn-papers100M_data_dict.pt?rlkey=0ptndnsybxoptxfvxia9ncylf&dl=1" -o ogbn-papers100M_data_dict.pt -J -L -k

## 2. Optional download files (automatically prepared in code repo, otherwise) | For main expts
curl "https://www.dropbox.com/scl/fi/puy19h1mx6bqy0k7g21sv/snap-patents_new_tokenizer_duplicates_50sample_node_len_2hop.pt?rlkey=tudi1p32t273xowr2uxpt5rr0&dl=1" -o snap-patents_sample_node_len_50.pt -J -L -k
curl "https://www.dropbox.com/scl/fi/c8vy6lumnge89krv0wsv0/ogbn-products_new_tokenizer_duplicates_100sample_node_len_2hop.pt?rlkey=sflq2gs6bs1n6qydm8fab1upm&dl=1" -o ogbn-products_sample_node_len_100.pt -J -L -k
curl "https://www.dropbox.com/scl/fi/s89fe40rpubz4wci5bhq0/ogbn-papers100M_new_tokenizer_duplicates_100sample_node_len_2hop.pt?rlkey=rak2qnbft7hacgsabapbxlsrn&dl=1" -o ogbn-papers100M_sample_node_len_100.pt -J -L -k
curl "https://www.dropbox.com/scl/fi/bj8oi70urszrk5ngophr8/ogbn-papers100M_new_tokenizer_duplicates_100sample_node_len_2hop_hop2token_feats.pt?rlkey=pifpa9qok19yn5ya7m19jz5rk&dl=1" -o ogbn-papers100M_sample_node_len_100_hop2token_feats.pt -J -L -k

## 3. Optional download files (automatically prepared in code repo, otherwise) | For ablation studies

# curl "https://www.dropbox.com/scl/fi/zvc8zh983pu8j60lxjw58/snap-patents_new_tokenizer_duplicates_20sample_node_len_2hop.pt?rlkey=r4gd4l28yd2k0f5y066pho0p1&dl=1" -o snap-patents_sample_node_len_20.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/7c2tb1lopwk4ctnfliffc/snap-patents_new_tokenizer_duplicates_40sample_node_len_2hop.pt?rlkey=54jlqn4fian5sstvi1w76z65q&dl=1" -o snap-patents_sample_node_len_40.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/17cg49vuc0q0k3lv9o64e/snap-patents_new_tokenizer_duplicates_60sample_node_len_2hop.pt?rlkey=kdcom0hgde1davlsbehentiet&dl=1" -o snap-patents_sample_node_len_60.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/78ipdkdvb8bh4bxfauz08/snap-patents_new_tokenizer_duplicates_80sample_node_len_2hop.pt?rlkey=6vip7gi8nlnen9uyiwzme376k&dl=1" -o snap-patents_sample_node_len_80.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/hi99mql9cy8aryq5uzvyt/snap-patents_new_tokenizer_duplicates_100sample_node_len_2hop.pt?rlkey=tqniki8g9nn3ob5ylmu8njy41&dl=1" -o snap-patents_sample_node_len_100.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/fom1bx36meib7o1kh5uhn/snap-patents_new_tokenizer_duplicates_150sample_node_len_2hop.pt?rlkey=8bqis8xth3frqyg3zqfsjjn6r&dl=1" -o snap-patents_sample_node_len_150.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/5j33m1tu9jpj6jba585lf/snap-patents_new_tokenizer_duplicates_200sample_node_len_2hop.pt?rlkey=unr06j8jbc9m7i3omr4j8il3t&dl=1" -o snap-patents_sample_node_len_200.pt -J -L -k

# curl "https://www.dropbox.com/scl/fi/ux3j7jvopnmk1m5jy4e0l/ogbn-products_new_tokenizer_duplicates_20sample_node_len_2hop.pt?rlkey=9c80gtuh54n6alcj71crvzol5&dl=1" -o ogbn-products_sample_node_len_20.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/lvvxs2l6td6bxc55kn71j/ogbn-products_new_tokenizer_duplicates_40sample_node_len_2hop.pt?rlkey=bx03569smrhth98rhpjhrfy55&dl=1" -o ogbn-products_sample_node_len_40.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/1avmqsyb0o5xtmshazwqi/ogbn-products_new_tokenizer_duplicates_60sample_node_len_2hop.pt?rlkey=mvjgchtnyp8bazdxmvznjaw0l&dl=1" -o ogbn-products_sample_node_len_60.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/zxjnwe8g5uemhh04jngfh/ogbn-products_new_tokenizer_duplicates_80sample_node_len_2hop.pt?rlkey=7ostgb5tuwscdeawzv9wfwxfu&dl=1" -o ogbn-products_sample_node_len_80.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/heuiotzx3kector1nkf2a/ogbn-products_new_tokenizer_duplicates_150sample_node_len_2hop.pt?rlkey=p5tkv7xyxdu0rfp8as25nes9h&dl=1" -o ogbn-products_sample_node_len_150.pt -J -L -k
# curl "https://www.dropbox.com/scl/fi/e88i28lvgr61or9m5erkq/ogbn-products_new_tokenizer_duplicates_200sample_node_len_2hop.pt?rlkey=lxa3i3we7gklw07ie0xef2z0o&dl=1" -o ogbn-products_sample_node_len_200.pt -J -L -k

# curl "https://www.dropbox.com/scl/fi/ei93gn3z6nwtxo7dmyj06/ogbn-papers100M_new_tokenizer_duplicates_50sample_node_len_2hop.pt?rlkey=0oa42z2lxe2qgq0jceb6dtpw1&dl=1" -o ogbn-papers100M_sample_node_len_50.pt -J -L -k