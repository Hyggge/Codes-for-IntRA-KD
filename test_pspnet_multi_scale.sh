python3 -u test_pspnet_multi_scale.py ApolloScape PSPNet train bdd100k_val \
                          --lr 0.01 \
                          --gpus 0 \
                          --npb \
                          --resume trained_model/pspnet_46_6.pth.tar \
                          --test_size 1410 \
                          -j 10 \
                          -b 1
