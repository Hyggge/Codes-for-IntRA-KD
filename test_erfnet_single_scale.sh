python3 -u test_erfnet_single_scale.py ApolloScape ERFNet train bdd100k_val \
                          --lr 0.01 \
                          --npb \
                          --resume trained_model/erfnet_trained.pth.tar \
                          --test_size 1440 \
                          -j 6 \
                          -b 2
