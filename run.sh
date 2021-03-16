export COMET_PROJECT_NAME="lits-unetpp"

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 256  \
               --neg-multiplier 2 --exp unetpp-cont-only-np256-nm2 --logging \
               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 256  \
               --neg-multiplier 2 --exp unetpp-cont-only-np256-nm2 --logging \
               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 256  \
               --neg-multiplier 2 --exp unetpp-cont-only-np256-nm2 --logging \
               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 256  \
               --neg-multiplier 2 --exp unetpp-bd-random-np256-nm2 --logging \
               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
               --boundary-aware --boundary-loc both --sampling-type random
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 256  \
               --neg-multiplier 2 --exp unetpp-bd-random-np256-nm2 --logging \
               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
               --boundary-aware --boundary-loc both --sampling-type random
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 256  \
#               --neg-multiplier 2 --exp unetpp-bd-random-np256-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type random
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-bd-linear-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type linear
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-bd-linear-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type linear
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-bd-linear-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type linear
#
#
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-cont-only-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-bd-fixed-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type fixed
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-bd-random-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type random
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --arch unetpp --loss-weight 1.0 --n-max-pos 128  \
#               --neg-multiplier 2 --exp unetpp-bd-linear-np128-nm2 --logging \
#               --batch-size 64 --gpus 4 --optim sgd --lr 0.1 --max-epochs 120 \
#               --boundary-aware --boundary-loc both --sampling-type linear
#
#
#
#
