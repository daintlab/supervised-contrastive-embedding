

## exp: unet-gather-w1-np32-nm6-bdonly
#python main.py --loss-weight 1.0 --n-max-pos 32 \
#               --neg-multiplier 6 --exp unet-gather-w1-np32-nm6-bdonly \
#               --boundary-aware --only-boundary --logging
#
## exp: unet-gather-w1-np32-nm6-bd
#python main.py --loss-weight 1.0 --n-max-pos 32 \
#               --neg-multiplier 6 --exp unet-gather-w1-np32-nm6-bd \
#               --boundary-aware --logging
#
## exp: unet-gather-baseline
#python main.py --loss-weight 0.0 --n-max-pos 32 \
#               --neg-multiplier 6 --exp unet-gather-baseline --logging
#
## exp: unet-gather-w1-np32-nm10
#python main.py --loss-weight 1.0 --n-max-pos 32 \
#               --neg-multiplier 10 --exp unet-gather-w1-np32-nm10 --logging
#
## NOTE: change learning schedule [60,80,90] --> [80,100,110]
##       change batch size 16/gpu -> 32/gpu
## exp: unet-gather-w1-np64-nm4
#python main.py --loss-weight 1.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-gather-w1-np64-nm4 --logging

## exp: unet-gather-w1-np64-nm4-onlybd
#python main.py --loss-weight 1.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-gather-w1-np64-nm4-onlybd --logging \
#               --boundary-aware --only-boundary
#
# exp: unet-gather-w1-np64-nm4-bd-linear
python main.py --loss-weight 1.0 --n-max-pos 64 \
               --neg-multiplier 4 --exp unet-gather-w1-np64-nm4-bd-linear --logging \
               --boundary-aware --sampling-type linear
#
## exp: unet-gather-w1-np64-nm8
#python main.py --loss-weight 1.0 --n-max-pos 64 \
#               --neg-multiplier 8 --exp unet-gather-w1-np64-nm8 --logging

## exp: unet-gather-baseline
#python main.py --loss-weight 0.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-gather-baseline --logging
#
## exp: unet-gather-w1-np8-nm4
#python main.py --loss-weight 1.0 --n-max-pos 8 \
#               --neg-multiplier 4 --exp unet-gather-w1-np8-nm4 --logging
#
## exp: unet-gather-w1-np16-nm4
#python main.py --loss-weight 1.0 --n-max-pos 16 \
#               --neg-multiplier 4 --exp unet-gather-w1-np16-nm4 --logging
#
## exp: unet-gather-w1-np32-nm4
#python main.py --loss-weight 1.0 --n-max-pos 32 \
#               --neg-multiplier 4 --exp unet-gather-w1-np32-nm4 --logging
#
## exp: unet-gather-w1-np64-nm4
#python main.py --loss-weight 1.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-gather-w1-np64-nm4 --logging

## exp: unet-gather-w1-np8-nm4-onlybd
#python main.py --loss-weight 1.0 --n-max-pos 8 \
#               --neg-multiplier 4 --exp unet-gather-w1-np8-nm4-onlybd --logging \
#               --boundary-aware --only-boundary
#
## exp: unet-gather-w1-np8-nm4-bd
#python main.py --loss-weight 1.0 --n-max-pos 8 \
#               --neg-multiplier 4 --exp unet-gather-w1-np8-nm4-bd --logging \
#               --boundary-aware
