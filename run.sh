
## exp: unet-baseline
#python main.py --loss-weight 0.0 --exp unet-baseline
#
## exp: unet-w10-np128-nm1
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 1 --exp unet-w10-np128-nm1
#
## exp: unet-w10-np128-nm2
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 2 --exp unet-w10-np128-nm2
#
## exp: unet-w10-np128-nm3
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w10-np128-nm3

## exp: unet-w10-np128-nm4
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 4 --exp unet-w10-np128-nm4
#
## exp: unet-w10-np256-nm1
#python main.py --loss-weight 10.0 --n-max-pos 256 \
#               --neg-multiplier 1 --exp unet-w10-np256-nm1
#
## exp: unet-w10-np256-nm2
#python main.py --loss-weight 10.0 --n-max-pos 256 \
#               --neg-multiplier 2 --exp unet-w10-np256-nm2
#
## exp: unet-w10-np256-nm3
#python main.py --loss-weight 10.0 --n-max-pos 256 \
#               --neg-multiplier 3 --exp unet-w10-np256-nm3
#
## exp: unet-w10-np256-nm4
#python main.py --loss-weight 10.0 --n-max-pos 256 \
#               --neg-multiplier 4 --exp unet-w10-np256-nm4

## exp: unet-w10-np128-nm3-logit
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w10-np128-nm3-logit

## exp: unet-w10-np128-nm3-bd
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w10-np128-nm3-bd \
#               --logging True --boundary-aware True

## exp: unet-w10-np128-nm3-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w10-np128-nm3-allbd-long-r1 \
#               --logging True --boundary-aware True

## exp: unet-w1-np128-nm3-allbd-long-t01
#python main.py --loss-weight 1.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w1-np128-nm3-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w5-np128-nm3-allbd-long-t01
#python main.py --loss-weight 5.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w5-np128-nm3-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np128-nm3-allbd-long-t01
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w10-np128-nm3-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w20-np128-nm3-allbd-long-t01
#python main.py --loss-weight 20.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w20-np128-nm3-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w1-np128-nm1-allbd-long-t01
#python main.py --loss-weight 1.0 --n-max-pos 128 \
#               --neg-multiplier 1 --exp unet-w1-np128-nm1-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w5-np128-nm1-allbd-long-t01
#python main.py --loss-weight 5.0 --n-max-pos 128 \
#               --neg-multiplier 1 --exp unet-w5-np128-nm1-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np128-nm1-allbd-long-t01
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 1 --exp unet-w10-np128-nm1-allbd-long-t01 \
#               --logging True --boundary-aware True
#
## exp: unet-w20-np128-nm1-allbd-long-t01
#python main.py --loss-weight 20.0 --n-max-pos 128 \
#               --neg-multiplier 1 --exp unet-w20-np128-nm1-allbd-long-t01 \
#               --logging True --boundary-aware True

## exp: unet-w20-np128-nm3-allbd-long-t10
#python main.py --loss-weight 20.0 --n-max-pos 128 \
#               --neg-multiplier 3 --exp unet-w20-np128-nm3-allbd-long-t10 \
#               --logging True --boundary-aware True

## exp: unet-w10-np128-nm4-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 4 --exp unet-w10-np128-nm4-allbd-long \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np128-nm6-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 128 \
#               --neg-multiplier 6 --exp unet-w10-np128-nm6-allbd-long \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np64-nm4-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-w10-np64-nm4-allbd-long \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np256-nm4-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 256 \
#               --neg-multiplier 4 --exp unet-w10-np256-nm4-allbd-long \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np64-nm4-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 64 \
#               --neg-multiplier 6 --exp unet-w10-np64-nm6-allbd-long \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np256-nm6-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 256 \
#               --neg-multiplier 6 --exp unet-w10-np256-nm6-allbd-long \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np64-nm4-allbd-long-r1
#python main.py --loss-weight 10.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-w10-np64-nm4-allbd-long-r1 \
#               --logging True --boundary-aware True
## exp: unet-w10-np64-nm4-allbd-long-r2
#python main.py --loss-weight 10.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-w10-np64-nm4-allbd-long-r2 \
#               --logging True --boundary-aware True
## exp: unet-w10-np64-nm4-allbd-long-r3
#python main.py --loss-weight 10.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-w10-np64-nm4-allbd-long-r3 \
#               --logging True --boundary-aware True
## exp: unet-w10-np64-nm4-allbd-long-r4
#python main.py --loss-weight 10.0 --n-max-pos 64 \
#               --neg-multiplier 4 --exp unet-w10-np64-nm4-allbd-long-r4 \
#               --logging True --boundary-aware True
#
## exp: unet-w10-np32-nm4-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 32 \
#               --neg-multiplier 4 --exp unet-w10-np32-nm4-allbd-long \
#               --logging True --boundary-aware True
## exp: unet-w10-np32-nm6-allbd-long
#python main.py --loss-weight 10.0 --n-max-pos 32 \
#               --neg-multiplier 6 --exp unet-w10-np32-nm6-allbd-long \
#               --logging True --boundary-aware True


