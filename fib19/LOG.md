# data

training on cubes{01, 02, 03}.hdf from FIB19 data

# train

## setup01

vanilla unet affinity prediction

## setup02

predicting 1st and 2nd moment 10d local shape descriptors (LSDs)

## setup03

predicting affinities from LSDs

## setup04

predicting affinities and LSDs together

## setup05

train setup02 to produce LSDs,
then predict affinities and LSDs from raw input + LSD predictions from setup02 in an autocontext setup
