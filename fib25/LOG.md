# data

training on trvol-{250-1, 250-2}, tstvol-{520-1, 520-2}.hdf from FIB25 data

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

## setup06

setup01 with MALIS loss

## setup07

setup02 with MALIS loss

## setup08

setup03 with MALIS loss

## setup09

setup04 with MALIS loss

## setup10

setup05 with MALIS loss
