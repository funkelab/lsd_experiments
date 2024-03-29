# data

samples{A, B, C}.hdf from CREMI

# train

## setup01

affinities, euclidean loss, mala unet, 5 fmul 

## setup02

local shape descriptors, euclidean loss, mala unet, 5 fmul

## setup03

affinities from LSDs, euclidean loss, mala unet, 5 fmul

## setup04

affinities + LSDs, euclidean loss, mala unet, 5 fmul

## setup05

affinities, euclidean loss, larissa's unet, 5 fmul

## setup06

LSDs, euclidean loss, larissa's unet, 5 fmul

## setup07

affinities from LSDs, euclidean loss, larissa's unet, 5 fmul

## setup08

affinities + LSDs , euclidean loss, larissa's unet, 5 fmul

## setup09

affinities, euclidean loss, mala unet, 6 fmul

## setup10

LSDs, euclidean loss, mala unet, 6 fmul

## setup11

affs from LSDs, euclidean loss, mala unet, 6 fmul

## setup12

affs + LSDs, euclidean loss, mala unet, 6 fmul

## setup13

affs, euclidean + malis loss, mala unet, 5 fmul

## setup14

affs from LSDs, euclidean + malis loss, mala unet, 5 fmul

## setup15

affs + LSDs, euclidean + malis loss, mala unet, 5 fmul

## setup16

affs, euclidean + malis loss, larissa's unet, 6 fmul

## setup17

affs from LSDs, euclidean + malis loss, larissa's unet, 6 fmul

## setup18

affs + LSDs, euclidean + malis loss, larissa's unet, 6 fmul

## setup19

affs, euclidean + malis loss, mala unet, 6 fmul

## setup20

affs from LSDs, euclidean + malis loss, mala unet, 6 fmul

## setup21

affs + LSDs, euclidean + malis loss, mala unet, 6 fmul

## setup22

long range affs, euclidean loss, mala unet, 5 fmul

## setup23

long range affs, euclidean loss, larissa's unet, 6 fmul

## setup24

long range affs, euclidean loss, mala unet, 6 fmul

## setup25

long range affs, euclidean + malis loss, mala unet, 5 fmul

## setup26

long range affs, euclidean + malis loss, larissa's unet, 6 fmul

## setup27

long range affs, euclidean + malis loss, mala unet, 6 fmul

## setup28

long range affs from lsds, euclidean loss, mala unet, 5 fmul

## setup29

long range affs from lsds, euclidean loss, larissa's unet, 6 fmul

## setup30

long range affs from lsds, euclidean loss, mala unet, 6 fmul

## setup31

long range affs from lsds, euclidean + malis loss, mala unet, 5 fmul

## setup32

long range affs from lsds, euclideam + malis loss, larissa's unet, 6 fmul

## setup33

long range affs from lsds, euclidean + malis loss, mala unet, 6 fmul

## setup34

long range affs + lsds, euclidean loss, mala unet, 5 fmul

## setup35 

long range affs + lsds, euclidean loss, larissa's unet, 6 fmul

## setup36

long range affs + lsds, euclidean loss, mala unet, 6 fmul

## setup37

long range affs + lsds, euclidean + malis loss, mala unet, 5 fmul

## setup38

long range affs + lsds, euclidean + malis loss, larissa's unet, 6 fmul

## setup39

long range affs + lsds, euclidean + malis loss, mala unet, 6 fmul

## setup40

affs + lsds from raw + lsds, euclidean loss, mala unet, 5 fmul

## setup41

affs + lsds from raw + lsds, eulcidean loss, larissa's unet, 6 fmul

## setup42

affs + lsds from raw + lsds, euclidean loss, mala unet, 6 fmul

## setup43

affs + lsds from raw + lsds, euclidean + malis loss, mala unet, 5 fmul

## setup44

affs + lsds from raw + lsds, euclidean + malis loss, larissa's unet, 6 fmul

## setup45

affs + lsds from raw + lsds, euclidean + malis loss, mala unet, 6 fmul

## setup46

long range affs + lsds from raw + lsds, euclidean loss, mala unet, 5 fmul

## setup47

long range affs + lsds from raw + lsds, euclidean loss, larissa's unet, 6 fmul

## setup48

long range affs + lsds from raw + lsds, euclidean loss, mala unet, 6 fmul

## setup49

long range affs + lsds from raw + lsds, euclidean + malis loss, mala unet, 5 fmul

## setup50

long range affs + lsds from raw + lsds, euclidean + malis loss, larissa's unet, 6 fmul

## setup51

long range affs + lsds from raw + lsds, euclidean + malis loss, larissa's unet, 6 fmul

## setup52_f

copy of setup34 with increased feature maps, full training volumes 

## setup53_f 

copy of setup40 with increased feature maps, full training volumes, lsds from setup02

## setup54_f

copy of setup46 with increased feature maps, full training volumes, lsds from setup02

## setup55_g

copy of setup04 - glial mask training on all 9 volumes (6 cremi, 3 scott's)

## setup56_g

copy of setup22 - glial mask training on all 9 volumes (6 cremi, 3 scott's)


## setup57_g

copy of setup52_f - glial mask training on all 9 volumes (6 cremi, 3 scott's)

# LSD paper setups

## setup58_p

vanilla affs, eucl loss, {A,B,C}, all labels, no autocontext

## setup59_p

vanilla affs, malis loss, {A,B,C}, all labels, no autocontext

## setup60_p

long range affs, {A,B,C}, all labels, no autocontext

## setup61_p

affs + lsds, {A,B,C}, all labels, no autocontext

## setup62_p

vanilla affs, eucl loss, {A,B,C,A+,B+,C+,0,1,2}, all labels, no autocontext

## setup63_p

vanilla affs, malis loss, {A,B,C,A+,B+,C+,0,1,2}, all labels, no autocontext

## setup64_p

long range affs, {A,B,C,A+,B+,C+,0,1,2}, all labels, no autocontext

## setup65_p

affs + lsds, {A,B,C,A+,B+,C+,0,1,2}, all labels, no autocontext

## setup66_p

vanilla affs, eucl loss, {A,B,C}, no glia, no autocontext

## setup67_p

vanilla affs, malis loss, {A,B,C}, no glia, no autocontext

## setup68_p

long range affs, {A,B,C}, no glia, no autocontext

## setup69_p

affs + lsds, {A,B,C}, no glia, no autocontext

## setup70_p

vanilla affs, eucl loss, {A,B,C,A+,B+,C+,0,1,2}, no glia, no autocontext

## setup71_p

vanilla affs, malis loss, {A,B,C,A+,B+,C+,0,1,2}, no glia, no autocontext

## setup72_p

long range affs, {A,B,C,A+,B+,C+,0,1,2}, no glia, no autocontext

## setup73_p

affs + lsds, {A,B,C,A+,B+,C+,0,1,2}, no glia, no autocontext

## setup74_p

vanilla affs, eucl loss, {A,B,C}, all labels, autocontext

## setup75_p

vanilla affs, malis loss, {A,B,C}, all labels, autocontext

## setup76_p

long range affs, {A,B,C}, all labels, autocontext

## setup77_p

affs + lsds, {A,B,C}, all labels, autocontext

## setup78_p

vanilla affs, eucl loss, {A,B,C,A+,B+,C+,0,1,2}, all labels, autocontext

## setup79_p

vanilla affs, malis loss, {A,B,C,A+,B+,C+,0,1,2}, all labels, autocontext

## setup80_p

long range affs, {A,B,C,A+,B+,C+,0,1,2}, all labels, autocontext

## setup81_p

affs + lsds, {A,B,C,A+,B+,C+,0,1,2}, all labels, autocontext

## setup82_p

vanilla affs, eucl loss, {A,B,C}, no glia, autocontext

## setup83_p

vanilla affs, malis loss, {A,B,C}, no glia, autocontext

## setup84_p

long range affs, {A,B,C}, no glia, autocontext

## setup85_p

affs + lsds, {A,B,C}, no glia, autocontext

## setup86_p

vanilla affs, eucl loss, {A,B,C,A+,B+,C+,0,1,2}, no glia, autocontext

## setup87_p

vanilla affs, malis loss, {A,B,C,A+,B+,C+,0,1,2}, no glia, autocontext

## setup88_p

long range affs, {A,B,C,A+,B+,C+,0,1,2}, no glia, autocontext

## setup89_p

affs + lsds, {A,B,C,A+,B+,C+,0,1,2}, no glia, autocontext

## setup90_p

lsds for setup77_p (see setup02 + setup40 for reference)

## setup91_p

lsds for setup81_p

## setup92_p

lsds for setup85_p

## setup93_p 

lsds for setup89_p

## setup94_p

copy of setup62_p using 6 data samples {A,B,C,0,1,2}

## setup95_p

copy of setup63_p using 6 data samples {A,B,C,0,1,2}

## setup96_p

copy of setup64_p using 6 data samples {A,B,C,0,1,2}

## setup97_p

copy of setup65_p using 6 data samples {A,B,C,0,1,2}

## setup98_p

copy of setup70_p using 6 data samples {A,B,C,0,1,2}

## setup99_p

copy of setup71_p using 6 data samples {A,B,C,0,1,2}

## setup100_p

copy of setup72_p using 6 data samples {A,B,C,0,1,2}

## setup101_p

copy of setup73_p using 6 data samples {A,B,C,0,1,2}

## setup102_p

copy of setup78_p using 6 data samples {A,B,C,0,1,2}

## setup103_p

copy of setup79_p using 6 data samples {A,B,C,0,1,2}

## setup104_p

copy of setup80_p using 6 data samples {A,B,C,0,1,2}

## setup105_p

copy of setup81_p using 6 data samples {A,B,C,0,1,2}

## setup106_p

copy of setup86_p using 6 data samples {A,B,C,0,1,2}

## setup107_p

copy of setup87_p using 6 data samples {A,B,C,0,1,2}

## setup108_p

copy of setup88_p using 6 data samples {A,B,C,0,1,2}

## setup109_p

copy of setup89_p using 6 data samples {A,B,C,0,1,2}

## setup110_p

copy of setup91_p using 6 data samples {A,B,C,0,1,2}

## setup111_p

copy of setup93_p using 6 data samples {A,B,C,0,1,2}

## setup112_p

copy of setup77_p using 14 fmaps out in unet

## setup113_p

copy of setup81_p using 14 fmaps out in unet

## setup114_p

copy of setup85_p using 14 fmaps out in unet

## setup115_p

copy of setup89_p using 14 fmaps out in unet

## setup116_p

copy of setup105_p using 14 fmaps out in unet

## setup117_p

copy of setup109_p using 14 fmaps out in unet
