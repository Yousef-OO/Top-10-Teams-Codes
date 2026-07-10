#!/bin/bash

BASEDIR=$(dirname "$0")

# Copy a subset of the files input the challenge input directory
cp -Rf $BASEDIR/temp/HR0000* $BASEDIR/../reduced_training_data/
cp -Rf $BASEDIR/temp/A000* $BASEDIR/../reduced_training_data/
cp -Rf $BASEDIR/temp/I000* $BASEDIR/../reduced_training_data/
cp -Rf $BASEDIR/temp/E0000* $BASEDIR/../reduced_training_data/
cp -Rf $BASEDIR/temp/S000* $BASEDIR/../reduced_training_data/
cp -Rf $BASEDIR/temp/Q000* $BASEDIR/../reduced_training_data/

# Copy a subset of the files input the challenge input directory
cp -Rf $BASEDIR/temp/HR0001* $BASEDIR/../reduced_test_data/
cp -Rf $BASEDIR/temp/A001* $BASEDIR/../reduced_test_data/
cp -Rf $BASEDIR/temp/I001* $BASEDIR/../reduced_test_data/
cp -Rf $BASEDIR/temp/E0001* $BASEDIR/../reduced_test_data/
cp -Rf $BASEDIR/temp/S001* $BASEDIR/../reduced_test_data/
cp -Rf $BASEDIR/temp/Q001* $BASEDIR/../reduced_test_data/