#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

NAME="fast_dbox_data"
EXT=".tgz"
FILE=$NAME$EXT
URL=https://www.dropbox.com/s/g0hjez84mwb2g2i/fast_dbox_data.tgz?dl=0
CHECKSUM=1489ffc9a84508fde71259cd76d3208b

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading Fast DeepBoxes for COCO train, val, and test-dev set..."

wget $URL -O $FILE

echo "Unzipping..."
tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."

