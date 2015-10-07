#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

NAME="coco_matlab_data"
EXT=".tgz"
FILE=$NAME$EXT
URL=http://www.cs.berkeley.edu/~wckuo/fast-dbox-data/$FILE
CHECKSUM=a731fa22fb3833033f9475e7558bf9c4

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

echo "Downloading matlab COCO image ids and ground truth bboxes ..."

wget $URL -O $FILE

echo "Unzipping..."
tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."

