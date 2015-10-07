#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=slid_window_data.tgz
URL=http://www.cs.berkeley.edu/~wckuo/fast-dbox-data/$FILE
CHECKSUM=351c1088ec4362f8a02df3bfd041a896

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

echo "Downloading Edge boxes for COCO train, val, and test-dev set..."

wget $URL -O $FILE

echo "Unzipping..."

#tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."

