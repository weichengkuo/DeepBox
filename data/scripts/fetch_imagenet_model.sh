#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

NAME="imagenet_model"
EXT=".tgz"
FILE=$NAME$EXT
URL=http://www.cs.berkeley.edu/~wckuo/fast-dbox-data/$FILE
CHECKSUM=8a30a5e0d8dd4f07d475e672d61ecb7a

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

echo "Downloading Alex net Imagenet model for initialization of DeepBox training..."

wget $URL -O $FILE

echo "Unzipping..."
tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."

