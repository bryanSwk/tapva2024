#!/bin/bash

# makedir
[ -d weights ] || mkdir weights

cd weights

curl -O https://transfer.sh/get/TOOQxVLVYt/FastSAM-x.pt

curl -O https://transfer.sh/Gx2iyrtMwF/ViT-B-32.pt