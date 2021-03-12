#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

declare -A MAP=( ["nq_reader_base"]="https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_base.tar.gz"\
                 ["nq_reader_large"]="https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_reader_large.tar.gz"\
                 ["nq_retriever"]="https://dl.fbaipublicfiles.com/FiD/pretrained_models/nq_retriever.tar.gz"\
                 ["tqa_reader_base"]="https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_base.tar.gz"\
                 ["tqa_reader_large"]="https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_reader_large.tar.gz"\
                 ["tqa_retriever"]="https://dl.fbaipublicfiles.com/FiD/pretrained_models/tqa_retriever.tar.gz"\ )

ROOT="pretrained_models"
allkeys=""
for key in "${!MAP[@]}"; do allkeys+="$key "; done

while [ "$#" -gt 0 ]; do
    arg=$1
    case $1 in
        # convert "--opt=the value" to --opt "the value".
        # the quotes around the equals sign is to work around a
        # bug in emacs' syntax parsing
        --*'='*) shift; set -- "${arg%%=*}" "${arg#*=}" "$@"; continue;;
        -m|--model) shift; MODEL=$1;;
        -o|--output) shift; ROOT=$1;;
        -h|--help) usage; exit 0;;
        --) shift; break;;
        -*) usage_fatal "unknown option: '$1'";;
        *) break;; # reached the list of file names
    esac
    shift || usage_fatal "option '${arg}' requires a value"
done

if ! [ -v MAP[$MODEL] ];then
    echo "Model id not recognized, available models:";
    echo $allkeys;
    exit 1;
fi


mkdir -p "${ROOT}"
echo "Saving pretrained model in "$ROOT""

wget -qO- "${MAP[$MODEL]}" | tar xvz -C $ROOT
