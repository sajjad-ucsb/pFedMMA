#!/bin/bash

cd ../

# custom config
DATA="../DATA/"
MODEL=FedAdapter
TRAINER=FedAdapter
OT=COT
TOP_PERCENT=0.80
EPS=0.1
# SUB=new
THRESH=0.01
MAX_ITER=100
PRETRAINED=True
LR=0.001
GAMMA=1
LOGITS2=False
USERS=8
FRAC=1
ROUND=50
LOCAL_EPOCH=1
NUM_PROMPT=2
N_DOMAIN=4
SPLIT_CLIENT=True
IMBALANCE_TRAIN=True 
#DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
IID=False
CSC=False  # class-specific context (False or True)
USEALL=True
TEMP=0.5
MU=0.5
TARGET_DOMAIN='amazon'
#SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
# oxford_pets dtd oxford_flowers caltech101 food101
for DATASET in office
do
  for BETA in 0.3 
    do
      for LR in  0.001
        do
          for ADAPTER_SCALE in 0.001 
          do
            for SEED in 2 3
            do
#              DIR=output_base/${DATASET}_mu${MU}/${MODEL}_${TRAINER}_${BOTTLENECK}neck/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}
             DIR_TRAIN=pFedMMA/Office/train/${DATASET}/${MODEL}_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}_loca${LOCAL_EPOCH}_beta${BETA}_scale${ADAPTER_SCALE}
            #  DIR_TEST=pFedMMA/Office/test/${DATASET}/${MODEL}_${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}_loca${LOCAL_EPOCH}_beta${BETA}_scale${ADAPTER_SCALE}
             
#               DIR=output_base/${DATASET}/${MODEL}_${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
              if [ -d "$DIR_TRAIN" ]; then
                echo "Oops! The results exist at ${DIR_TRAIN} (so skip this job)"
              else
                python federated_main.py \
                --root ${DATA} \
                --model ${MODEL} \
                --seed ${SEED} \
                --num_users ${USERS} \
                --frac ${FRAC} \
                --lr ${LR} \
                --mu ${MU} \
                --temp ${TEMP} \
                --logits2 ${LOGITS2} \
                --OT ${OT} \
                --top_percent ${TOP_PERCENT} \
                --eps ${EPS} \
                --thresh ${THRESH} \
                --max_iter ${MAX_ITER} \
                --trainer ${TRAINER} \
                --round ${ROUND} \
                --target_domain ${TARGET_DOMAIN} \
                --local_epoch ${LOCAL_EPOCH} \
                --num_prompt ${NUM_PROMPT} \
                --train_batch_size 32 \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR_TRAIN} \
                --num_domain ${N_DOMAIN} \
                --split_client ${SPLIT_CLIENT} \
                --imbalance_train ${IMBALANCE_TRAIN} \
                DATASET.USEALL True
              fi

              # if [ -d "$DIR_TEST" ]; then
              #   echo "Oops! The results exist at ${DIR_TEST} (so skip this job)"
              # else
              #   python federated_main.py \
              #   --root ${DATA} \
              #   --model ${MODEL} \
              #   --seed ${SEED} \
              #   --num_users ${USERS} \
              #   --frac ${FRAC} \
              #   --lr ${LR} \
              #   --mu ${MU} \
              #   --temp ${TEMP} \
              #   --logits2 ${LOGITS2} \
              #   --OT ${OT} \
              #   --top_percent ${TOP_PERCENT} \
              #   --eps ${EPS} \
              #   --thresh ${THRESH} \
              #   --max_iter ${MAX_ITER} \
              #   --trainer ${TRAINER} \
              #   --round ${ROUND} \
              #   --target_domain ${TARGET_DOMAIN} \
              #   --local_epoch ${LOCAL_EPOCH} \
              #   --num_prompt ${NUM_PROMPT} \
              #   --train_batch_size 32 \
              #   --dataset-config-file configs/datasets/${DATASET}.yaml \
              #   --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
              #   --num_domain ${N_DOMAIN} \
              #   --split_client ${SPLIT_CLIENT} \
              #   --imbalance_train ${IMBALANCE_TRAIN} \
              #   --output-dir ${DIR_TEST} \
              #   --eval-only \
              #   --model-dir ${DIR_TRAIN} \
              #   DATASET.USEALL True
              # fi
            done
          done
        done
    done
done

