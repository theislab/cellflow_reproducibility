SUB_SCRIPT=/home/fleckj/projects/cellflow/scripts/data/submit/preproc_and_map_split.sh
DATA_DIR=/home/fleckj/projects/cellflow/data/datasets/organoids_combined/

find $DATA_DIR -mindepth 3 -maxdepth 3 -type d | while read d;
do
    echo $d
    THIS_ENV="all, SPLIT_DIR=$d"
    bsub -env "$THIS_ENV" < $SUB_SCRIPT
done

