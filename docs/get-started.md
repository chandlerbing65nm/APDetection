# Model Start

This page provides the basic tutorials for training and testing oriented models.

## Huge Image Demo

A demo script is provided to test a single huge image, like DOTA images.

```shell
python demo/huge_image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SPLIT_CONFIG_FILE} \
	 [--device ${GPU_ID}] [--score-thr ${SCORE_THR}]
```
**note**: `${SPLIT_CONFIG_FILE}` is from `BBoxToolkit`. Refer below to get the split config. \
`e.g. BboxToolkit/tools/split_configs/dota1_0/ss_dota_test.json`

## Prepare dataset

All config files of oriented object datasets are put at `APDetection/configs/obb/_base_/dataset`. Before training and testing, you need to add the dataset path to config files.

Especially, DOTA dataset need to be splitted and add the splitted dataset path to DOTA config files. There is a script `img_split.py` at `APDetection/BboxToolkit/tools/` to split images and generate patch labels.
The simplest way to use `img_split.py` is loading the json config in `BboxToolkit/tools/split_configs`. Please refer to [USAGE.md](https://github.com/jbwang1997/BboxToolkit/USAGE.md) for the details of `img_split.py`.

**example**
```shell
cd BboxToolkit/tools/
# modify the img_dirs, ann_dirs and save_dir in split_configs/dota1_0/ss_dota_train.json
python img_split.py --base_json split_configs/dota1_0/ss_dota_train.json
```

**note**: the `ss` and `ms` mean `single and multi scale splitting`, repectively.

## Training

```shell
# one GPU training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multiple GPUs traing
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**note**: We test our model on 1 GPU and with batch size of 2. the basic learning rate is 0.005 for SGD. if your training batch size is different from ours, please remember to change the learing rate based the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

## Testing

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```
**note**: `--eval` and `--show` function are not applicable if you trained the trainval-test in DOTA dataset. Use these arguments only if you use tran-val-test. Also, you can add `CUDA_VISIBLE_DEVICES=<gpu_ids> PORT=<port_number>` in multi-gpu distributed training.

```shell
# multi-gpu testing
CUDA_VISIBLE_DEVICES=<gpu_ids> PORT=<port_number> ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```

If you use DOTA dataset, you should convert and merge bounding boxes from the patch coordinate system to the full image coordinate system.
This function is merged in the testing process. It can automatically generate full image results without running other program.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --options save_dir=${SAVE_DIR} nproc=1
```
where, `${SAVE_DIR}` is the output path for full image results. You can set it to your working directory `work_dirs`.
