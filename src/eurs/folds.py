"""leave-scene-out 折切分。20 個訓練場景固定分 4 折；elasmobranch（ray1、sharkFish）分在不同折，
other（seahorse1、starfish2、turtles1）盡量分散。固定寫死以便跨實驗可比。"""
TRAIN_SCENES = ['fish1', 'fish2', 'fish4', 'fish5', 'fish7', 'fish9', 'fish11', 'fish12', 'fish13', 'fish14',
                'fish15', 'fish17', 'fish18', 'fish20', 'fish21', 'ray1', 'seahorse1', 'sharkFish', 'starfish2', 'turtles1']
DEV_SCENES = ['fish3', 'fish6', 'fish10', 'fish19', 'sharRayFish', 'starfish1']

FOLDS = {
    0: ['fish1', 'fish11', 'fish13', 'fish20', 'ray1'],
    1: ['fish2', 'fish12', 'fish15', 'fish21', 'sharkFish'],
    2: ['fish4', 'fish14', 'fish17', 'fish9', 'seahorse1'],
    3: ['fish5', 'fish7', 'fish18', 'starfish2', 'turtles1'],
}
assert sorted(sum(FOLDS.values(), [])) == sorted(TRAIN_SCENES)


def split(fold):
    """回傳 (train_scenes, val_scenes)。fold=-1 → 全部訓練、無驗證。"""
    if fold < 0:
        return list(TRAIN_SCENES), []
    val = FOLDS[fold]
    return [s for s in TRAIN_SCENES if s not in val], list(val)
