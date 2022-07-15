import importlib
from random import shuffle
import torch.utils.data
from data.base_dataset import BaseDataSet
from data.face_dataset import FaceTestDataSet
def create_dataloader(opt):

    instance = FaceTestDataSet()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" % (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.Dataloader(
        instance, 
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
    )
    return dataloader
    