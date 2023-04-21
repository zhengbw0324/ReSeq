import os
import pickle
from logging import getLogger

from data.dataloader import BiSeqRecTrainDataloader, BiSeqRecEvalDataloader

from data.dataset import BiSeqRecDataset
from utils.logger import set_color


def create_dataset(config):

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        logger = getLogger()
        logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
        return dataset
    dataset = BiSeqRecDataset(config)
    if config['save_dataset']:
        dataset.save()
    return dataset


def data_preparation(config, dataset):

    train_dataset = dataset.set_phase("train")
    valid_u_dataset = dataset.set_phase("valid-user")
    valid_i_dataset = dataset.set_phase("valid-item")
    test_u_dataset = dataset.set_phase("test-user")
    test_i_dataset = dataset.set_phase("test-item")

    train_data = BiSeqRecTrainDataloader(config, train_dataset, shuffle=True)
    valid_u_data = BiSeqRecEvalDataloader(config, valid_u_dataset, shuffle=False)
    valid_i_data = BiSeqRecEvalDataloader(config, valid_i_dataset, shuffle=False)
    test_u_data = BiSeqRecEvalDataloader(config, test_u_dataset, shuffle=False)
    test_i_data = BiSeqRecEvalDataloader(config, test_i_dataset, shuffle=False)

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow')
    )
    return train_data, (valid_u_data, valid_i_data), (test_u_data, test_i_data)

