import argparse
from logging import getLogger

from configurator import Config
from data.utils import create_dataset, data_preparation
from trainer import BiSeqRecTrainer
from utils.logger import set_color, init_logger
from utils.utils import get_model, init_seed


def main_process(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)

    # # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = BiSeqRecTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result_u, test_result_i = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('user test result', 'yellow') + f': {test_result_u}')
    logger.info(set_color('item test result', 'yellow') + f': {test_result_i}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result_u': test_result_u,
        'test_result_i': test_result_i
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='ReSeq', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ask', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    main_process(model=args.model, dataset=args.dataset, config_file_list=config_file_list)