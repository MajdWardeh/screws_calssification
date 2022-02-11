import os
from optparse import OptionParser

from classifier import Classifier

def process_args():
    usage="Usage: %prog --train|--evaluate --config_file CONFIG_FILE"
     
    parser = OptionParser(usage=usage)
    parser.add_option("--train", action="store_true", dest="train", 
                        help="train the classifier")
    parser.add_option("--evaluate", action="store_true", dest="evaluate",
                        help="evaluate the classifier")
    parser.add_option("--config_file", dest="config_file", 
                        help="a configuration file for the classifier")

    (options, args) = parser.parse_args()
    if options.train is None and options.evaluate is None:
        parser.error("--train or --evaluate flags must be provided")
    if options.config_file is None:
        parser.error("a config file must be provided")
    
    if not os.path.exists(options.config_file):
        raise FileNotFoundError('{} was not found.'.format(options.config_file))

    return options.train, options.evaluate, options.config_file


def main():
    train_flg, eval_flag, config = process_args()
    classifier = Classifier(config)
    if train_flg:
        classifier.train()
    if eval_flag:
        classifier.evaluate()


if __name__=='__main__':
	main()