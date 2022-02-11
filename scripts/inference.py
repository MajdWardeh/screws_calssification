import os
import shutil
from optparse import OptionParser
from classifier import Classifier

def process_args():
    usage = 'usage: %prog --input_dir INPUT_DIR --output_dir OUTPUT_DIR [--weights WEIGHTS]'

    parser = OptionParser(usage=usage)
    parser.add_option("--input_dir", dest="input_dir", 
                        help="The path to the directory which contains screw images that needs to be classified")
    parser.add_option("--output_dir", dest="output_dir",
                        help="The path to the directory which the classified images will be moved to one of its subdirectories")
    parser.add_option("--config_file", dest="config_file", 
                        help="a configuration file for the classifier")


    (options, args) = parser.parse_args()
    if options.input_dir is None or options.output_dir is None:
        parser.error("input and output directories must be provided")

    if options.config_file is None:
        parser.error("a config file must be provided")

    return options.input_dir, options.output_dir, options.config_file


def main():
    in_dir, out_dir, config = process_args()

    classifier = Classifier(config)
    predicted_classes_dict = classifier.classify(in_dir)

    for img_name, pred_class in predicted_classes_dict.items():
        from_path = os.path.join(in_dir, img_name)
        to_path = os.path.join(out_dir, pred_class, img_name)
        shutil.move(from_path, to_path)

    print('finished')

if __name__=='__main__':
    main()