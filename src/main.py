import argparse
import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path
import os

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from SpellChecker import correct_sentence


class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def train(model: Model,
          loader: DataLoaderIAM,
          line_mode: bool,
          early_stopping: int = 25) -> None:
    """Trains NN."""
    epoch = 0  # number of training epochs since start
    summary_char_error_rates = []
    summary_word_accuracies = []
    preprocessor = Preprocessor(get_img_size(line_mode), data_augmentation=True, line_mode=line_mode)
    best_char_error_rate = float('inf')  # best validation character error rate
    no_improvement_since = 0  # number of epochs no improvement of character error rate occurred
    # stop training after this number of epochs without improvement

    # create a summary writer object
    writer = tf.summary.create_file_writer("logs/")
    with writer.as_default():
        tf.summary.experimental.set_step(epoch)


    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')

        # validate
        char_error_rate, word_accuracy = validate(model, loader, line_mode)

        # write summary
        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        write_summary(summary_char_error_rates, summary_word_accuracies)

        # if best validation accuracy so far, save model parameters
        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {char_error_rate * 100.0}%')
            no_improvement_since += 1

        # stop training if no more improvement in the last x epochs
        if no_improvement_since >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: DataLoaderIAM, line_mode: bool) -> Tuple[float, float]:
    """Validates NN."""
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size(line_mode), line_mode=line_mode)
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    # print validation result
    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy

def load_different_image():
    imgs = []
    path = 'D:/Ranjila Main/Backend-SimpleHTR/croppedDir'
    myPicList = os.listdir(path)
    print(myPicList)
    # for index in range(len(myPicList)):


    for i in range(0, 29):
            # list=[6,8,14,15,16,22,23,25,26,27,28]
            list=[0,1,2,3,4,5,6,7,8,11,12,13,14,15,16,19,20,21,22,23,24,25]
            if i in list:
                # cv2.imshow('img',cv2.imread(("../croppedDir/cropped_{}.png".format(i),cv2.IMREAD_GRAYSCALE)))
            #imgs.append(preprocessor(cv2.imread("../data/urza.png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize, enhance=False))
                # img = cv2.imread("../data/urza.png".format(i), cv2.IMREAD_GRAYSCALE)
                preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
                # img = preprocessor.process_img(cv2.imread("../data/lines/selfMade_{}.png".format(i),cv2.IMREAD_GRAYSCALE
                # img = 
                imgs.append(preprocessor.process_img(cv2.imread("../croppedDir/cropped_{}.png".format(i),cv2.IMREAD_GRAYSCALE)))
                # print(imgs)
    return imgs


def infer(model: Model, fn_img: Path) -> None:
    """Recognizes text in image provided by file path."""
    # img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    # assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    # img = preprocessor.process_img(img)

    imgs = load_different_image()
    # img = img.tolist()
    
    recognizedFromInfer=[]
    counter = 0
    for i in imgs:
         counter = counter+1
         batch = Batch([i], None, 1)
         recognized, probability = model.infer_batch(batch, True)
         correct_string = recognized[0]
         
        #  print(recognized)
         print(f'Recognized: "{recognized}"')
        #  print(f'Corrected: "{correct}"')
         print(f'Probability: {probability}')
         if (counter==9):
            correct = correct_sentence(correct_string)
            # recognizedFromInfer.append(correct)
            new_string = correct.replace(",", "")
            print(new_string)
            recognizedFromInfer.append(new_string)

         elif (counter ==14):
            correct = correct_sentence(correct_string)
            # recognizedFromInfer.append(correct)
            new_string = correct.replace(",", "")
            print(new_string)
            recognizedFromInfer.append(new_string)

         else: 
            new_string = correct_string.replace(",", "")
            print(new_string)
            recognizedFromInfer.append(new_string)

         
    return recognizedFromInfer    


def infer_by_web(path, option):
    decoderType = DecoderType.BestPath
    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(),
                  decoderType, mustRestore=False)
    recognized = infer(model, path)

    return recognized


def parse_args() -> argparse.Namespace:
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--decoder', choices=['bestpath', 'beamsearch', 'wordbeamsearch'], default='bestpath')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--fast', help='Load samples from LMDB.', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of single words.', action='store_true')
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/self-self-10.png')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--dump', help='Dump output of NN to CSV file(s).', action='store_true')

    return parser.parse_args()


def main():
    """Main function."""

    # parse arguments and set CTC decoder
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                       'beamsearch': DecoderType.BeamSearch,
                       'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping[args.decoder]

    # train the model
    if args.mode == 'train':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)

        # when in line mode, take care to have a whitespace in the char list
        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        # save characters and words
        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list, decoder_type)
        train(model, loader, line_mode=args.line_mode, early_stopping=args.early_stopping)

    # evaluate it on the validation set
    elif args.mode == 'validate':
        loader = DataLoaderIAM(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file(), decoder_type, must_restore=True)
        validate(model, loader, args.line_mode)

    # infer text on test image
    elif args.mode == 'infer':
        model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
        recognized = infer(model, args.img_file)

        return recognized

if __name__ == '__main__':
    main()
