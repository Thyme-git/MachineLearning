from Adaboost import Adaboost
from DataReader import DataReader
import numpy as np
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Adaboost implementation based on DecisionStump and LogisticRegression')
    parser.add_argument('-t', '--task', type=str, choices=['train', 'predict'], default='train', help='train or predict, if predict, a weight file must be specified')
    parser.add_argument('-c', '--classifier', type=str,
                        choices=['DecisionStump', 'LogisticRegression'],
                        default='DecisionStump',
                        help='base classifier used in Adaboost algorithm')
    parser.add_argument('-d', '--data-file', type=str, default='../data/data.csv',
                        help='file storing features of data set, labels not included')
    parser.add_argument('-l', '--label-file', type=str, default='../data/targets.csv',
                        help='file storing labels of data set')
    parser.add_argument('-f', '--weight-file', type=str, default='../weight/AdaboostWeight.npy',
                        help='file storing weight of the pretrained model')
    parser.add_argument('-o', '--output-dir', type=str, default='../data/experiments',
                        help='prediction result dir')
    return parser.parse_args()

def main():
    args = get_args()
    base_classifier     = args.classifier
    data_path           = args.data_file
    target_path         = args.label_file
    weight_path         = args.weight_file
    pred_dir            = args.output_dir
    
    base_list = [1, 5, 10, 100]
    dataReader = DataReader(data_path, target_path)
    if args.task == 'train':
        for i in range(1, 11):
            print(f'training on fold {i}')
            best_acc = 0
            accuracy = []
            X_train, y_train, index_train,\
            X_validate , y_validate , index_validate = dataReader.get_train_validate_set(i-1)
            
            adaboost = Adaboost(base_classifier, 100)
            adaboost.fit(X_train, y_train)
            
            for n_base_classifier in base_list:
                print(f'use {n_base_classifier} classifiers:')

                y_pred = adaboost.predict(X_validate, n_base_classifier)
    
                acc = (y_pred == y_validate).sum() / len(y_validate)
                y_pred[np.argwhere(y_pred == -1.0)] = 0.0

                data_pred = np.hstack([index_validate.reshape((-1, 1)), y_pred.reshape((-1, 1))])
                np.savetxt(pred_dir+'/base%d_fold%d.csv' % (n_base_classifier, i), data_pred, delimiter=',')
                print(f'Accuracy of validate set : {acc*100:.2f}% ({len(X_validate)} samples)')
                accuracy.append(acc)
                if acc > best_acc:
                    best_acc = acc
                    adaboost.save_weight(weight_path)
            print(f'Average accuracy : {100*np.array(accuracy).mean():.2f}%')
            print('')
    
    if args.task == 'predict':
        adaboost = Adaboost(base_classifier)
        adaboost.load_weight(weight_path)
        
        X_test, y_test, index_test = dataReader.get_test_set()
        y_pred = adaboost.predict(X_test, 100)
        
        acc = (y_pred == y_test).sum() / len(y_test)
        
        y_pred[np.argwhere(y_pred == -1.0)] = 0.0
        data_pred = np.hstack([index_test.reshape((-1, 1)), y_pred.reshape((-1, 1))])
        
        np.savetxt(pred_dir+'/pred.csv', data_pred, delimiter=',')
        print(f'Accuracy : {100*acc:.2f}%')
        print('Prediction output file at '+pred_dir+'/pred.csv')

if __name__ == '__main__':
    main()