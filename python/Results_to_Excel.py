'''
Trabajo Final de Grado
Guarda los datos de los ficheros output.out de la optimización de PAMAP2 en un excel
Author: Javier López Iniesta Díaz del Campo

Editado por José Manuel Bravo para poder añadir un campo extra al guardar los registros. Este campo extra puede guardar configuración adicional del entrenamiento.
'''
import numpy as np
import pandas as pd
import math
import os
import datetime
import sys

import openpyxl
from openpyxl.styles import PatternFill
from openpyxl.styles import Font

import xlutils
from xlutils.copy import copy
import xlrd
import xlwt

output_path = sys.argv[1]
output_file = sys.argv[1] + "output.out"

dataset = sys.argv[2]
approach = sys.argv[3]
network_type = sys.argv[4]
final_points = int(sys.argv[5])
num_classes = int(sys.argv[6])
norm_type = sys.argv[7]
epochs = sys.argv[8]
batch_size = sys.argv[9]
if (len(sys.argv) == 11):
    extraSettings = sys.argv[10]
else:
    extraSettings = ""

output_excel_file = '../output/Results.xls'
# output_excel_file = '../output/Results_HHAR_v3.xls'

exists = os.path.isfile(output_excel_file)

if not exists:
    exp_num = 1
else:
    original_df = pd.read_excel(output_excel_file)
    rows = (len(original_df))
    exp_num = rows+1


accuracy_train_total_text = ': Train accuracy'
accuracy_F1score_train_text = ': Train fmeasure weighted'
num_training_examples_total_text = ': Number of training examples'

accuracy_test_total_text = ': Test accuracy'
accuracy_F1score_test_text = ': Test fmeasure weighted'
num_testing_examples_total_text = ': Number of testing examples'


with open(output_file) as search:
    for line in search:
        line = line.rstrip()  # remove '\n' at end of line
        if accuracy_F1score_test_text in line:
            accuracy_F1score_test = float(
                line.replace(accuracy_F1score_test_text, ''))
        elif accuracy_test_total_text in line:
            accuracy_test_total = float(
                line.replace(accuracy_test_total_text, ''))
        elif num_testing_examples_total_text in line:
            num_testing_examples_total = int(
                line.replace(num_testing_examples_total_text, ''))
        elif accuracy_train_total_text in line:
            accuracy_train_total = float(
                line.replace(accuracy_train_total_text, ''))
        elif accuracy_F1score_train_text in line:
            accuracy_F1score_train = float(
                line.replace(accuracy_F1score_train_text, ''))
        elif num_training_examples_total_text in line:
            num_training_examples_total = int(
                line.replace(num_training_examples_total_text, ''))

z = 1.96  # Distribución normal con un 95 % de intervalo de confianza
CI_acc_test = float(z*math.sqrt((accuracy_test_total *
                    (1-accuracy_test_total))/num_testing_examples_total)*100)

CI_acc_test_F1_score = float(
    z*math.sqrt((accuracy_F1score_test*(1-accuracy_F1score_test))/num_testing_examples_total)*100)

accuracy_test = str(round(accuracy_test_total*100, 2)) + \
    " ± " + str(round(CI_acc_test, 2))

accuracy_test_F1_score = str(round(
    accuracy_F1score_test*100, 2)) + " ± " + str(round(CI_acc_test_F1_score, 2))


CI_acc_train = float(z*math.sqrt((accuracy_train_total *
                     (1-accuracy_train_total))/num_training_examples_total)*100)

CI_acc_train_F1_score = float(
    z*math.sqrt((accuracy_F1score_train*(1-accuracy_F1score_train))/num_training_examples_total)*100)

accuracy_train = str(round(accuracy_train_total*100, 2)) + \
    " ± " + str(round(CI_acc_train, 2))

accuracy_train_F1_score = str(round(
    accuracy_F1score_train*100, 2)) + " ± " + str(round(CI_acc_train_F1_score, 2))


if exp_num == 1:
    data = [exp_num, dataset, approach, norm_type, network_type, final_points, num_classes, epochs, batch_size, accuracy_test,
            accuracy_test_F1_score, num_testing_examples_total, accuracy_train, accuracy_train_F1_score, num_training_examples_total, extraSettings]
    columns = ['exp_num', 'dataset', 'approach', 'norm_type', 'network_type', 'final_points', 'num_classes', 'epochs', 'batch_size',
               'acc_test', 'acc_test_F1score', 'num_testing_examples', 'acc_train', 'acc_train_F1score', 'num_training_examples', 'extra_settings']

    # data= [exp_num, dataset, evaluation, strategy_1, fs, type_norm, ws,ss,epochs,batch_size,time, accuracy_test, accuracy_test_F1_score, num_testing_examples_total, accuracy_val,accuracy_val_F1_score, num_validation_examples_total]
    # columns= ['exp_num','dataset','evaluation','strategy','fs (Hz)','type_norm','ws','ss','epochs','batch_size','time','acc_test','acc_test_F1score','num_testing_examples','acc_val','acc_val_F1score','num_validation_examples']

    wb = xlwt.Workbook()
    worksheet = wb.add_sheet('Mediapipe')

    for column, value in enumerate(columns, start=1):
        worksheet.write(0, column-1, value)
    for column, value in enumerate(data, start=1):
        worksheet.write(1, column-1, value)

    wb.save(output_excel_file)

else:
    workbook = copy(xlrd.open_workbook(output_excel_file))
    worksheet = workbook.get_sheet(0)

    data = [exp_num, dataset, approach, norm_type, network_type, final_points, num_classes, epochs, batch_size, accuracy_test,
            accuracy_test_F1_score, num_testing_examples_total, accuracy_train, accuracy_train_F1_score, num_training_examples_total, extraSettings]

    for column, value in enumerate(data, start=1):
        worksheet.write(exp_num, column-1, value)

    workbook.save(output_excel_file)
