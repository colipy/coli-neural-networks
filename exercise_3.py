"""
Exercise 3: Textgenerierung


"""
import archive as A
list_of_input_files = [
    'data/tx_DE_1_Schneewitchen.txt',
    'data/tx_DE_6_Das_Sandmännchen.txt',
    'data/tx_DE_2_Mathemathische_Grundlagen_1.txt',
    'data/tx_DE_7_Coli_Studiengang_Beschreibung.txt',
    'data/tx_DE_4_Oh_Tannenbaum.txt',
    'data/tx_DE_3_Der_Erlkönig.txt'
]

# Concatenate all of the above texts into a single training file, which can subsequently be used to train the network
A.create_training_file(list_of_input_files, output_file='data/training_file.txt')







