#!/usr/bin/env python

import argparse
import numpy as np


def confusion_matrix(acc:float=0.995, subpop:float=1e-4, population:float=8.5e6) -> dict:
    """
    Generates confusion matrix and derived variables
    based on accuracy of detecting a fraction (subpop) of a population 

    Args:
        acc (float):    accuracy of detecting subpop (0-1)
        subpop (float): fraction of population in subpopulation (0-1)
        population (float): Total population size (absolute number)
        
    Returns:
        dict: derived variables as a dictionary
    """

    # Inputs
    population = int(population)
    print(f"\nInputs\n------------")
    print(f"Accuracy (%): {100*acc}")
    print(f"Subpopulation (%): {100*subpop}")
    print(f"Population size: {population}")
    print(f"Predicted subpopulation size: {int(subpop*population)}")

    # Check variables
    if acc > 1 or acc < 0:
        print("\nERROR: give valid accuracy (0 to 1).")
        return
    if subpop > 1 or subpop < 0:
        print("\nERROR: give valid subpop percent (0 to 1).")
        return
    if population < 1:
        print("\nERROR: cannot have zero or negative populations.")
        return

    # confusion matrix
    tp = np.rint(population*subpop*acc).astype(int)
    fp = np.rint(population*(1-acc)).astype(int)
    tn = np.rint(population*(1-subpop)*acc).astype(int)
    fn = np.rint(population*subpop).astype(int) - tp
    print(f"\nResults\n------------")
    print(f"True Positives (Power): {tp}")
    print(f"False Positives (Type I): {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives (Type II): {fn}")

    # derivations
    round_var = 4 # round vars to this place
    tpr = np.round((tp)/(tp+fn), round_var)
    fpr = np.round((fp)/(fp+tn), round_var)
    precision = np.round((tp)/(tp+fp), round_var)
    specificity = np.round((tn)/(tn+fp), round_var)
    fdr = np.round((fp)/(fp+tp), round_var)
    fscore = np.round(2*tp/(2*tp+fp+fn), round_var)
    print(f"\nDerivations\n------------")
    print(f"True Positive Rate (Recall): {tpr}")
    print(f"False Positive Rate: {fpr}")
    print(f"Precision: {precision}")
    print(f"Specificity: {specificity}")
    print(f"False Discovery Rate: {fdr}")
    print(f"F-Score: {fscore}")

    # output a dictionary of derived variables
    output = {
        'True_Positives':tp,
        'False_Positives':fp,
        'True_Negatives':tn,
        'False_Negatives':fn,
        'TPR':tpr,
        'FPR':fpr,
        'Precision':precision,
        'Specificity':specificity,
        'FDR':fdr,
        'FScore':fscore
    }

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--acc", type=float, default=0.995, help="Accuracy (0 -> 1)")
    parser.add_argument("-s", "--sub", type=float, default=1e-4, help="Subpop fraction (0 -> 1)")
    parser.add_argument("-p", "--pop", type=float, default=8.5e6, help="Pop size")
    args = parser.parse_args()
    accuracy = args.acc
    subpopulation = args.sub
    population = args.pop

    # main
    confusion_matrix(accuracy, subpopulation, population)
